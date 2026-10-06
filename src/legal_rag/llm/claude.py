"""
Claude answer backend, grounded in the retrieved text through document citations.

Each retrieved chunk goes to Claude as its own plain-text document block, titled
with the Act and section, with citations enabled.  Claude's answer comes back
in text blocks, and each cited block carries the exact span of the document it
relies on, so every citation in the answer points at retrieved text.

This is a user-facing chat path, so requests opt into server-side fallbacks:
if Claude Opus 5.5's safety classifiers decline a benign legal question, the
API reruns it on the model Anthropic recommends for that refusal category
instead of returning a refusal.  The answer records when that happened.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ClaudeError(RuntimeError):
    """A Claude API failure, explained for the person at the chat prompt."""


@dataclass
class Citation:
    """One span of a retrieved document that the answer relies on."""

    source_index: int  # zero-based position in the sources sent with the question
    document_title: Optional[str]
    cited_text: str
    start_char: int
    end_char: int


@dataclass
class ClaudeAnswer:
    """What came back for one question."""

    text: str
    citations: List[Citation] = field(default_factory=list)
    stop_reason: Optional[str] = None
    model: Optional[str] = None
    served_by_fallback: bool = False
    refusal_category: Optional[str] = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class ClaudeLLM:
    """Answers questions with Claude through the official anthropic SDK."""

    def __init__(
        self,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        max_tokens: int = 16000,
        client: Optional[anthropic.Anthropic] = None,
    ):
        """
        Args:
            model: Claude model ID.  Falls back to CLAUDE_MODEL, then
                claude-opus-5-5.
            effort: How much the model thinks before answering, one of low,
                medium, high, xhigh, or max.  Falls back to CLAUDE_EFFORT, then
                high.  Opus 5.5 always thinks adaptively, so effort is the
                setting that trades depth for cost and latency.
            max_tokens: Ceiling on the answer, thinking included.
            client: An anthropic.Anthropic client.  Built from the environment
                (ANTHROPIC_API_KEY) when not given.
        """
        self.model = model or os.getenv("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
        self.effort = effort or os.getenv("CLAUDE_EFFORT") or DEFAULT_EFFORT
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def generate(
        self,
        prompt: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> str:
        """Answer a prompt with no documents, for chat with retrieval off."""
        answer = self._send(
            [*(history or []), {"role": "user", "content": prompt}], system
        )
        return answer.text

    def answer_with_documents(
        self,
        question: str,
        sources: Sequence[Any],
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> ClaudeAnswer:
        """
        Answer a question from retrieved chunks, citing them.

        sources are retrieval results with text, document_name, section, and
        metadata, in the order the citations will index them.
        """
        content: List[Dict[str, Any]] = [document_block(source) for source in sources]
        content.append({"type": "text", "text": question})
        return self._send(
            [*(history or []), {"role": "user", "content": content}], system
        )

    def _send(
        self, messages: List[Dict[str, Any]], system: Optional[str]
    ) -> ClaudeAnswer:
        request: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            # Opus 5.5 defaults to medium effort, so set it explicitly.
            # No temperature, top_p, top_k, or thinking: Opus 5.5 rejects the
            # sampling parameters and always thinks adaptively.
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        try:
            response = self.client.beta.messages.create(**request)
        except anthropic.AuthenticationError as error:
            raise ClaudeError(
                "The Claude API rejected the key.  Check ANTHROPIC_API_KEY."
            ) from error
        except anthropic.NotFoundError as error:
            raise ClaudeError(
                f"The Claude API does not know the model {self.model}."
            ) from error
        except anthropic.RateLimitError as error:
            raise ClaudeError(
                "The Claude API is rate limiting this key.  Try again shortly."
            ) from error
        except anthropic.APIStatusError as error:
            raise ClaudeError(
                f"The Claude API returned {error.status_code}: {error.message}"
            ) from error
        except anthropic.APIConnectionError as error:
            raise ClaudeError("Could not reach the Claude API.") from error
        return read_response(response)


def document_block(source: Any) -> Dict[str, Any]:
    """A retrieved chunk as a plain-text document block with citations on."""
    block: Dict[str, Any] = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": source.text},
        "title": source_title(source),
        "citations": {"enabled": True},
    }
    reason = getattr(source, "expansion_reason", None)
    if reason:
        block["context"] = f"Added after retrieval because it {reason}."
    return block


def source_title(source: Any) -> str:
    """Name a chunk by its Act and provision, such as "Spam Act 2003, s 16"."""
    metadata = getattr(source, "metadata", None) or {}
    act = metadata.get("document_title") or source.document_name
    section = getattr(source, "section_number", None)
    schedule = metadata.get("schedule")
    if schedule and section:
        label = f"Schedule {schedule}, clause {section}"
    elif section:
        label = f"s {section}"
    else:
        label = "text outside any section"
    part = metadata.get("part_of_section")
    return f"{act}, {label}" + (f" (part {part})" if part else "")


def read_response(response: Any) -> ClaudeAnswer:
    """Turn a Messages API response into an answer, checking stop_reason first."""
    served_by_fallback = any(
        getattr(entry, "type", None) == "fallback_message"
        for entry in (getattr(response.usage, "iterations", None) or [])
    )
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        return ClaudeAnswer(
            text=(
                "Claude declined to answer this question"
                + (f" (refusal category: {category})." if category else ".")
            ),
            stop_reason="refusal",
            model=response.model,
            served_by_fallback=served_by_fallback,
            refusal_category=category,
        )

    parts: List[str] = []
    citations: List[Citation] = []
    for block in response.content:
        if block.type == "fallback":
            served_by_fallback = True
        if block.type != "text":
            continue
        parts.append(block.text)
        cited = []
        for citation in block.citations or []:
            if citation.type != "char_location":
                continue
            citations.append(
                Citation(
                    source_index=citation.document_index,
                    document_title=citation.document_title,
                    cited_text=citation.cited_text,
                    start_char=citation.start_char_index,
                    end_char=citation.end_char_index,
                )
            )
            if citation.document_index not in cited:
                cited.append(citation.document_index)
        if cited:
            parts.append("".join(f"[{index + 1}]" for index in cited))

    text = "".join(parts).strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n[The answer was cut off at the max_tokens limit.]"
    elif response.stop_reason not in ("end_turn", "stop_sequence"):
        logger.warning("Unexpected stop_reason from Claude: %s", response.stop_reason)
    return ClaudeAnswer(
        text=text,
        citations=citations,
        stop_reason=response.stop_reason,
        model=response.model,
        served_by_fallback=served_by_fallback,
    )
