"""
Answer generation over retrieved legislation.

Two local backends live here: GemmaLLM, which loads a checkpoint with
transformers, and OpenAIChatLLM, which talks to any OpenAI-compatible server.
get_llm() picks one from the environment, and LegalChatBot joins retrieval to
whichever backend it is given.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

# `torch` and `transformers` are imported lazily inside GemmaLLM so the
# retrieval and metrics path can be imported without the multi-gigabyte ML
# stack.  See tests/test_imports.py.

logger = logging.getLogger(__name__)

DEFAULT_GEMMA_MODEL = "google/gemma-4-12B-it"


def describe_corpus(titles: Sequence[str]) -> str:
    """Name the indexed documents in a phrase, such as "the A and the B"."""
    unique = list(dict.fromkeys(t.strip() for t in titles if t and t.strip()))
    if not unique:
        return "the indexed documents"
    if len(unique) == 1:
        return f"the {unique[0]}"
    return ", ".join(f"the {t}" for t in unique[:-1]) + f" and the {unique[-1]}"


def build_system_prompt(corpus: str = "the indexed documents") -> str:
    """The system prompt shared by every backend, naming what was indexed."""
    return f"""You are a legal research assistant for Australian legislation.
You answer questions using extracts from {corpus}.

Guidelines:
1. Answer only from the provided extracts.
2. Cite the section number of every provision you rely on, for example "s 11(1)".
3. If the extracts do not contain the answer, say "I cannot find this information in the provided documents".
4. For complex questions, set out your reasoning step by step.
5. Be precise.  Say when a provision has exceptions or conditions.
6. If several provisions apply, cite each of them.
7. Explain defined terms using the definitions in the extracts."""


def _messages(
    prompt: str, history: Optional[Sequence[Dict[str, str]]], system: Optional[str]
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system or build_system_prompt()},
        *(history or []),
        {"role": "user", "content": prompt},
    ]


class GemmaLLM:
    """
    Local generation with a Gemma checkpoint through transformers.

    The checkpoint must be in transformers format.  A GGUF file needs a
    llama.cpp-style server instead, reached through OpenAIChatLLM.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        """
        Args:
            model_name: Hugging Face model ID or local path.  Falls back to
                GEMMA_MODEL, then google/gemma-4-12B-it.
            device: 'cuda', 'mps', or 'cpu'.  Detected when not given.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.  0 means greedy decoding.

        Raises:
            ValueError: The model name is a GGUF checkpoint.
            RuntimeError: transformers could not load the tokenizer or model.
        """
        model_name = model_name or os.getenv("GEMMA_MODEL") or DEFAULT_GEMMA_MODEL
        if "gguf" in model_name.lower():
            raise ValueError(
                f"{model_name} is a GGUF checkpoint, which transformers cannot load "
                "by name.  Serve it with llama.cpp or another OpenAI-compatible "
                "server and set CHAT_BASE_URL, or set GEMMA_MODEL to a "
                "transformers checkpoint."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or self._get_device()
        self.max_tokens = max_tokens
        self.temperature = temperature

        logger.info("Loading %s on %s", model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load {model_name} with transformers: {error}"
            ) from error

    def _get_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> str:
        """Answer one prompt, after the system prompt and any earlier turns."""
        return self.chat(
            _messages(prompt, history, system),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate the next assistant turn for a list of messages."""
        import torch

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature

        # Some Gemma chat templates reject a system role, so the system
        # prompt is folded into the first user turn.
        turns = [dict(m) for m in messages if m["role"] != "system"]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system and turns:
            turns[0]["content"] = f"{system}\n\n{turns[0]['content']}"

        input_ids = self.tokenizer.apply_chat_template(
            turns, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            options["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(
                input_ids, pad_token_id=self.tokenizer.eos_token_id, **options
            )
        new_tokens = output[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAIChatLLM:
    """
    Chat client for any OpenAI-compatible endpoint (for example a local
    llama.cpp, Ollama, or vLLM server exposing /v1/chat/completions).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        """
        Args:
            base_url: OpenAI-compatible base URL.  Falls back to the
                CHAT_BASE_URL and OPENAI_CHAT_BASE_URL environment variables.
            model: Model name to request.  Falls back to CHAT_MODEL and
                OPENAI_CHAT_MODEL, then 'local'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        """
        self.base_url = (
            base_url or os.getenv("CHAT_BASE_URL") or os.getenv("OPENAI_CHAT_BASE_URL")
        )
        self.model = (
            model
            or os.getenv("CHAT_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
            or "local"
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Local servers ignore the key; the SDK still requires a non-empty one.
        api_key = os.getenv("OPENAI_API_KEY") or "local"
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        logger.info(
            "Initialized OpenAI-compatible chat client -> %s (%s)",
            self.base_url,
            self.model,
        )

    def chat(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> str:
        """Send a chat completion and return the assistant text."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> str:
        """Answer one prompt, after the system prompt and any earlier turns."""
        return self.chat(_messages(prompt, history, system), max_tokens=max_tokens)


def get_llm(**kwargs):
    """
    Build the chat client from the environment.

    If an OpenAI-compatible endpoint is configured (CHAT_BASE_URL or
    OPENAI_CHAT_BASE_URL), return OpenAIChatLLM.  Otherwise fall back to the
    local transformers-backed GemmaLLM.
    """
    endpoint = os.getenv("CHAT_BASE_URL") or os.getenv("OPENAI_CHAT_BASE_URL")
    if endpoint:
        return OpenAIChatLLM(**kwargs)
    return GemmaLLM(**kwargs)


class LegalChatBot:
    """
    Joins retrieval to an answer backend for questions about the indexed Acts.

    Each answer is generated after the last few question and answer turns, so
    a follow-up question can refer back.  Retrieval uses the current question
    only.
    """

    def __init__(
        self,
        retriever,
        llm,
        max_retrieved: int = 10,
        corpus_description: str = "the indexed documents",
        history_turns: int = 3,
    ):
        """
        Args:
            retriever: HybridRetriever instance.
            llm: Any backend from get_llm().
            max_retrieved: Chunks to retrieve before expansion.
            corpus_description: Names the indexed documents in the system
                prompt, for example "the Spam Act 2003".
            history_turns: Earlier question and answer pairs sent with each
                new question.
        """
        self.retriever = retriever
        self.llm = llm
        self.max_retrieved = max_retrieved
        self.corpus_description = corpus_description
        self.system_prompt = build_system_prompt(corpus_description)
        self.history_turns = history_turns
        self.conversation_history: List[Dict[str, str]] = []

    def _build_context(self, query: str) -> Tuple[str, List[Dict]]:
        """Retrieve for the query and format the chunks as numbered extracts."""
        results = self.retriever.retrieve(query, k=self.max_retrieved, expand=True)

        context_parts = []
        sources = []
        for i, result in enumerate(results, 1):
            section_info = (
                f"Section {result.section_number}"
                if result.section_number
                else "Unknown section"
            )
            if result.expansion:
                section_info += f" (added because it {result.expansion_reason})"
            context_parts.append(
                f"[{i}] {result.document_name} - {section_info}\n"
                f"Type: {result.chunk_type}\nText: {result.text}"
            )
            sources.append(
                {
                    "chunk_id": result.chunk_id,
                    "section": result.section_number,
                    "document": result.document_name,
                    "expansion": result.expansion,
                    "score": result.combined_score,
                }
            )
        return "\n\n".join(context_parts), sources

    def recent_history(self) -> List[Dict[str, str]]:
        """The last history_turns question and answer pairs."""
        if self.history_turns <= 0:
            return []
        return self.conversation_history[-2 * self.history_turns :]

    def answer(self, query: str, use_rag: bool = True) -> Dict:
        """
        Answer a question with retrieval, or directly when use_rag is False.

        Returns a dict with 'response', 'sources', and 'metadata'.
        """
        sources: List[Dict] = []
        if use_rag:
            context, sources = self._build_context(query)
            prompt = f"""Answer the following question using only the extracts below.

Question: {query}

Extracts:
{context}

Instructions:
1. Answer only from the extracts.
2. Cite specific section numbers.
3. If the extracts do not contain the answer, say so.
4. Be precise.

Answer:"""
        else:
            prompt = f"""Answer the following question about Australian legislation:

{query}

Answer:"""

        response = self.llm.generate(
            prompt, history=self.recent_history(), system=self.system_prompt
        )

        self.conversation_history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        )
        return {
            "response": response,
            "sources": sources,
            "metadata": {
                "query": query,
                "use_rag": use_rag,
                "conversation_length": len(self.conversation_history),
            },
        }

    def clear_history(self):
        """Forget earlier turns."""
        self.conversation_history = []
