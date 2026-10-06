"""
Find the provisions a chunk cites and the terms it defines.

Retrieval uses both.  A retrieved section pulls in the sections it cites, and
a defined term that appears in the question or in a retrieved section pulls in
the chunk that defines it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from legal_rag.citations import parse_references

# A definition entry starts a line with the term, as Commonwealth Acts set
# them out: "Register means the Do Not Call Register." or
# "commercial electronic message has the meaning given by section 6."
DEFINITION_ENTRY = re.compile(
    r"^[ \t]*(?P<term>[A-Za-z][A-Za-z0-9'\u2019 -]{0,60}?)[ \t]+"
    r"(?P<verb>means|includes|has[ \t]+the[ \t]+(?:same[ \t]+)?meaning[ \t]+given"
    r"[ \t]+(?:by|in))\b(?P<rest>[^\n]*)",
    re.MULTILINE,
)
_NOT_TERMS = {
    "a", "an", "and", "any", "each", "example", "for", "if", "in", "it", "note",
    "or", "paragraph", "section", "subsection", "such", "that", "the", "this",
    "where", "which", "who",
}  # fmt: skip


@dataclass
class Definition:
    """A term defined in the text, and where its meaning is given."""

    term: str
    definition_text: str
    target_section: Optional[str] = None
    target_schedule: Optional[str] = None


class MetadataExtractor:
    """Extracts definitions and cross-references from a chunk of an Act."""

    def extract_definitions(self, text: str) -> List[Definition]:
        """Return the terms defined in text, first definition of each term."""
        definitions: List[Definition] = []
        seen = set()
        for match in DEFINITION_ENTRY.finditer(text):
            term = re.sub(r"\s+", " ", match.group("term")).strip()
            words = term.split()
            if not words or words[0].lower() in _NOT_TERMS or len(words) > 6:
                continue
            if term.lower() in seen:
                continue
            seen.add(term.lower())
            definition = Definition(
                term=term, definition_text=match.group("rest").strip()
            )
            if match.group("verb").startswith("has"):
                for reference in parse_references(match.group("rest"))[:1]:
                    if reference.act is None:
                        definition.target_section = reference.section
                        definition.target_schedule = reference.schedule
            definitions.append(definition)
        return definitions

    def extract_cross_references(
        self, text: str, own_sections: Iterable[str] = ()
    ) -> List[str]:
        """
        Return the provisions of the same Act that text cites, in order.

        Sections come back as their number ("6") and schedules as
        "Schedule 2".  A reference followed by the name of an Act, such as
        "section 7 of the Telecommunications Act 1997", points outside the
        document and is left out, as are the chunk's own sections.
        """
        own = {section.upper() for section in own_sections if section}
        targets: List[str] = []
        for reference in parse_references(text):
            if reference.act is not None:
                continue
            if reference.section is not None:
                target = reference.section
                if target in own:
                    continue
            else:
                target = f"Schedule {reference.schedule}"
            if target not in targets:
                targets.append(target)
        return targets
