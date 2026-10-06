"""
Parse references to provisions, such as "s 11(1)", "sections 16 to 18", or
"Schedule 2", out of statute text or a model's answer.

The same parser feeds two jobs: cross-reference expansion at retrieval time,
where a retrieved section's references pull in the sections it cites, and the
citation verifier, which checks an answer's references against the text that
was actually retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

_NUMBER = r"\d{1,4}[A-Z]{0,4}"
_SUBDIVISIONS = r"(?:\([0-9A-Za-z]{1,6}\))*"

# "s 11", "s. 11(1)", "ss 16-17", "section 6", "subsections 11(1) and (2)",
# "paragraph 7(a)".  The leading word boundary keeps "has 30" out.
SECTION_REFERENCE = re.compile(
    rf"(?<!['\u2019])\b(?P<kind>sections?|subsections?|paragraphs?|subparagraphs?|ss?)\.?[ \t]*"
    rf"(?P<first>{_NUMBER}){_SUBDIVISIONS}"
    rf"(?P<rest>(?:[ \t]*(?:,|-|\u2013|to|and|or)[ \t]*(?:{_NUMBER})?{_SUBDIVISIONS})*)",
    re.IGNORECASE,
)
SCHEDULE_REFERENCE = re.compile(
    r"\b(?:Schedules?|Sch)\.?[ \t]+(?P<number>\d{1,3}[A-Z]?)\b", re.IGNORECASE
)
# "of the Spam Act 2003" right after a reference names the Act it points to.
ACT_AFTER = re.compile(
    r"^[ \t,]*(?:of|in|under)[ \t]+(?:the[ \t]+)?"
    r"(?P<act>(?:[A-Z][\w'()-]*[ \t]+){1,10}?(?:Act|Regulations?|Rules)"
    r"(?:[ \t]+\d{4})?)"
)
_LIST_ITEM = re.compile(rf"(?P<sep>,|-|\u2013|to|and|or)[ \t]*(?P<number>{_NUMBER})?")


@dataclass(frozen=True)
class SectionReference:
    """One provision a piece of text points to."""

    text: str
    section: Optional[str]
    schedule: Optional[str] = None
    act: Optional[str] = None
    start: int = 0
    end: int = 0


def parse_references(text: str) -> List[SectionReference]:
    """Return every section and schedule reference in text, in order."""
    references: List[SectionReference] = []
    for match in SECTION_REFERENCE.finditer(text):
        act = _act_after(text, match.end())
        for number in _expand(match.group("first"), match.group("rest")):
            references.append(
                SectionReference(
                    text=match.group(0).strip(),
                    section=number.upper(),
                    act=act,
                    start=match.start(),
                    end=match.end(),
                )
            )
    for match in SCHEDULE_REFERENCE.finditer(text):
        references.append(
            SectionReference(
                text=match.group(0),
                section=None,
                schedule=match.group("number").upper(),
                act=_act_after(text, match.end()),
                start=match.start(),
                end=match.end(),
            )
        )
    references.sort(key=lambda reference: reference.start)
    return references


def normalise_act(name: str) -> str:
    """Lower-case an Act name and drop "the", jurisdiction tags, and spacing."""
    name = re.sub(r"\((?:Cth|Vic|NSW|Qld|SA|WA|Tas|ACT|NT)\)", "", name)
    name = re.sub(r"\bthe\b", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def _act_after(text: str, position: int) -> Optional[str]:
    match = ACT_AFTER.match(text[position : position + 120])
    return match.group("act").strip() if match else None


def _expand(first: str, rest: str) -> List[str]:
    """Turn "16" plus " to 18" into 16, 17, 18, and "16" plus ", 17" into 16, 17."""
    numbers = [first]
    previous = first
    for item in _LIST_ITEM.finditer(rest or ""):
        number = item.group("number")
        if not number:
            continue
        if item.group("sep") in ("to", "-", "\u2013"):
            numbers.extend(_range(previous, number))
        else:
            numbers.append(number)
        previous = number
    seen = []
    for number in numbers:
        if number not in seen:
            seen.append(number)
    return seen


def _range(low: str, high: str) -> List[str]:
    if low.isdigit() and high.isdigit() and 0 < int(high) - int(low) <= 20:
        return [str(n) for n in range(int(low) + 1, int(high) + 1)]
    return [high]


@dataclass
class CitationCheck:
    """How each provision an answer cites relates to what was retrieved."""

    supported: List[str]
    outside_retrieved: List[str]
    not_in_corpus: List[str]

    def as_dict(self) -> dict:
        return {
            "supported": self.supported,
            "outside_retrieved": self.outside_retrieved,
            "not_in_corpus": self.not_in_corpus,
        }

    def lines(self) -> List[str]:
        """A short report for the chat output."""
        if not (self.supported or self.outside_retrieved or self.not_in_corpus):
            return ["Section references: none in the answer."]
        lines = [
            f"Section references: {len(self.supported)} found in the retrieved text."
        ]
        if self.outside_retrieved:
            lines.append(
                "  Not in the retrieved text, though in the indexed Acts: "
                + ", ".join(self.outside_retrieved)
            )
        if self.not_in_corpus:
            lines.append("  Not in the indexed Acts: " + ", ".join(self.not_in_corpus))
        return lines


def verify_citations(
    answer: str, retrieved: Sequence, corpus: Sequence
) -> CitationCheck:
    """
    Check every section and schedule an answer cites.

    retrieved holds the chunks the model was given and corpus holds every
    indexed chunk.  Both need document_name, section_number, and metadata
    with "sections", "schedule", and "document_title".  A reference that
    names an Act is matched only against that Act.  The check is purely
    textual: it shows whether a cited provision was in front of the model,
    not whether the answer reads it correctly.
    """
    retrieved_keys = _provision_keys(retrieved)
    corpus_keys = _provision_keys(corpus)
    titles = _document_titles(list(corpus) + list(retrieved))

    check = CitationCheck([], [], [])
    for reference in parse_references(answer):
        label = (
            f"s {reference.section}"
            if reference.section
            else f"Schedule {reference.schedule}"
        )
        if reference.act:
            label += f" of the {reference.act}"
        if label in check.supported + check.outside_retrieved + check.not_in_corpus:
            continue
        documents = _documents_named(reference.act, titles)
        keys = {(doc, reference.schedule, reference.section) for doc in documents}
        if keys & retrieved_keys:
            check.supported.append(label)
        elif keys & corpus_keys:
            check.outside_retrieved.append(label)
        else:
            check.not_in_corpus.append(label)
    return check


def _provision_keys(chunks: Sequence) -> set:
    keys = set()
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        schedule = metadata.get("schedule")
        doc = chunk.document_name
        if schedule:
            keys.add((doc, str(schedule).upper(), None))
            continue  # clause numbers inside a schedule are not sections
        for section in metadata.get("sections") or [chunk.section_number]:
            if section:
                keys.add((doc, None, str(section).upper()))
    return keys


def _document_titles(chunks: Sequence) -> dict:
    titles = {}
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        title = metadata.get("document_title") or chunk.document_name.replace("_", " ")
        titles.setdefault(chunk.document_name, normalise_act(title))
    return titles


def _documents_named(act: Optional[str], titles: dict) -> List[str]:
    if act is None:
        return list(titles)
    wanted = normalise_act(act)
    return [doc for doc, title in titles.items() if _same_act(wanted, title)]


def _same_act(first: str, second: str) -> bool:
    """Equal names, or equal once a year that only one of them gives is dropped."""
    year = re.compile(r"\s+\d{4}$")
    if first == second:
        return True
    if bool(year.search(first)) != bool(year.search(second)):
        return year.sub("", first) == year.sub("", second)
    return False
