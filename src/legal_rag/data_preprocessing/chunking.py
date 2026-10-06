"""
Section-aware chunking for Australian legislation.

The chunker joins a document's pages into one text and cuts it at section
headings that start a line, such as "11  Calls to numbers on the Register".
It never drops text.  Every character of the document lands in exactly one
chunk, a fragment too short to stand alone joins its neighbour, and a section
too long for one chunk is split at subsection boundaries.  Page numbers are
tracked by character offset, so a section that crosses a page break stays in
one chunk and records both pages.  Each chunk takes its section number from
the heading that starts it.

Compilations on the Federal Register of Legislation also carry a table of
contents, running headers and footers on every page, and schedules whose
clause numbers restart at 1.  Contents entries and running lines are not
treated as headings, and schedule clauses are labelled with their schedule.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from typing import Dict, List, Optional, Sequence, Tuple

from legal_rag.data_preprocessing.pdf_parser import LegalDocument

# A section heading starts a line with the section number (11, 15A, 789FD),
# whitespace, and a capitalised title.  Anchoring the number to the start of
# the line is what keeps "has 30 days" and "penalty is 2" out.
SECTION_HEADING = re.compile(
    r"^[ \t]*(?P<number>\d{1,4}[A-Z]{0,4})[ \t]+(?P<title>[A-Z][^\n]*?)[ \t]*$"
)
SCHEDULE_HEADING = re.compile(
    r"^[ \t]*Schedule[ \t]+(?P<number>\d{1,3}[A-Z]?)"
    r"(?:(?:[ \t]*[\u2014\u2013-][ \t]*|[ \t]+)(?P<title>[A-Z][^\n]*))?[ \t]*$"
)
STRUCTURAL_HEADING = re.compile(
    r"^[ \t]*(?:Chapter|Part|Division|Subdivision)[ \t]+[0-9A-Z]+(?:-[0-9A-Z]+)?\b"
)
ENDNOTES_HEADING = re.compile(
    r"^[ \t]*Endnotes?\b(?:[ \t]+\d+)?(?:[ \t]*[\u2014\u2013-][^\n]*)?[ \t]*$"
)
SUBSECTION_START = re.compile(r"^[ \t]*\(\d+[A-Z]*\)", re.MULTILINE)
DOT_LEADER = re.compile(r"\.{4,}|(?:\.[ \t]){3,}|\u2026")
PAGE_NUMBER_ONLY = re.compile(r"^[ \t]*\d{1,4}[ \t]*$")
DEFINITION_TITLE = re.compile(
    r"\b(?:Definitions?|Dictionary|Meaning of|Interpretation)\b", re.IGNORECASE
)
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
DATE_TITLE = re.compile(rf"^(?:{_MONTHS})\b")
YEAR_NUMBER = re.compile(r"^(?:1[89]|20)\d\d$")

# Line kinds that carry no provision text of their own.  A heading followed
# only by these before the next heading is a contents entry, not a section.
_FILLER_KINDS = {"blank", "running", "filler", "structural", "schedule", "endnotes"}


class ChunkType(Enum):
    """What a chunk holds."""

    SECTION = "section"
    DEFINITION = "definition"
    SCHEDULE = "schedule"
    OTHER = "other"  # front matter, contents, and endnotes


@dataclass
class Chunk:
    """A contiguous span of a document's text."""

    text: str
    chunk_type: ChunkType
    section_number: Optional[str]
    page_numbers: List[int]
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class _Line:
    start: int
    end: int
    text: str


@dataclass
class _Segment:
    start: int
    end: int = 0
    sections: List[str] = field(default_factory=list)
    title: Optional[str] = None
    schedule: Optional[str] = None
    kind: str = "section"  # "section", "front", or "endnotes"


class LegalChunker:
    """
    Chunks a parsed Act at its section headings.

    Args:
        max_chunk_size: Longest chunk in characters.  A longer section is
            split at subsection boundaries, and every piece keeps the
            section number.
        min_chunk_size: Shortest chunk in non-space characters.  A shorter
            fragment joins the chunk before it, or the one after it when it
            comes first, so no text is lost.
    """

    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 40):
        if max_chunk_size <= min_chunk_size:
            raise ValueError("max_chunk_size must be larger than min_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: LegalDocument) -> List[Chunk]:
        """Split a document into chunks that partition its text."""
        page_texts = [page.text for page in document.pages]
        text = "\n".join(page_texts)
        if not text.strip():
            return []

        page_starts: List[int] = []
        offset = 0
        for page_text in page_texts:
            page_starts.append(offset)
            offset += len(page_text) + 1
        page_numbers = [page.page_number for page in document.pages]

        lines = _split_lines(text)
        running = _running_lines(page_texts)
        kinds = [_classify(line.text, running) for line in lines]
        accepted = _accept_headings(lines, kinds)
        segments = self._segments(text, lines, kinds, accepted)
        segments = self._merge_short(text, segments)

        chunks: List[Chunk] = []
        for segment in segments:
            pieces = self._split_long(text, segment.start, segment.end)
            for index, (start, end) in enumerate(pieces, 1):
                start, end = _trim(text, start, end)
                if start >= end:
                    continue
                chunks.append(
                    self._make_chunk(
                        text,
                        segment,
                        start,
                        end,
                        index,
                        len(pieces),
                        page_starts,
                        page_numbers,
                    )
                )
        return chunks

    def _segments(
        self,
        text: str,
        lines: Sequence[_Line],
        kinds: Sequence[str],
        accepted: Sequence[bool],
    ) -> List[_Segment]:
        """Cut the text at accepted headings, schedules, and the endnotes."""
        segments = [_Segment(start=0, kind="front")]
        body_started = False
        in_endnotes = False
        schedule: Optional[str] = None
        schedule_start: Optional[int] = None

        for index, line in enumerate(lines):
            kind = kinds[index]
            if accepted[index]:
                body_started = True
            if not body_started or in_endnotes:
                continue
            if kind == "endnotes":
                segments.append(_Segment(start=line.start, kind="endnotes"))
                in_endnotes = True
                continue
            if kind == "schedule":
                number = SCHEDULE_HEADING.match(line.text).group("number")
                # A running header repeats the current schedule's heading on
                # each of its pages.  Only a new schedule starts a new unit.
                if number != schedule:
                    schedule = number
                    schedule_start = line.start
                continue
            if not accepted[index]:
                continue
            match = SECTION_HEADING.match(line.text)
            if schedule_start is not None:
                start = schedule_start
                schedule_start = None
            else:
                start = _walk_back(lines, kinds, index)
            segments.append(
                _Segment(
                    start=start,
                    sections=[match.group("number")],
                    title=match.group("title").strip(),
                    schedule=schedule,
                )
            )

        for current, following in pairwise(segments):
            current.end = following.start
        segments[-1].end = len(text)
        return [s for s in segments if text[s.start : s.end].strip()]

    def _merge_short(self, text: str, segments: List[_Segment]) -> List[_Segment]:
        """Fold fragments shorter than min_chunk_size into a neighbour."""
        merged: List[_Segment] = []
        for segment in segments:
            if merged and _visible_length(text, segment) < self.min_chunk_size:
                previous = merged[-1]
                previous.end = segment.end
                previous.sections.extend(segment.sections)
            else:
                merged.append(segment)
        if len(merged) > 1 and _visible_length(text, merged[0]) < self.min_chunk_size:
            first, second = merged[0], merged[1]
            second.start = first.start
            second.sections = first.sections + second.sections
            merged = merged[1:]
        return merged

    def _split_long(self, text: str, start: int, end: int) -> List[Tuple[int, int]]:
        """Split [start, end) into pieces no longer than max_chunk_size."""
        pieces = []
        while end - start > self.max_chunk_size:
            limit = start + self.max_chunk_size
            cut = None
            # Prefer the last subsection that begins in the window, as long
            # as it leaves a first piece of a useful size.
            for match in SUBSECTION_START.finditer(text, start + 1, limit):
                if match.start() - start >= self.max_chunk_size // 4:
                    cut = match.start()
            if cut is None:
                newline = text.rfind("\n", start + 1, limit)
                if newline > start:
                    cut = newline + 1
            if cut is None:
                space = text.rfind(" ", start + 1, limit)
                cut = space + 1 if space > start else limit
            pieces.append((start, cut))
            start = cut
        pieces.append((start, end))
        return pieces

    def _make_chunk(
        self,
        text: str,
        segment: _Segment,
        start: int,
        end: int,
        piece: int,
        pieces: int,
        page_starts: Sequence[int],
        page_numbers: Sequence[int],
    ) -> Chunk:
        sections = [s for s in segment.sections if s]
        section = sections[0] if sections else None
        if segment.kind != "section" and section is None:
            chunk_type = ChunkType.OTHER
        elif segment.schedule:
            chunk_type = ChunkType.SCHEDULE
        elif segment.title and DEFINITION_TITLE.search(segment.title):
            chunk_type = ChunkType.DEFINITION
        else:
            chunk_type = ChunkType.SECTION

        first_page = bisect.bisect_right(page_starts, start) - 1
        last_page = bisect.bisect_right(page_starts, end - 1) - 1
        metadata = {
            "heading": segment.title or "",
            "sections": sections,
            "schedule": segment.schedule,
            "part_of_section": f"{piece} of {pieces}" if pieces > 1 else None,
        }
        if segment.kind == "endnotes":
            metadata["region"] = "endnotes"
        return Chunk(
            text=text[start:end],
            chunk_type=chunk_type,
            section_number=section,
            page_numbers=list(page_numbers[first_page : last_page + 1]),
            start_char=start,
            end_char=end,
            metadata=metadata,
        )


def _split_lines(text: str) -> List[_Line]:
    lines = []
    offset = 0
    for raw in text.split("\n"):
        lines.append(_Line(start=offset, end=offset + len(raw), text=raw))
        offset += len(raw) + 1
    return lines


def _normalise_running(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def _running_lines(page_texts: Sequence[str], edge: int = 3) -> set:
    """Lines that repeat at the top or bottom of many pages."""
    if len(page_texts) < 3:
        return set()
    counts: Counter = Counter()
    for page_text in page_texts:
        lines = [line for line in page_text.split("\n") if line.strip()]
        edges = lines[:edge] + lines[-edge:]
        counts.update({_normalise_running(line) for line in edges})
    threshold = max(3, math.ceil(0.3 * len(page_texts)))
    return {line for line, count in counts.items() if count >= threshold}


def _classify(line: str, running: set) -> str:
    if not line.strip():
        return "blank"
    if _normalise_running(line) in running:
        return "running"
    if DOT_LEADER.search(line) or PAGE_NUMBER_ONLY.match(line):
        return "filler"
    if ENDNOTES_HEADING.match(line):
        return "endnotes"
    if SCHEDULE_HEADING.match(line):
        return "schedule"
    if STRUCTURAL_HEADING.match(line):
        return "structural"
    match = SECTION_HEADING.match(line)
    if match and _plausible_heading(match):
        return "heading"
    return "body"


def _plausible_heading(match: re.Match) -> bool:
    number, title = match.group("number"), match.group("title").strip()
    if YEAR_NUMBER.match(number) or DATE_TITLE.match(title):
        return False
    if len(title) > 200 or title[-1] in ",;:":
        return False
    return True


def _title_key(title: str) -> str:
    title = DOT_LEADER.split(title)[0]
    title = re.sub(r"\s+\d{1,4}\s*$", "", title)
    return re.sub(r"\s+", " ", title).strip().lower()


def _accept_headings(lines: Sequence[_Line], kinds: Sequence[str]) -> List[bool]:
    """
    Decide which heading-shaped lines really start a section.

    A contents entry is followed by nothing but other entries, leader dots,
    page numbers, and running lines, so a heading with no provision text
    after it is rejected.  A heading that appears again later in the body is
    also rejected when the next meaningful line after it is not provision
    text, which catches a contents entry wrapped over two lines, or when it
    closes a run of rejected entries, which catches the last entry before
    the Act's long title.
    """
    candidates = [i for i, kind in enumerate(kinds) if kind == "heading"]
    accepted = [False] * len(lines)
    if not candidates:
        return accepted

    def body_between(first: int, last: int) -> bool:
        return any(kinds[i] not in _FILLER_KINDS for i in range(first + 1, last))

    def next_line_is_body(index: int) -> bool:
        for kind in kinds[index + 1 :]:
            if kind not in ("blank", "running"):
                return kind == "body"
        return False

    keys = []
    for index in candidates:
        match = SECTION_HEADING.match(lines[index].text)
        keys.append((match.group("number"), _title_key(match.group("title"))))

    def recurs_later(position: int) -> bool:
        number, key = keys[position]
        return bool(key) and any(
            later_number == number
            and (later_key.startswith(key) or key.startswith(later_key))
            for later_number, later_key in keys[position + 1 :]
        )

    previous: Optional[int] = None
    previous_ok = True
    for position, index in enumerate(candidates):
        following = (
            candidates[position + 1] if position + 1 < len(candidates) else len(lines)
        )
        ok = body_between(index, following)
        if ok and recurs_later(position):
            closes_rejected_run = (
                previous is not None
                and not previous_ok
                and not body_between(previous, index)
            )
            if closes_rejected_run or not next_line_is_body(index):
                ok = False
        accepted[index] = ok
        previous, previous_ok = index, ok
    return accepted


def _walk_back(lines: Sequence[_Line], kinds: Sequence[str], index: int) -> int:
    """Start a section's chunk at the Part or Division heading right above it."""
    start = lines[index].start
    position = index - 1
    while position >= 0 and kinds[position] in ("blank", "structural"):
        if kinds[position] == "structural":
            start = lines[position].start
        position -= 1
    return start


def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _visible_length(text: str, segment: _Segment) -> int:
    return len("".join(text[segment.start : segment.end].split()))
