"""The chunker cuts at real section headings and never loses text.

The fixtures are made-up Acts shaped like compilations on the Federal Register
of Legislation: a contents list, running headers and footers, section headings
that start a line, and schedules whose clause numbers restart at 1.
"""

from itertools import pairwise

from legal_rag.data_preprocessing.chunking import ChunkType, LegalChunker
from legal_rag.data_preprocessing.pdf_parser import DocumentPage, LegalDocument


def _doc(*pages):
    return LegalDocument(
        title="Toy Act 2024",
        document_type="Act",
        pages=[DocumentPage(i, text, 0.0, 0.0) for i, text in enumerate(pages, 1)],
    )


def _visible(text):
    return "".join(text.split())


TOY_PAGE_1 = """Toy Act 2024
37 Notice of a claim
(1) A person who makes a claim has 30 days to give notice of it to the
employer.
(2) The penalty is 2 penalty units for each day the notice is late.
38 Form of notice
A notice under section 37 must be in writing and signed by the person.
39 Withdrawal of a claim
(1) A claim may be withdrawn at any time"""

TOY_PAGE_2 = """by written notice to the Commissioner.
(2) A withdrawn claim cannot be made again.
40 Calls to registered numbers
(1) A person must not make a call to a number on the Register."""


def _chunks(*pages, **kwargs):
    return LegalChunker(**kwargs).chunk_document(_doc(*pages))


def test_numbers_in_running_text_are_not_section_boundaries():
    chunks = _chunks(TOY_PAGE_1, TOY_PAGE_2)
    sections = [c.section_number for c in chunks]
    assert "30" not in sections and "2" not in sections
    first = next(c for c in chunks if c.section_number == "37")
    assert "has 30 days" in first.text and "penalty is 2" in first.text


def test_each_real_heading_starts_its_own_chunk():
    chunks = _chunks(TOY_PAGE_1, TOY_PAGE_2)
    assert [c.section_number for c in chunks] == ["37", "38", "39", "40"]
    for chunk in chunks[1:]:
        assert chunk.text.startswith(f"{chunk.section_number} ")
    assert chunks[2].metadata["heading"] == "Withdrawal of a claim"


def test_no_text_is_lost_and_chunks_partition_the_document():
    pages = (TOY_PAGE_1, TOY_PAGE_2)
    full = "\n".join(pages)
    chunks = _chunks(*pages)
    assert _visible("".join(c.text for c in chunks)) == _visible(full)
    assert full[: chunks[0].start_char].strip() == ""
    for before, after in pairwise(chunks):
        assert full[before.end_char : after.start_char].strip() == ""
    for chunk in chunks:
        assert full[chunk.start_char : chunk.end_char] == chunk.text


def test_a_section_across_a_page_break_stays_whole_with_both_pages():
    chunks = _chunks(TOY_PAGE_1, TOY_PAGE_2)
    withdrawal = [c for c in chunks if c.section_number == "39"]
    assert len(withdrawal) == 1
    assert "at any time\nby written notice" in withdrawal[0].text
    assert withdrawal[0].page_numbers == [1, 2]


def test_page_numbers_follow_character_offsets():
    chunks = _chunks(TOY_PAGE_1, TOY_PAGE_2)
    pages = {c.section_number: c.page_numbers for c in chunks}
    assert pages == {"37": [1], "38": [1], "39": [1, 2], "40": [2]}


def test_a_short_fragment_joins_its_neighbour_instead_of_being_dropped():
    page = """1 Short title
This Act may be cited as the Toy Act 2024 for every purpose.
2 Repeal
See Act No. 9.
3 Application
This Act applies to every call made in Australia after commencement."""
    chunks = _chunks(page)
    assert [c.section_number for c in chunks] == ["1", "3"]
    assert "2 Repeal\nSee Act No. 9." in chunks[0].text
    assert chunks[0].metadata["sections"] == ["1", "2"]
    assert _visible("".join(c.text for c in chunks)) == _visible(page)


def test_a_long_section_is_split_at_subsections_and_keeps_its_number():
    body = "\n".join(
        f"({n}) The holder of a licence must comply with condition {n} of the "
        "licence at all times while the licence is in force."
        for n in range(1, 13)
    )
    page = f"5 Conditions of a licence\n{body}\n6 Cancellation\nThe licence may be cancelled."
    chunks = _chunks(page, max_chunk_size=400)
    fives = [c for c in chunks if c.section_number == "5"]
    assert len(fives) > 1
    assert all(len(c.text) <= 400 for c in fives)
    assert all(c.text.startswith(("5 ", "(")) for c in fives)
    assert fives[0].metadata["part_of_section"] == f"1 of {len(fives)}"
    assert _visible("".join(c.text for c in chunks)) == _visible(page)


def test_contents_entries_and_running_lines_are_not_headings():
    pages = [
        """Toy Act 2024
Contents
1 Short title ........................................ 1
2 Definitions ........................................ 1
3 Calls to numbers on the Register
.................................................... 2
Schedule 1—Exempt calls 3
1 Charities ........................................ 3""",
        """An Act about calls, and for related purposes
Part 1—Preliminary
1 Short title
This Act may be cited as the Toy Act 2024 for every purpose.
2 Definitions
In this Act:
Register means the Do Not Call Register.
Toy Act 2024 2""",
        """3 Calls to numbers on the Register
(1) A person must not make a call to a number on the Register.
(2) Subsection (1) does not apply to an exempt call (see Schedule 1).
Toy Act 2024 3""",
        """Schedule 1—Exempt calls
1 Charities
A call made by a registered charity is an exempt call.
2 Emergency services
A call made by an emergency service is an exempt call.
Toy Act 2024 4""",
    ]
    chunks = _chunks(*pages)
    labelled = [(c.section_number, c.metadata["schedule"]) for c in chunks]
    assert labelled == [
        (None, None),
        ("1", None),
        ("2", None),
        ("3", None),
        ("1", "1"),
        ("2", "1"),
    ]
    assert chunks[0].chunk_type is ChunkType.OTHER
    assert "Contents" in chunks[0].text and "Part 1" not in chunks[0].text
    assert chunks[1].text.startswith("Part 1—Preliminary\n1 Short title")
    assert chunks[2].chunk_type is ChunkType.DEFINITION
    assert chunks[4].chunk_type is ChunkType.SCHEDULE
    assert chunks[4].text.startswith("Schedule 1—Exempt calls\n1 Charities")
    assert _visible("".join(c.text for c in chunks)) == _visible("\n".join(pages))


def test_a_date_at_the_start_of_a_line_is_not_a_heading():
    page = """2 Commencement
This Act commences on
1 July 2025, or on an earlier day fixed by Proclamation.
3 Objects
The object of this Act is to reduce unsolicited calls."""
    assert [c.section_number for c in _chunks(page)] == ["2", "3"]
