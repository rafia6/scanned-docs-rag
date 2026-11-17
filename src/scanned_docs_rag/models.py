"""Plain data types shared across the pipeline."""

from dataclasses import dataclass, field


@dataclass
class Page:
    """Text of one PDF page and how it was extracted."""

    document: str
    page_number: int  # 1-based, as a human would cite it
    text: str
    ocr: bool = False


@dataclass
class Chunk:
    """A piece of a page that gets embedded and stored."""

    id: str
    document: str
    page_number: int
    text: str
    ocr: bool = False

    @property
    def metadata(self) -> dict:
        return {"document": self.document, "page": self.page_number, "ocr": self.ocr}


@dataclass
class Source:
    """A retrieved chunk, numbered so the answer can cite it as [n]."""

    number: int
    document: str
    page: int
    text: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
