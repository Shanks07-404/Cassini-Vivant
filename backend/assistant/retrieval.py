"""Rulebook retrieval engine for illustrative demo rules and glossary documents."""

import re
from pathlib import Path
from typing import Any, Dict, List
from backend.app.config import RULES_DIR


class RulebookChunk:
    def __init__(self, doc_id: str, doc_title: str, section_id: str, section_title: str, content: str, file_path: Path):
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.section_id = section_id
        self.section_title = section_title
        self.content = content.strip()
        self.file_path = file_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "content": self.content,
            "citation": f"{self.doc_id} § {self.section_id} ({self.section_title})",
        }


class RulebookRetriever:
    """Indexes markdown rule documents and retrieves relevant sections for queries."""

    def __init__(self, rules_dir: Path = RULES_DIR):
        self.rules_dir = rules_dir
        self.chunks: List[RulebookChunk] = []
        self._load_and_index()

    def _load_and_index(self):
        self.chunks.clear()
        if not self.rules_dir.exists():
            return

        for md_path in self.rules_dir.glob("*.md"):
            self._parse_file(md_path)

    def _parse_file(self, path: Path):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Extract doc id and title
        doc_id = path.stem
        doc_title = path.stem
        for line in lines[:10]:
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif "Document ID:" in line:
                m = re.search(r"`([^`]+)`", line)
                if m:
                    doc_id = m.group(1)

        # Split by section headers (## § ...)
        current_section_id = "Intro"
        current_section_title = "Dispositions Générales"
        current_lines: List[str] = []

        for line in lines:
            sec_match = re.match(r"^##\s+(?:§\s*([\d\.]+)\s*—\s*(.+)|(.+))", line)
            if sec_match:
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        self.chunks.append(RulebookChunk(
                            doc_id=doc_id,
                            doc_title=doc_title,
                            section_id=current_section_id,
                            section_title=current_section_title,
                            content=content,
                            file_path=path,
                        ))
                    current_lines = []

                if sec_match.group(1):
                    current_section_id = sec_match.group(1).strip()
                    current_section_title = sec_match.group(2).strip()
                else:
                    current_section_id = "Section"
                    current_section_title = sec_match.group(3).strip()
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                self.chunks.append(RulebookChunk(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_id=current_section_id,
                    section_title=current_section_title,
                    content=content,
                    file_path=path,
                ))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Score and return the top relevant chunks for a given query."""
        if not self.chunks:
            self._load_and_index()

        q_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        if not q_terms:
            return [c.to_dict() for c in self.chunks[:top_k]]

        scored = []
        for chunk in self.chunks:
            text_lower = f"{chunk.doc_id} {chunk.doc_title} {chunk.section_id} {chunk.section_title} {chunk.content}".lower()
            score = 0
            for term in q_terms:
                count = text_lower.count(term)
                if count > 0:
                    # Give higher weight to matches in doc_id or section_title
                    if term in chunk.doc_id.lower():
                        score += 5 * count
                    elif term in chunk.section_title.lower():
                        score += 3 * count
                    else:
                        score += 1 * count
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c.to_dict() for _, c in scored[:top_k]]


retriever = RulebookRetriever()
