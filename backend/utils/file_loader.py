import json
from typing import List, Dict, Any

def load_cases_as_docs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for row in data:
        case_name = row.get("case_name", "Unknown Case")
        year = row.get("year", "")
        court = row.get("court", "")
        citation = row.get("citation", "")
        area = row.get("area_of_law", [])
        summary = row.get("summary", "")
        bench = row.get("bench", [])

        # Build RAG text field
        text = (
            f"Case: {case_name} ({year})\n"
            f"Court: {court}\n"
            f"Citation: {citation}\n"
            f"Areas: {', '.join(area)}\n"
            f"Bench: {', '.join(bench)}\n"
            f"Summary: {summary}\n"
        )

        meta = {
            "case_name": case_name,
            "year": year,
            "court": court,
            "citation": citation,
            "area_of_law": area,
            "summary": summary,
            "bench": bench,
        }
        docs.append({"id": row.get("id"), "text": text, "meta": meta})
    return docs
