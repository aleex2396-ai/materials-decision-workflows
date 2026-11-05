"""Summarise the public OpenAlex snapshots without calling a network service."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = []
seen = set()
for path in sorted((ROOT / "openalex").glob("*.json")):
    snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
    response = snapshot["response"]
    if isinstance(response, str):
        response = json.loads(response)
    if "results" in response:
        continue
    if response["id"] in seen:
        continue
    seen.add(response["id"])
    records.append({
        "key": path.stem,
        "openalex_id": response["id"],
        "doi": response.get("doi"),
        "title": response["title"],
        "year": response["publication_year"],
        "authors": [a["author"]["display_name"] for a in response["authorships"]],
        "source_url": response["primary_location"].get("landing_page_url"),
        "retrieved_at": snapshot["retrieved_at"],
        "query_url": snapshot["url"],
        "snapshot": path.relative_to(ROOT).as_posix(),
    })
(ROOT / "bibliography.json").write_text(json.dumps(records, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
entries = []
for record in records:
    fields = {
        "title": record["title"],
        "author": " and ".join(record["authors"]),
        "year": str(record["year"]),
        "doi": (record["doi"] or "").removeprefix("https://doi.org/"),
        "url": record["source_url"] or record["openalex_id"],
        "note": "Metadata retrieved from OpenAlex on 2026-11-05; publication venue/version should be checked against primary source",
    }
    entries.append("@misc{"+record["key"]+",\n"+",\n".join(f"  {key} = {{{value}}}" for key,value in fields.items())+"\n}")
(ROOT / "references.bib").write_text("\n\n".join(entries)+"\n", encoding="utf-8")
print(f"Wrote {len(records)} deduplicated selected paper records")
