"""Body-only exact deduplication for a single-label JSONL news corpus.

No external packages required. Input is never modified. Existing outputs
are not overwritten. All copies of bodies with conflicting harmonized
categories are excluded; one copy of consistent-label bodies is retained.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

CATEGORY_MAPPING = {"business": "economy"}


def normalize_text(text):
    # Same normalization as the earlier script: whitespace + lowercase.
    # Does not normalize letter variants or detect near duplicates.
    return re.sub(r"\s+", " ", text).strip().lower()


def generate_article_hash(content):
    return hashlib.sha256(normalize_text(content).encode("utf-8")).hexdigest()


def get_body(article, line):
    # Prefer the field used by the benchmark, then fall back to content.
    for key in ("text", "content"):
        value = article.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    raise ValueError(f"Line {line}: no nonempty string in text or content.")


def get_category(article, line):
    # This benchmark requires one category, not a list of labels.
    value = article.get("category")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line}: category must be a nonempty string.")
    category = value.strip()
    return CATEGORY_MAPPING.get(category, category)


def reference(record):
    article = record["article"]
    return {
        "input_line": record["line"],
        "category": record["category"],
        "title": article.get("title", ""),
        "source": article.get("source", article.get("publisher", "")),
        "url": article.get("url", article.get("link", "")),
        "body_field": record["body_field"],
    }


def write_jsonl(path, records):
    # Exclusive creation avoids accidentally replacing a previous dataset.
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def deduplicate_articles(input_file, output_file, report_prefix):
    if not input_file.is_file():
        raise FileNotFoundError(f"Input not found: {input_file.resolve()}")
    duplicate_file = Path(str(report_prefix) + "_duplicates.jsonl")
    conflict_file = Path(str(report_prefix) + "_conflicts.jsonl")
    summary_file = Path(str(report_prefix) + "_summary.json")
    targets = [output_file, duplicate_file, conflict_file, summary_file]
    if len({path.resolve() for path in targets + [input_file]}) != 5:
        raise ValueError("Input, output, and report paths must all be different.")
    for path in targets:
        if path.exists():
            raise FileExistsError(f"Already exists: {path}. Choose new output/report names.")
        path.parent.mkdir(parents=True, exist_ok=True)

    groups = {}
    input_counts = Counter()
    total = 0
    print(f"Loading {input_file} ...")
    with input_file.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON: {error}") from error
            if not isinstance(article, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            body, body_field = get_body(article, line_number)
            category = get_category(article, line_number)
            body_hash = generate_article_hash(body)
            groups.setdefault(body_hash, []).append({
                "article": article, "line": line_number, "body": body,
                "body_field": body_field, "category": category,
            })
            input_counts[category] += 1
            total += 1
    if not total:
        raise ValueError("Input contains no article records.")

    duplicate_reports = []
    conflicts = []
    cleaned = []
    conflict_records_removed = 0
    same_category_duplicates_removed = 0
    conflict_category_counts = Counter()
    for body_hash, records in groups.items():
        categories = sorted({record["category"] for record in records})
        if len(records) > 1:
            duplicate_reports.append({
                "body_hash": body_hash, "count": len(records),
                "category_conflict": len(categories) > 1,
                "occurrences": [reference(record) for record in records],
            })
        if len(categories) > 1:
            conflict_records_removed += len(records)
            conflict_category_counts.update(record["category"] for record in records)
            # Compact report: preserve each title/category/URL association.
            # Use original category names to locate the category JSONL files.
            conflicts.append({
                "categories": categories,
                "articles": [{
                    "title": record["article"].get("title", ""),
                    "category": record["category"],
                    "url": record["article"].get("url") or record["article"].get("link", ""),
                } for record in records],
            })
            continue

        same_category_duplicates_removed += len(records) - 1
        first = records[0]
        retained = dict(first["article"])
        # Preserve any earlier title+body hash under its original hash field.
        retained["body_hash"] = body_hash
        retained["text"] = first["body"]
        retained["category"] = first["category"]
        cleaned.append(retained)

    summary = {
        "input_file": str(input_file.resolve()),
        "input_records": total,
        "unique_normalized_bodies": len(groups),
        "duplicate_groups": len(duplicate_reports),
        "extra_duplicate_records": total - len(groups),
        "conflicting_label_groups": len(conflicts),
        "records_excluded_for_category_conflicts": conflict_records_removed,
        "same_category_duplicate_records_removed": same_category_duplicates_removed,
        "total_records_removed": total - len(cleaned),
        "conflict_removals_by_category": dict(sorted(conflict_category_counts.items())),
        "category_mapping": CATEGORY_MAPPING,
        "normalization": "Whitespace collapse, trim, lowercase; body only",
        "retention_rule": "Exclude all copies of conflicting bodies; otherwise retain first occurrence and provenance",
        "input_category_counts": dict(sorted(input_counts.items())),
        "cleaned_dataset_written": True,
    }
    write_jsonl(duplicate_file, duplicate_reports)
    write_jsonl(conflict_file, conflicts)
    assert total == len(cleaned) + conflict_records_removed + same_category_duplicates_removed
    assert len(cleaned) == len(groups) - len(conflicts)
    write_jsonl(output_file, cleaned)
    summary["output_file"] = str(output_file.resolve())
    summary["output_records"] = len(cleaned)
    summary["output_category_counts"] = dict(sorted(Counter(
        record["category"] for record in cleaned).items()))
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nExcluded {len(conflicts)} conflicting groups ({conflict_records_removed} records).")
    print(f"\nSaved cleaned dataset: {output_file.resolve()}")
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("raw_dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("Pashto-news-24K.jsonl"))
    parser.add_argument("--report-prefix", type=Path, default=Path("conflict_exclusion"))
    args = parser.parse_args()
    success = deduplicate_articles(args.input, args.output, args.report_prefix)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
