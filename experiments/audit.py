#!/usr/bin/env python3
"""CLI for append-only project audit trail."""

from __future__ import annotations

import argparse
import json
import sys

from qsp_jax.audit import append_audit_entry, read_audit_entries


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    append = sub.add_parser("append", help="Append one audit entry to docs/audit/LOG.jsonl")
    append.add_argument("--category", required=True)
    append.add_argument("--status", required=True)
    append.add_argument("--title", required=True)
    append.add_argument("--what", required=True)
    append.add_argument("--why", default="")
    append.add_argument("--action", default="")
    append.add_argument("--rationale", default="")
    append.add_argument("--evidence", default="", help="Comma-separated paths")
    append.add_argument("--labels", default="", help="Comma-separated labels")
    append.add_argument("--related-ids", default="", help="Comma-separated AUD- IDs")

    listing = sub.add_parser("list", help="List audit entries")
    listing.add_argument("--last", type=int, default=20)
    listing.add_argument("--label", default=None)
    listing.add_argument("--json", action="store_true", help="Print raw JSON lines")

    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()

    if args.command == "append":
        entry = append_audit_entry(
            category=args.category,
            status=args.status,
            title=args.title,
            what=args.what,
            why=args.why,
            action=args.action,
            rationale=args.rationale,
            evidence=_split_csv(args.evidence),
            labels=_split_csv(args.labels),
            related_ids=_split_csv(args.related_ids),
        )
        print(f"Appended {entry.id} → docs/audit/LOG.jsonl")
        return

    entries = read_audit_entries(last=args.last, label=args.label)
    if args.json:
        for entry in entries:
            print(json.dumps(entry, ensure_ascii=False))
        return

    if not entries:
        print("No audit entries found.")
        return

    for entry in entries:
        labels = ", ".join(entry.get("labels", []))
        print(
            f"{entry['id']} [{entry['status']}/{entry['category']}] "
            f"{entry['title']} ({labels})"
        )
        print(f"  what: {entry.get('what', '')}")
        if entry.get("why"):
            print(f"  why:  {entry['why']}")
        if entry.get("action"):
            print(f"  action: {entry['action']}")


if __name__ == "__main__":
    main()
