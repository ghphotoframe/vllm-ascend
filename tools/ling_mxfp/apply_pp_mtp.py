#!/usr/bin/env python3
"""Apply the archived Ling PP/MTP fix to an explicitly selected vLLM tree."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True, help="vLLM source repository root")
    parser.add_argument("--check", action="store_true", help="validate source and patch without writing")
    parser.add_argument("--rollback", action="store_true", help="restore the archived original version")
    args = parser.parse_args()

    archive = Path(__file__).resolve().parent / "pp_mtp_embedding"
    manifest = json.loads((archive / "manifest.json").read_text())
    relative = Path("vllm/model_executor/models/bailing_moe_v3_mtp.py")
    target = args.vllm_root.resolve() / relative
    source, destination = ("after", "before") if args.rollback else ("before", "after")
    replacement = (archive / (destination + ".py")).read_bytes()
    expected = manifest[destination + "_sha256"]
    if hashlib.sha256(replacement).hexdigest() != expected:
        raise SystemExit("Archived replacement hash mismatch; no files changed")
    compile(replacement, str(target), "exec")
    original = target.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual == expected:
        status = "already-restored" if args.rollback else "already-applied"
    elif actual != manifest[source + "_sha256"]:
        raise SystemExit("Unknown vLLM source hash " + actual + "; no files changed")
    elif args.check:
        status = "ready-to-restore" if args.rollback else "ready-to-apply"
    else:
        if target.read_bytes() != original:
            raise SystemExit("Source changed during validation; no files changed")
        target.write_bytes(replacement)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            target.write_bytes(original)
            raise SystemExit("Verification failed; original source restored")
        status = "restored" if args.rollback else "applied"
    print(json.dumps({"status": status, "path": str(target), "sha256": actual if args.check else expected}))


if __name__ == "__main__":
    main()
