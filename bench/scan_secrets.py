"""Scan every blob in git history for credential shapes, and print the denominator.

A "0 matches" line is not a result unless you know how many blobs were looked at: the
first version of this check parsed `git rev-list --objects` with the wrong field count,
walked nothing, and came back green. So the count of blobs scanned is compared against
the count of blobs the object database reports, and a mismatch exits 1 (SPEC §9.46).

Two classes of finding, because they are not the same fact. A hit in a blob reachable
from a ref is in the set a push transfers, so it fails the check. A hit in an
unreachable blob — an amended commit's leftover, or this tool's own earlier drafts
sitting in the object database after `git add` — is local debt on this machine and is
printed under its own heading without turning the gate red, since a refs-only scan
would have missed it either way.

Every shape here is printable-ASCII-only, which is what a leaked token in a file looks
like. The loosest version of the URL shape matched three screenshot blobs, where
compressed PNG byte streams happen to spell `://`…`@` — read, confirmed `\x89PNG` magic
with no printable match, and the reason the pattern narrowed rather than an allowlist.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

# Named shapes. Each is deliberately long-form: `KGAT_` in prose is a quotation, a
# credential is `KGAT_` followed by a token, so the {12,} is what separates the two.
PATTERNS = {
    "kaggle-token": rb"KGAT_[A-Za-z0-9_-]{12,}",
    "gh-oauth": rb"gho_[A-Za-z0-9]{20,}",
    "gh-classic": rb"ghp_[A-Za-z0-9]{20,}",
    "gh-fine-grained": rb"github_pat_[A-Za-z0-9_]{20,}",
    "private-key": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "aws-access-key": rb"AKIA[0-9A-Z]{16}",
    "slack-token": rb"xox[baprs]-[0-9A-Za-z-]{10,}",
    "openai-key": rb"sk-[A-Za-z0-9]{20,}",
    # Split across two literals on purpose: written as one string this line matches its
    # own shape, and the tool would report a credential in its own source. The space
    # between the literals is outside the printable class, so the concatenation still
    # compiles to one pattern and the source still reads as two.
    "url-credentials": rb"://[\x21-\x7e]{1,64}" rb":[\x21-\x7e]{1,128}@",
}

COMPILED = [(name, re.compile(rx)) for name, rx in PATTERNS.items()]


def git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, text=True, capture_output=True).stdout


def reachable_labels() -> dict[str, list[str]]:
    """sha -> the paths it appears at, for every object reachable from a ref.

    This is also the definition of *published*: a push transfers this set and nothing
    else. So it is the denominator for the failure decision, while the object database
    stays the denominator for what was actually looked at — a secret parked in an
    unreachable blob is local debt, not a leak, and the two get printed apart rather
    than one of them being quietly picked."""
    labels: dict[str, list[str]] = {}
    for line in git("rev-list", "--all", "--objects").splitlines():
        sha, _, name = line.partition(" ")
        if name:
            labels.setdefault(sha, []).append(name)
    return labels


def main() -> int:
    kinds: Counter[str] = Counter()
    blob_sizes: dict[str, int] = {}
    for line in git("cat-file", "--batch-all-objects", "--batch-check").splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        sha, kind, size = fields
        kinds[kind] += 1
        if kind == "blob":
            blob_sizes[sha] = int(size)

    total_blobs = len(blob_sizes)
    if not total_blobs:
        print("FAIL: object database reported 0 blobs — the parse is broken", file=sys.stderr)
        return 1

    unreachable = [
        line.split()[-1]
        for line in git("fsck", "--no-progress", "--unreachable").splitlines()
        if line.startswith("unreachable blob ")
    ]

    scanned = 0
    hits: list[tuple[str, str, str]] = []  # (sha, pattern, snippet)
    proc = subprocess.Popen(
        ("git", "cat-file", "--batch"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False,
    )
    assert proc.stdin is not None and proc.stdout is not None
    for sha in sorted(blob_sizes):
        proc.stdin.write((sha + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline()
        if not header.endswith(b"\n"):
            break
        size = int(header.split()[-1])
        data = proc.stdout.read(size + 1)
        scanned += 1
        for name, rx in COMPILED:
            found = rx.findall(data)
            if found:
                hits.append((sha, name, found[0].decode("utf-8", "replace")[:40]))

    print(f"blobs scanned     {scanned}")
    print(f"blobs in the db   {total_blobs}")
    print(f"trees / commits   {kinds['tree']} / {kinds['commit']}")
    print(f"blob bytes        {sum(blob_sizes.values()):,}")
    print(f"unreachable blobs {len(unreachable)}")
    if scanned != total_blobs:
        print(f"FAIL: scanned {scanned} of {total_blobs} — denominator does not match", file=sys.stderr)
        return 1

    labels = reachable_labels() if hits else {}
    published = [hit for hit in hits if hit[0] in labels]
    local_debt = [hit for hit in hits if hit[0] not in labels]

    if published:
        print(f"\n{len(published)} hit(s) reachable from a ref — this is the set a push transfers:")
        for sha, name, snippet in published:
            at = ", ".join(sorted(set(labels[sha]))[:3])
            print(f"  {sha}  {name}  {at}\n    {snippet}")
        return 1
    if local_debt:
        print(f"\n{len(local_debt)} hit(s) in unreachable blobs — local debt, in no ref, so no push")
        print("transfers them. Read each one; do not 'fix' it with a broad `git gc` without")
        print("checking what else that prunes:")
        for sha, name, snippet in local_debt:
            print(f"  {sha}  {name}  (unreachable)\n    {snippet}")
    print(f"\n0 matches for {len(PATTERNS)} shapes across {scanned} blobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
