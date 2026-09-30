"""`bench/scan_secrets.py` has to be able to fail, or its "0 matches" means nothing.

The defect this protects against already happened: a one-off scan parsed
`git rev-list --objects` with the wrong field count, walked zero blobs, printed "0
blobs scanned, 0 matches", and the sentence shipped into SPEC §5 P0 as proof the
history was clean. Three arms, in a throwaway repo so nothing here touches this
tree's object database:

* a clean repo prints its denominator — and that denominator is the assertion, so a
  broken parse shows up as a wrong blob count rather than a quiet green;
* a planted credential is reported by shape and path, and the tool exits 1;
* the same credential after `--amend`, when no ref points at it any more, is still
  found — the scan walks the object database, not the refs — but is classified as
  local debt and does not fail the gate, because a push transfers refs and nothing
  else. Both halves of that arm matter: an object-database walk that hid the finding
  would be blind, and one that failed on it could never go green on its own tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "bench" / "scan_secrets.py"

# Built from parts so this file's own committed blob does not match the shapes it is
# testing — a scanner that flags its own test fixture can never be run on its own tree.
FAKE = "token: " + "ghp_" + "A" * 24 + " and " + "KGAT_" + "X" * 20 + "\n"


def run_scan(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(TOOL)), cwd=cwd, capture_output=True, text=True, timeout=120
    )


def git(cwd: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True, text=True)


def commit(cwd: Path, name: str, body: str, message: str) -> None:
    (cwd / name).write_text(body)
    git(cwd, "add", name)
    git(cwd, "commit", "-qm", message)


def init(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@t.invalid")
    git(tmp_path, "config", "user.name", "t")
    return tmp_path


def test_clean_repo_prints_its_denominator(tmp_path: Path) -> None:
    init(tmp_path)
    commit(tmp_path, "note.txt", "hello\n", "benign")

    done = run_scan(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "blobs scanned     1" in done.stdout
    assert "blobs in the db   1" in done.stdout
    assert "0 matches for 9 shapes across 1 blobs" in done.stdout


def test_planted_credential_is_named_by_shape_and_path(tmp_path: Path) -> None:
    init(tmp_path)
    commit(tmp_path, "note.txt", "hello\n", "benign")
    commit(tmp_path, "leak.txt", FAKE, "plant")

    done = run_scan(tmp_path)
    assert done.returncode == 1
    assert "2 hit(s)" in done.stdout
    assert "gh-classic" in done.stdout
    assert "kaggle-token" in done.stdout
    assert "leak.txt" in done.stdout


def test_a_credential_in_an_unreachable_blob_is_still_caught(tmp_path: Path) -> None:
    init(tmp_path)
    commit(tmp_path, "note.txt", "hello\n", "benign")
    commit(tmp_path, "leak.txt", FAKE, "plant")
    leak_sha = subprocess.run(
        ("git", "rev-parse", "HEAD:leak.txt"), cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    (tmp_path / "leak.txt").write_text("nothing here\n")
    git(tmp_path, "add", "leak.txt")
    git(tmp_path, "commit", "-q", "--amend", "-m", "plant-removed")

    # No ref points at the leak blob any more, which is the case a ref-walk misses.
    listed = subprocess.run(
        ("git", "rev-list", "--all", "--objects"), cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout
    assert leak_sha not in listed

    done = run_scan(tmp_path)
    assert "2 hit(s) in unreachable blobs" in done.stdout
    assert "local debt" in done.stdout
    assert leak_sha in done.stdout
    assert "(unreachable)" in done.stdout
    # Not a push, so not a red gate — but it is printed, which a refs-only scan never does.
    assert done.returncode == 0, done.stdout + done.stderr
