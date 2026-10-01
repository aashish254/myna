"""Fetch the published weights into a local cache — stdlib only, no `gh` required.

    python -m myna.weights --tag antiprior_off_s0-weights
    from myna import fetch_weights
    myna = Myna(fetch_weights())            # one line, no release page open

The weights are never inside the wheel: this repo's checkpoints are 65 MB of
`model.pt` per arm and a package that carried one would make `pip install` a
download of a particular training run. So the wheel ships code, and this module
pulls the bytes a claim is attached to out of the GitHub Release that carries it.

Verification is the point, not a nicety. `myna.report` will print a macro over any
`model.pt` you hand it, so a truncated or swapped download produces a confident
number that belongs to nobody. Every file is hashed as it lands and required to
match the digest GitHub publishes for that asset; the digests are then written to
`SHA256SUMS` in the format `sha256sum -c` reads, so the cache can be audited by a
tool that is not this module. A later call with a cache that satisfies its own
`SHA256SUMS` returns without touching the network — offline reuse is the normal
case once a machine has the weights, and the GitHub API's 60 requests per hour
unauthenticated is not a budget a CI job should spend on a re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "aashish254/myna"
ASSETS = ("model.pt", "tokenizer.json")
MANIFEST = "SHA256SUMS"
# The arm the repo currently quotes: `antiprior-off-macro` in bench/reproduce.py.
DEFAULT_TAG = "antiprior_off_s0-weights"
USER_AGENT = "myna-weights"


class WeightsError(RuntimeError):
    """Every message here names the tag, the asset and what to type by hand."""


def cache_root() -> Path:
    """`$XDG_CACHE_HOME/myna`, else `~/.cache/myna` — and a cache that a reader
    can point `ls` at, because 'dest' being a temp directory is how weights get
    downloaded four times in one afternoon."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "myna"


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def release(tag: str, repo: str, timeout: float) -> dict[str, dict]:
    """The release's assets, keyed by name, with the digest GitHub computed."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise WeightsError(f"no release tagged {tag!r} in {repo} — list them with "
                               f"`gh release list -R {repo}`") from e
        raise WeightsError(f"GET {url} returned HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise WeightsError(f"cannot reach {url} ({e.reason if hasattr(e, 'reason') else e}); "
                           f"if the weights are already cached at {cache_root() / tag} this "
                           f"call should not have needed the network") from e
    assets = {a["name"]: {"url": a["browser_download_url"], "size": a["size"],
                          "sha256": (a.get("digest") or "").removeprefix("sha256:") or None}
              for a in payload.get("assets", [])}
    missing = [name for name in ASSETS if name not in assets]
    if missing:
        raise WeightsError(f"release {tag} has no {', '.join(missing)} asset — it carries "
                           f"{', '.join(assets) or 'nothing'}")
    return assets


def download(url: str, dest: Path, timeout: float) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        tmp.unlink(missing_ok=True)
        reason = e.reason if isinstance(e, urllib.error.URLError) else e
        raise WeightsError(f"download of {url} failed ({reason})") from e
    # A partial body is a success to urllib, so the temp file is only renamed once
    # it hashes to something — an empty file on disk would otherwise outlive the run.
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise WeightsError(f"{url} came back empty")
    tmp.replace(dest)


def read_manifest(dirn: Path) -> dict[str, str]:
    path = dirn / MANIFEST
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def cached(dirn: Path) -> bool:
    """True when the directory holds every asset and each hashes to what the
    manifest beside it claims. A cache written by hand with `gh release download`
    leaves no manifest, so it is not *cached* here: the next call asks the release
    for its digests and re-hashes those bytes, and only then writes a manifest —
    a directory this module did not place there gets verified, not trusted."""
    claimed = read_manifest(dirn)
    if not claimed:
        return False
    return all((dirn / name).is_file()
               and claimed.get(name) == _sha256(dirn / name) for name in ASSETS)


def fetch_weights(tag: str = DEFAULT_TAG, dest: str | os.PathLike[str] | None = None, *,
                  repo: str = REPO, force: bool = False,
                  timeout: float = 60.0) -> Path:
    """Return the directory holding `model.pt` and `tokenizer.json`, ready to hand
    to `Myna(...)`. Downloads only what is missing or fails its digest."""
    dirn = Path(dest) if dest is not None else cache_root() / tag
    if not force and cached(dirn):
        return dirn
    assets = release(tag, repo, timeout)
    dirn.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in ASSETS:
        spec = assets[name]
        path = dirn / name
        got = _sha256(path) if path.is_file() else None
        if got is not None and not force and (spec["sha256"] is None or got == spec["sha256"]):
            digests[name] = got
        else:
            download(spec["url"], path, timeout)
            digests[name] = _sha256(path)
        if spec["sha256"] is not None and digests[name] != spec["sha256"]:
            path.unlink(missing_ok=True)
            raise WeightsError(
                f"{name} from {tag} hashed {digests[name]}, not the release's {spec['sha256']}. "
                f"The bytes were deleted; retry `python -m myna.weights --tag {tag}`, and if "
                f"it repeats the Release asset itself is the problem, not this fetch.")
    # A release whose digest field is empty still gets a hash on disk, because
    # that is what makes the next call's reuse a check rather than a hope.
    (dirn / MANIFEST).write_text("".join(f"{digests[name]}  {name}\n" for name in ASSETS))
    return dirn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="myna-weights",
        description="Fetch a Myna checkpoint from its GitHub Release into the local cache. "
                    "The wheel carries code only; weights stay on the Release that a "
                    "published figure is quoted against.")
    ap.add_argument("--tag", default=DEFAULT_TAG, metavar="TAG",
                    help=f"release tag to fetch (default {DEFAULT_TAG})")
    ap.add_argument("--repo", default=REPO, metavar="OWNER/NAME",
                    help=f"release owner/repo (default {REPO})")
    ap.add_argument("--dest", default=None, metavar="DIR",
                    help="directory to write into (default $XDG_CACHE_HOME or ~/.cache, "
                         "then myna/<tag>)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even when the cache already hashes correctly")
    ap.add_argument("--check", action="store_true",
                    help="print whether the cache verifies and exit 0 or 1; never writes")
    ap.add_argument("--print-path", action="store_true",
                    help="print only the directory, for `Myna($(myna-weights ...))`")
    args = ap.parse_args(argv)

    dest = Path(args.dest) if args.dest else cache_root() / args.tag
    if args.check:
        ok = cached(dest)
        print(f"{dest}: {'verified' if ok else 'not usable'}")
        return 0 if ok else 1
    try:
        dirn = fetch_weights(args.tag, dest=args.dest, repo=args.repo, force=args.force)
    except WeightsError as e:
        print(f"myna-weights: {e}", file=sys.stderr)
        return 2
    if args.print_path:
        print(dirn)
        return 0
    manifest = read_manifest(dirn)
    print(f"tag   {args.tag}")
    print(f"dir   {dirn}")
    for name in ASSETS:
        print(f"      {name}  {manifest[name]}")
    print(f"      Myna(\"{dirn}\") is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
