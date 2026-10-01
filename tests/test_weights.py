"""`myna.weights` — the fetch is the seam where a published figure meets the bytes.

    uv run pytest tests/test_weights.py

Two claims this file exists to hold, both from the brief that added the module:
*"users never touch `gh`"*, and *"never bundle weights into the wheel"*. The second
is cheap to break silently — a `model.pt` copied under `src/` by an eager `git add -f`
would ride in every wheel forever — so it gets an executable guard rather than a
sentence: the default cache must sit outside the package, and the module must import
nothing but the standard library, because a `requests` import would make the fetcher
a dependency instead of a convenience.

The first claim is where the risk actually lives. `myna.report` prints a macro over
any `model.pt` it is handed, so a truncated download is not an outage — it is a
confident number that belongs to nobody. Every test below therefore asks what the
fetcher *refuses*, with the network faked rather than reached: CI has no GitHub
credentials and no business pulling 65 MB to prove a hash.

The last test is the seam. Bytes that only this module's own code path could have
produced go into `Myna(...)` and answer a typed question — the difference between
"the downloader writes files" and "the downloader writes a checkpoint the engine
loads".
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

import myna
from myna import weights as W
from myna.engine import Myna
from myna.model import MynaConfig, MynaModel
from myna.tokenizer import train_tokenizer

STATES = [f"order {i} never arrived and i want a refund ticket {i}" for i in range(64)]
DEPARTMENT = {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": ["billing", "technical", "shipping", "returns", "other"]}
TAG = "t-1"
REPO = "o/r"


class _Resp:
    """What `urlopen` hands back: readable once, as a context manager.

    `die_after` makes the *body read* fail after some bytes have already been
    returned, which is what a reset connection looks like from inside a chunked
    write — not an empty file, a partial one.
    """

    def __init__(self, body: bytes, die_after: int | None = None):
        self._buf = io.BytesIO(body)
        self._die_after = die_after

    def read(self, n: int = -1) -> bytes:
        if self._die_after is not None and self._buf.tell() > 0 and \
                self._buf.tell() >= self._die_after:
            raise urllib.error.URLError("connection reset mid-body")
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeGitHub:
    """The two URLs the module asks for, answered from memory.

    `publish_digest=False` reproduces an older release asset with no `digest` field,
    which the module must still hash and record rather than wave through.
    `lies` serves different bytes than the release advertises — the shape of a
    swapped download. `truncated` serves a short 200, which is the same shape with
    no error raised anywhere. `fail` dies mid-body.
    """

    def __init__(self, assets: dict[str, bytes], *, publish_digest: bool = True,
                 empty: set[str] = frozenset(), fail: set[str] = frozenset(),
                 truncated: dict[str, int] = None, lies: dict[str, bytes] = None):
        self.assets = assets
        self.publish_digest = publish_digest
        self.empty = set(empty)
        self.fail = set(fail)
        self.truncated = dict(truncated or {})
        self.lies = dict(lies or {})
        self.api_calls = 0
        self.download_calls: list[str] = []

    def digest(self, name: str) -> str:
        return hashlib.sha256(self.assets[name]).hexdigest()

    def _api_body(self) -> bytes:
        entries = []
        for name, body in self.assets.items():
            entry = {"name": name, "size": len(body),
                     "browser_download_url": f"https://g.example/{TAG}/{name}"}
            if self.publish_digest:
                entry["digest"] = f"sha256:{self.digest(name)}"
            entries.append(entry)
        return json.dumps({"assets": entries}).encode()

    def urlopen(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.startswith("https://api.github.com/"):
            self.api_calls += 1
            return _Resp(self._api_body())
        name = url.rsplit("/", 1)[-1]
        if name in self.empty:
            self.download_calls.append(name)
            return _Resp(b"")
        if name not in self.assets:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        self.download_calls.append(name)
        if name in self.fail:
            return _Resp(self.assets[name], die_after=1024)
        if name in self.truncated:
            return _Resp(self.assets[name][: self.truncated[name]])
        return _Resp(self.lies.get(name, self.assets[name]))

    def install(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", self.urlopen)
        return self

    def fetch(self, monkeypatch, dest: Path, **kw):
        self.install(monkeypatch)
        return W.fetch_weights(TAG, dest=dest, repo=REPO, **kw)


@pytest.fixture
def ckpt_bytes(tmp_path_factory):
    """A real checkpoint in the release's format: `model.pt` carries `cfg`,
    `state_dict` and `temperature`, `tokenizer.json` sits beside it. Small and
    untrained — the shape is what the fetcher must preserve, not the accuracy."""
    tok = train_tokenizer(STATES, vocab_size=400)
    cfg = MynaConfig(vocab=tok.get_vocab_size(), d_model=32, n_layers=1, n_heads=2,
                     d_k=8, d_v=8, d_ff=32, d_ptr=16)
    torch.manual_seed(3)
    buf = io.BytesIO()
    torch.save({"state_dict": MynaModel(cfg).state_dict(), "cfg": vars(cfg),
                "temperature": 1.0}, buf)
    d = tmp_path_factory.mktemp("ckpt-src")
    tok.save(str(d / "tokenizer.json"))
    return {"model.pt": buf.getvalue(),
            "tokenizer.json": (d / "tokenizer.json").read_bytes()}


def test_fetch_writes_both_assets_and_returns_their_directory(tmp_path, monkeypatch, ckpt_bytes):
    dirn = FakeGitHub(ckpt_bytes).fetch(monkeypatch, tmp_path / "f")
    assert dirn == tmp_path / "f"
    assert sorted(p.name for p in dirn.iterdir()) == ["SHA256SUMS", "model.pt", "tokenizer.json"]
    assert (dirn / "model.pt").read_bytes() == ckpt_bytes["model.pt"]


def test_the_digest_written_to_disk_is_the_one_the_release_publishes(tmp_path, monkeypatch,
                                                                    ckpt_bytes):
    server = FakeGitHub(ckpt_bytes)
    dirn = server.fetch(monkeypatch, tmp_path / "d")
    manifest = W.read_manifest(dirn)
    for name in W.ASSETS:
        assert manifest[name] == server.digest(name) == W._sha256(dirn / name)


def test_the_manifest_is_readable_by_a_tool_that_is_not_this_module(tmp_path, monkeypatch,
                                                                   ckpt_bytes):
    dirn = FakeGitHub(ckpt_bytes).fetch(monkeypatch, tmp_path / "m")
    lines = (dirn / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        digest, gap, name = line.partition("  ")
        assert gap == "  ", "two spaces is what `sha256sum -c` parses"
        assert name in W.ASSETS
        assert digest == hashlib.sha256((dirn / name).read_bytes()).hexdigest()


def test_a_complete_cache_is_reused_without_touching_the_network(tmp_path, monkeypatch,
                                                                ckpt_bytes):
    dirn = FakeGitHub(ckpt_bytes).fetch(monkeypatch, tmp_path / "r")
    second = FakeGitHub(ckpt_bytes).install(monkeypatch)
    assert W.fetch_weights(TAG, dest=dirn, repo=REPO) == dirn
    assert second.api_calls == 0 and second.download_calls == []


def test_a_tampered_asset_is_caught_by_its_own_manifest(tmp_path, monkeypatch, ckpt_bytes):
    dirn = FakeGitHub(ckpt_bytes).fetch(monkeypatch, tmp_path / "t")
    good = (dirn / "model.pt").read_bytes()
    (dirn / "model.pt").write_bytes(b"\x00" * len(good))
    assert W.cached(dirn) is False
    server = FakeGitHub(ckpt_bytes).install(monkeypatch)
    W.fetch_weights(TAG, dest=dirn, repo=REPO)
    assert (dirn / "model.pt").read_bytes() == good
    assert server.download_calls == ["model.pt"], "the untouched asset must not be re-pulled"


def test_a_cache_written_by_hand_is_verified_then_adopted(tmp_path, monkeypatch, ckpt_bytes):
    """`gh release download` leaves no manifest, so those bytes get checked against
    the release rather than trusted because they exist."""
    dirn = tmp_path / "byhand"
    dirn.mkdir()
    for name, body in ckpt_bytes.items():
        (dirn / name).write_bytes(body)
    assert W.cached(dirn) is False, "no manifest yet means not yet *cached*"
    server = FakeGitHub(ckpt_bytes).install(monkeypatch)
    W.fetch_weights(TAG, dest=dirn, repo=REPO)
    assert server.api_calls == 1 and server.download_calls == []
    assert W.cached(dirn) is True


def test_a_digest_mismatch_raises_and_leaves_no_bad_bytes(tmp_path, monkeypatch, ckpt_bytes):
    bad = ckpt_bytes["model.pt"] + b"extra"
    server = FakeGitHub(ckpt_bytes, lies={"model.pt": bad})
    dirn = tmp_path / "lie"
    with pytest.raises(W.WeightsError) as e:
        server.fetch(monkeypatch, dirn)
    assert "hashed" in str(e.value) and "release's" in str(e.value)
    assert not (dirn / "model.pt").exists(), "refused bytes must not stay on disk"
    assert not (dirn / "SHA256SUMS").exists()


def test_an_older_asset_with_no_digest_field_still_gets_hashed(tmp_path, monkeypatch, ckpt_bytes):
    server = FakeGitHub(ckpt_bytes, publish_digest=False)
    dirn = server.fetch(monkeypatch, tmp_path / "nd")
    assert W.read_manifest(dirn)["model.pt"] == server.digest("model.pt")


def test_an_unknown_tag_names_the_tag_and_what_to_type(tmp_path, monkeypatch):
    def notfound(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
    monkeypatch.setattr(urllib.request, "urlopen", notfound)
    with pytest.raises(W.WeightsError) as e:
        W.fetch_weights("no-such-tag", dest=tmp_path / "x", repo=REPO)
    assert "no-such-tag" in str(e.value) and "gh release list" in str(e.value)


def test_a_release_missing_one_asset_says_which_one(tmp_path, monkeypatch, ckpt_bytes):
    FakeGitHub({"model.pt": ckpt_bytes["model.pt"]}).install(monkeypatch)
    with pytest.raises(W.WeightsError) as e:
        W.fetch_weights(TAG, dest=tmp_path / "p", repo=REPO)
    assert "tokenizer.json" in str(e.value)


def test_an_empty_body_leaves_no_zero_byte_file(tmp_path, monkeypatch, ckpt_bytes):
    server = FakeGitHub(ckpt_bytes, empty={"model.pt"})
    dirn = tmp_path / "empty"
    with pytest.raises(W.WeightsError) as e:
        server.fetch(monkeypatch, dirn)
    assert "came back empty" in str(e.value)
    assert not (dirn / "model.pt").exists()
    assert not list(dirn.glob("*.part")), "the temp file is a refused download's remains"


def test_a_connection_that_dies_mid_download_leaves_no_temp_file(tmp_path, monkeypatch,
                                                                  ckpt_bytes):
    """Truncation is the failure this module exists for, and urllib reports a dead
    connection as an ordinary `URLError` from the body read rather than as a short
    file — so the `.part` is what has to disappear, not just the final name."""
    server = FakeGitHub(ckpt_bytes, fail={"model.pt"})
    dirn = tmp_path / "died"
    with pytest.raises(W.WeightsError) as e:
        server.fetch(monkeypatch, dirn)
    assert "failed" in str(e.value)
    assert not (dirn / "model.pt").exists()
    assert not list(dirn.glob("*.part"))
    assert not (dirn / "SHA256SUMS").exists(), "half a checkpoint is not a cache"


def test_a_truncated_body_is_refused_by_its_digest(tmp_path, monkeypatch, ckpt_bytes):
    """A short 200 is *not* an error to urllib, so the only thing standing between a
    truncated download and a published-looking number is the hash."""
    server = FakeGitHub(ckpt_bytes, truncated={"model.pt": 4096})
    dirn = tmp_path / "short"
    with pytest.raises(W.WeightsError) as e:
        server.fetch(monkeypatch, dirn)
    assert "hashed" in str(e.value) and "release's" in str(e.value)
    assert not (dirn / "model.pt").exists()


def test_check_is_a_verb_that_never_writes(tmp_path, monkeypatch, ckpt_bytes, capsys):
    dirn = FakeGitHub(ckpt_bytes).fetch(monkeypatch, tmp_path / "c")
    FakeGitHub(ckpt_bytes).install(monkeypatch)
    assert W.main(["--tag", TAG, "--dest", str(dirn), "--check"]) == 0
    assert "verified" in capsys.readouterr().out

    untouched = tmp_path / "c-empty"
    untouched.mkdir()
    assert W.main(["--tag", TAG, "--dest", str(untouched), "--check"]) == 1
    assert not list(untouched.iterdir()), "--check wrote into the directory"

    (dirn / "model.pt").write_bytes(b"\x00" * 4)
    assert W.main(["--tag", TAG, "--dest", str(dirn), "--check"]) == 1
    assert (dirn / "tokenizer.json").read_bytes() == ckpt_bytes["tokenizer.json"]


def test_the_cli_reports_a_failure_without_a_traceback(tmp_path, monkeypatch, capsys):
    def boom(req, timeout=None):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert W.main(["--tag", "t-9", "--dest", str(tmp_path / "nope")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("myna-weights:") and "Traceback" not in err


def test_the_fetcher_adds_no_dependency():
    """`urllib` only. A `requests` import would make the documented install line
    wrong on any box that has not installed it — and torch is already the reason the
    wheel is not smaller."""
    tree = ast.parse(Path(myna.__file__).with_name("weights.py").read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    allowed = {"__future__", "argparse", "hashlib", "json", "os", "sys", "urllib", "pathlib"}
    assert mods <= allowed, f"non-stdlib import in weights.py: {sorted(mods - allowed)}"


def test_weights_never_land_inside_the_package(tmp_path, monkeypatch):
    """The cache is `$XDG_CACHE_HOME/myna/<tag>`. A cache rooted in the package would
    be one careless `git add -f` away from shipping a 65 MB checkpoint in every wheel."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = W.cache_root()
    pkg = Path(myna.__file__).resolve().parent
    assert root == tmp_path / "myna"
    assert not str(root.resolve()).startswith(str(pkg))
    assert not list(pkg.glob("**/*.pt"))


def test_default_tag_and_repo_are_the_ones_the_registry_quotes():
    assert W.DEFAULT_TAG == "antiprior_off_s0-weights"
    assert W.REPO == "aashish254/myna"


def test_a_fetched_checkpoint_loads_and_answers_a_typed_question(tmp_path, monkeypatch,
                                                                ckpt_bytes):
    """The seam: this module's output goes straight into the engine's input, with no
    path handed in by hand and no file renamed. `SHA256SUMS` shares the directory, so
    this also proves the engine ignores a neighbour it does not know."""
    dirn = FakeGitHub(ckpt_bytes).fetch(monkeypatch, tmp_path / "seam")
    engine = Myna(dirn)
    out = engine.predict("My order never arrived and I want a refund.",
                         {"department": DEPARTMENT})
    answer = out["answers"]["department"]
    assert answer["type"] == "choice" and answer["choice"] in DEPARTMENT["criteria"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert out["policy"]["abstain_below"] is None
    assert out["usage"]["state_tokens"] > 0
