"""Mutation battery for the browser gate (SPEC §5 P6 6b, §7.1).

    python bench/mutation_browser.py

`bench/mutation_onnx.py` proves the exported *graphs* agree with torch. That is not the
same sentence as "the page agrees with torch", because the page is a second
implementation: a JavaScript BPE, a hand-written pointer head, a chained scan that feeds
the graphs in their input layout, and one weight file mounted into two sessions. Each of
those four is a place where a wrong number appears while every file loads and no error is
raised. So this mutates them, one edit at a time, and requires `node
browser/selftest.mjs` — the same module the tab imports — to go red.

The mutations cluster in five places:

* **the tokenizer.** One id that differs shifts every option span after it, and a wrong
  span is a wrong decision printed as a right one. So the greedy rule, the merge window,
  the ByteLevel symbol range and the added-token split are all on trial.
* **the chain.** The document in `browser/expected.json` spans three scan calls on
  purpose: on a single-call document, re-reading the first chunk, restarting the
  positions and dropping the state between calls are all invisible. The empty document is
  the other way round — it is the input where the loop's own bound is load-bearing, since
  a zero-token document must still cost one padded call.
* **the head.** Temperature, the 1/sqrt(d) scale, the bias, mean-pooling over an option's
  span and the normalisation of that mean are five separate arithmetic claims, and the
  option spans in the expectations include a seven-token one so a mean is not the same as
  either endpoint.
* **the abstention gate.** `Myna` ships with `abstain_below = None`, so nothing else in
  this repo ever runs the JS gate with a floor. `browser/expected.json` carries answers at
  three probe floors — one where every answer commits, one that drops two of three, one
  that drops all three — so both branches of all three types run, and the score answer's
  expectation is compared as a number rather than as `null` on both sides.
* **the bytes and the mount.** The sharing claim is that `weights.bin` crosses the network
  once and is mounted by both sessions; a mount that names the file wrongly, and a
  transfer total that adds only the graph skeletons, are the two ways that claim survives
  its own artifact.

Deliberately absent, in both directions:

* **`browser/parity.mjs` itself.** Its thresholds are the instrument, and an instrument
  cannot be on trial against itself: loosening `maxGateAbs` would be caught only by an
  assertion written to catch it. What keeps those numbers honest is that each one is the
  same bound the Python side gates on (`onnx_export.parity`'s `max_rel_error` /
  `max_abs_error`, mutation-checked by `bench/mutation_onnx.py`), plus the comment at each
  site naming the drift it is sized against.
* **padding ids, as opposed to padding masks.** Feeding `[PAD]` (id 0) or an arbitrary id
  into a masked-out position is genuinely the same answer — the mask, not the id, is what
  makes padding inert, and the graph is built that way. Mutating the pad id would survive
  because it *should*, and a battery that reports a correct program as a survivor is worse
  than no battery. The two mutations above that touch padding change the mask instead, and
  both die.
* **the scan loop's `break`,** for the same reason and by measurement rather than by
  argument: the first run of this battery disabled it and the gate stayed green, because
  the loop bound (`fed < ids.length || fed === 0`) already stops an empty document after
  its one padded call. Two exits that each suffice is redundancy, not a bug, so the
  mutation was replaced by the one that kills the same property from the other side — the
  bound. `selftest.mjs`'s empty-document check is what makes that one die.

Same scratch-repo mechanics as the other batteries: copy `browser/` + `package.json`
(node needs `"type": "module"` to import `./myna.js` as ESM), symlink `node_modules` and
`runs` rather than copying either, green baseline required, abort on a first survivor, and
any pattern that does not match exactly once is reported instead of skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
SELFTEST = ["browser/selftest.mjs", "--repeats", "2"]
TIMEOUT = 180
BP, HD, MY = "browser/bpe.js", "browser/head.js", "browser/myna.js"

MUTATIONS = [
    # --- the tokenizer: one wrong id moves every span after it ----------------------
    ("the greedy merge takes the rarest pair instead of the commonest", BP,
     "        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }",
     "        if (r !== undefined && r > bestRank) { bestRank = r; best = i; }"),
    ("the merge never considers the word's last pair", BP,
     "      for (let i = 0; i < parts.length - 1; i++) {",
     "      for (let i = 0; i < parts.length - 2; i++) {"),
    ("byte-level symbols stop being shifted out of the text's own range", BP,
     "  return String.fromCharCode(0x100 + ByteLevelBPE.OTHER_BYTES.indexOf(b));",
     "  return String.fromCharCode(ByteLevelBPE.OTHER_BYTES.indexOf(b));"),
    ("an option's span ends one token early", BP,
     "      spans.push([start, ids.length]);",
     "      spans.push([start, ids.length - 1]);"),
    ("added tokens are split on but thrown away", BP,
     '      ? new RegExp(`(${this.added.map(([c]) => escapeRe(c)).join("|")})`, "g")',
     '      ? new RegExp(this.added.map(([c]) => escapeRe(c)).join("|"), "g")'),
    # --- the chain: three calls, and one document that must not hang ---------------
    ("the scan mask covers the padding too", MY,
     "      const mask = new Float32Array(chunk); mask.fill(1, 0, take.length);",
     "      const mask = new Float32Array(chunk).fill(1);"),
    ("every chunk is fed as if it started at position zero", MY,
     '                      pos_offset: new this.ort.Tensor("int64", BigInt64Array.of(BigInt(fed)), []),',
     '                      pos_offset: new this.ort.Tensor("int64", BigInt64Array.of(BigInt(0)), []),'),
    ("the layer stack comes back in the other order", MY,
     "      for (let i = 0; i < this.L; i++) S[i] = out[`S_out_${i}`];",
     "      for (let i = 0; i < this.L; i++) S[i] = out[`S_out_${this.L - 1 - i}`];"),
    ("each call re-reads the document's first chunk", MY,
     "      const take = ids.slice(fed, fed + chunk);",
     "      const take = ids.slice(0, chunk);"),
    ("the empty document gets no padded scan call at all", MY,
     "    for (let fed = 0; fed < ids.length || fed === 0; fed += chunk) {",
     "    for (let fed = 0; fed < ids.length; fed += chunk) {"),
    ("question padding is left unmasked", MY,
     "    const qMask = new Float32Array(N * Lq);",
     "    const qMask = new Float32Array(N * Lq).fill(1);"),
    # --- the head: five arithmetic claims, each its own ----------------------------
    ("the temperature is read and then not used", HD,
     "    logits.push(acc / Math.sqrt(dPtr) / temperature);",
     "    logits.push(acc / Math.sqrt(dPtr));"),
    ("the pointer scale multiplies by sqrt(d) instead of dividing", HD,
     "    logits.push(acc / Math.sqrt(dPtr) / temperature);",
     "    logits.push(acc * Math.sqrt(dPtr) / temperature);"),
    ("the projection biases are dropped", HD,
     "    let acc = bias.data[r];",
     "    let acc = 0;"),
    ("an option is its first token rather than its span's mean", HD,
     "    for (let t = s; t < e; t++) {",
     "    for (let t = s; t < s + 1; t++) {"),
    ("the span sum is never divided by the span's length", HD,
     "    for (let c = 0; c < dModel; c++) pooled[c] /= (e - s);",
     "    for (let c = 0; c < dModel; c++) pooled[c] /= 1;"),
    # --- the abstention gate -------------------------------------------------------
    ("the margin is reported as the top probability", HD,
     "  const margin = ordered[0] - (ordered.length > 1 ? ordered[1] : 0);",
     "  const margin = ordered[0];"),
    ("the gate commits on the margin instead of the confidence", HD,
     "  return { abstain: conf < threshold, confidence: conf, margin,",
     "  return { abstain: margin < threshold, confidence: conf, margin,"),
    ("a score becomes the sum of probabilities instead of their expectation", HD,
     "             score: gate.abstain ? null : probs.reduce((acc, p, j) => acc + j * p, 0),",
     "             score: gate.abstain ? null : probs.reduce((acc, p, j) => acc + p, 0),"),
    ("noul reports p(No) as the number the Brier score is defined on", HD,
     "  return { ...common, noul: probs[1], yes: gate.abstain ? null : probs[1] >= 0.5 };",
     "  return { ...common, noul: probs[0], yes: gate.abstain ? null : probs[0] >= 0.5 };"),
    ("a choice commits through the gate", HD,
     "    return { ...common, choice: gate.abstain ? null : labels[i],",
     "    return { ...common, choice: labels[i],"),
    ("a score level commits through the gate", HD,
     "             level: gate.abstain ? null : labels[i], probabilities,",
     "             level: labels[i], probabilities,"),
    # --- the mount and the byte count ----------------------------------------------
    ("the shared weights are mounted under a name neither graph asks for", MY,
     '                   externalData: [{ path: "weights.bin", data: weights }] };',
     '                   externalData: [{ path: "weights.onnx", data: weights }] };'),
    ("the transfer total counts only the graph skeletons", MY,
     "    const total = [...this.fetchedBytes.values()].reduce((a, b) => a + b, 0);",
     "    const total = [...this.fetchedBytes.entries()].filter(([k]) => k.endsWith(\".onnx\"))\n"
     "      .reduce((a, [, b]) => a + b, 0);"),
]


def make_scratch(tmp):
    repo = Path(tmp) / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    shutil.copytree(ROOT / "browser", repo / "browser",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "package.json", repo / "package.json")
    for name in ("node_modules", "runs"):
        if not (ROOT / name).exists():
            raise SystemExit(f"{name}/ is missing: the gate cannot run without it "
                             "(npm ci, and an exported artifact in runs/onnx)")
        (repo / name).symlink_to(ROOT / name)
    return repo


def selftest_in(repo):
    """Run the gate, and tell a red check from a hung one.

    No mutation in the list below is expected to hang — the empty-document case fails as a
    printed number now that the loop bound is the thing being mutated — but a hung gate
    would otherwise be indistinguishable from a slow one, and a battery that waits forever
    reports nothing. So a timeout counts as caught, with the reason printed rather than a
    bare nonzero status.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        return subprocess.run([os.environ.get("NODE", "node"), *SELFTEST],
                              capture_output=True, text=True, cwd=repo, env=env,
                              timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        class Hung:
            returncode = -1
            stdout = ""
            stderr = f"timed out after {TIMEOUT}s (hang)"
        return Hung()


def run_one(rel, old, new, tmp):
    repo = make_scratch(tmp)
    path = repo / rel
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        return f"BAD-PATTERN ({n} matches)", ""
    path.write_text(text.replace(old, new))
    r = selftest_in(repo)
    if r.returncode == 0:
        return "SURVIVED", ""
    bad = [ln for ln in r.stdout.splitlines() if ln.startswith("FAIL")]
    tail = [ln for ln in (r.stderr or "").strip().splitlines() if ln]
    detail = (bad[0] if bad else
              tail[-1] if tail else "exited nonzero with no FAIL line")
    return "caught", detail[:120]


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return 0
    survivors, bad = [], []
    with tempfile.TemporaryDirectory() as tmp:
        base = make_scratch(tmp)
        r0 = selftest_in(base)
        if r0.returncode != 0:
            print("baseline (unmutated) copy is RED -- the battery proves nothing")
            print(r0.stdout[-4000:] + r0.stderr[-2000:])
            return 2
        passed = sum(1 for ln in r0.stdout.splitlines() if ln.startswith("ok"))
        print(f"baseline copy: green ({passed} checks)\n")
        for i, (label, rel, old, new) in enumerate(MUTATIONS, 1):
            verdict, detail = run_one(rel, old, new, tmp)
            print(f"[{i:2d}/{len(MUTATIONS)}] {verdict:11s} {label}"
                  + (f"\n            {detail}" if detail and verdict == "caught" else ""),
                  flush=True)
            if verdict == "SURVIVED":
                survivors.append(label)
                if i == 1:
                    print("  first mutation survived: the battery is not testing the "
                          "mutated code; aborting", flush=True)
                    return 2
            elif verdict.startswith("BAD-PATTERN"):
                bad.append((label, verdict))
        print(f"\n{len(MUTATIONS) - len(survivors) - len(bad)}/{len(MUTATIONS)} mutations caught")
        for s in survivors:
            print(f"  SURVIVED: {s}")
        for b in bad:
            print(f"  {b}")
        return 1 if survivors or bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
