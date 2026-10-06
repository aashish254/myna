"""Independent witness for the Pages site: re-derive every printed literal, or refuse the page.

$ uv run python bench/site_witness.py

This file shares no data-gathering code with `bench/gen_site.py`: it re-reads the committed
artifacts, the two source files the demo copy quotes, `bench/gates.py` and `bench/reproduce.py`,
and asserts each derived value lands on the page in the position it should occupy. Positional
parsing is the point — a presence-only check is satisfied by a duplicate anywhere else in the
document, which is how a swapped pair of figures survives a green witness (§9.30).

Two further assertions hold the line this page was written to:

* **Every figure carries its honesty label.** `measured` may only sit on a value recomputed from
  an artifact; the target must read `gated — not met`; and the artifact's `pass` must still be
  the boolean the page prints.
* **No number without a source.** The last sweep tokenises the page's visible text and fails on
  any number not derived from a file — section citations included, checked against SPEC.md.

`bench/mutation_site.py` is the check on this check.
"""

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEMO_URL = "https://myna-demo.onrender.com"
REPO_URL = "https://github.com/aashish254/myna"

# the axis tick labels are a 0..1 domain, not a claim about the model
AXIS = ("0", "0.25", "0.5", "0.75", "1")


class Witness:
    def __init__(self, html):
        self.html = html
        self.checks = 0
        self.fails = []

    def ok(self, cond, what):
        self.checks += 1
        if not cond:
            self.fails.append(what)
        return cond

    def grab(self, pattern, text, what, flags=0):
        """A source file no longer prints the line the page was generated from. Refuse in words:
        a traceback is also a refusal, but it names the witness instead of the drift."""
        m = re.search(pattern, text, flags)
        if not m:
            self.fails.append(f"{what}: {pattern[:44]} no longer matches its source file")
        return m

    def cell(self, obj, path, what):
        """Same, for a JSON artifact's key path."""
        cur = obj
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                self.fails.append(f"{what}: {'/'.join(path)} is not in the committed artifact")
                return None
            cur = cur[k]
        return cur

    # ---------------------------------------------------------------- helpers
    def _load(self, rel):
        return json.loads((REPO / rel).read_text())

    def _import(self, rel, modname):
        spec = importlib.util.spec_from_file_location(modname, REPO / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    @staticmethod
    def fmt(x, nd=4):
        """The page's short display form: 4 dp, trailing zeros dropped."""
        s = f"{float(x):.{nd}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    @staticmethod
    def comma(x):
        return f"{int(x):,}"

    @staticmethod
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def text(self, fragment):
        """A cell's visible text with layout ignored entirely: the page may re-space the
        artifact's words, but the characters and their order must be the artifact's own."""
        return "".join(re.sub(r"<[^>]+>", " ", str(fragment)).split())

    def strip_markup(self, html):
        """Text nodes only: style, comments, tags and attributes are not visible numbers."""
        html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
        html = re.sub(r"<style>.*?</style>", " ", html, flags=re.S)
        return re.sub(r"<[^>]+>", " ", html)


def derive(w):
    """Every expectation, read out of the tree rather than remembered."""
    d = {}
    off = w._load("runs/antiprior_off_s0.report.json")
    on = w._load("runs/antiprior_on_s0c.report.json")
    v1b = w._load("runs/v1b_kaggle_3600b.report.json")

    def art(obj, *path):
        """`repr()` of a committed field, or '' after a legible refusal."""
        v = w.cell(obj, path, " · ".join(str(p) for p in path))
        return "" if v is None else repr(v)

    def artnum(obj, *path):
        """The same field in the display form the page uses for a count."""
        v = w.cell(obj, path, " · ".join(str(p) for p in path))
        return "" if v is None else str(v)

    d["acc"] = {"off": art(off, "macro", "acc"), "on": art(on, "macro", "acc"),
                "v1b": art(v1b, "macro", "acc")}
    # the page may display an arm short in prose, but only as this float at four decimals
    d["acc_short"] = {k: w.fmt(v) if v else "" for k, v in d["acc"].items()}
    d["margin"] = {k: art(v, "g1", "margin_observed")
                   for k, v in (("off", off), ("on", on), ("v1b", v1b))}
    d["witness_path"] = {"off": "runs/antiprior_off_s0.report.json",
                         "on": "runs/antiprior_on_s0c.report.json",
                         "v1b": "runs/v1b_kaggle_3600b.report.json"}
    d["floor"] = art(off, "macro", "majority")
    d["uniform"] = art(off, "macro", "uniform")
    d["target"] = art(off, "g1", "target")
    d["margin_req"] = art(off, "g1", "margin")
    d["g1_pass"] = w.cell(off, ("g1", "pass"), "off report")
    d["cells"] = art(off, "macro", "both", "cells").strip("'")
    d["laya"] = art(off, "macro", "both", "laya")
    d["delta4"] = (w.fmt(abs(float(d["acc"]["on"]) - float(d["acc"]["off"])))
                   if all(d["acc"].values()) else "")

    d["metrics"] = {"off": w._load("runs/antiprior_off_s0.metrics.json"),
                    "on": w._load("runs/antiprior_on_s0c.metrics.json"),
                    "v1b": w._load("runs/v1b_kaggle_3600b.metrics.json")}
    d["metrics_path"] = {"off": "runs/antiprior_off_s0.metrics.json",
                         "on": "runs/antiprior_on_s0c.metrics.json",
                         "v1b": "runs/v1b_kaggle_3600b.metrics.json"}
    d["executed"] = {}
    for k, m in d["metrics"].items():
        step = w.cell(m, ("last_step",), f"{k} metrics")
        d["executed"][k] = "" if step is None else w.comma(int(step) + 1)

    # each run prints its own launch line; the device and dose claims come from there
    d["launch"] = {}
    for tag, path in (("off", "runs/antiprior_off_s0.train.log"),
                      ("on", "runs/antiprior_on_s0c.train.log"),
                      ("v1b", "runs/v1b_kaggle_3600b.train.log")):
        txt = (REPO / path).read_text()
        line = w.grab(r"[\w./-]*python\d? -m myna\.train [^\n]*", txt,
                      f"{path}'s launch command")
        opts = {}
        for part in (line.group(0) if line else "").split("--")[1:]:
            toks = part.split()
            opts[toks[0]] = " ".join(toks[1:]).strip()
        # a Kaggle log prints the device it resolved; a local run's evidence is its own
        # `--device` flag. Either is a source; neither is remembered.
        gpu = re.search(r"^device: (.+)$", txt, re.M)
        device = (gpu.group(1) if gpu else opts.get("device")) or ""
        if not device:
            w.fails.append(f"{path} prints neither a `device:` line nor a --device flag")
        d["launch"][tag] = {"opts": opts, "path": path, "gpu": device}
        if line and not d["metrics"][tag].get("steps_requested"):
            assert d["executed"][tag] == w.comma(int(opts["steps"])), \
                f"{tag}: a completed run's executed count is not its --steps"
    d["requested"] = {tag: w.comma(d["metrics"][tag].get("steps_requested")
                                   or d["launch"][tag]["opts"].get("steps") or 0)
                      for tag in d["metrics"]}

    off_opts, on_opts = d["launch"]["off"]["opts"], d["launch"]["on"]["opts"]
    differing = sorted(k for k in set(off_opts) | set(on_opts)
                       if off_opts.get(k) != on_opts.get(k))
    d["pair_diff"] = ", ".join(f"--{k}" for k in differing)
    d["seed"] = off_opts.get("seed", "")
    d["batch"] = off_opts.get("batch", "")

    t0 = w._load("runs/tier0_antiprior_off_s0.json")
    d["params"] = artnum(t0, "model", "params")
    d["cfg"] = w.cell(t0, ("model", "cfg"), "model · cfg") or {}
    d["sha"] = artnum(t0, "model", "model_sha256")
    d["temperature"] = art(t0, "model", "temperature")

    qk = (REPO / "runs/quickstart_from_public_install.log").read_text()
    d["qk"] = {"choice": "", "confidence": "", "weak_label": "", "weak_prob": "",
               "params": "", "temperature": ""}
    for key, pat in (("choice", r"^(billing)$"),
                     ("qk3", r"^confidence (\S+) weakest (\S+) (\S+)$"),
                     ("qk2", r"^params (\d+) temperature (\S+) latency_ms")):
        m = w.grab(pat, qk, f"runs/quickstart_from_public_install.log's {key} line", re.M)
        if not m:
            continue
        if key == "choice":
            d["qk"]["choice"] = m.group(1)
        elif key == "qk3":
            d["qk"]["confidence"], d["qk"]["weak_label"], d["qk"]["weak_prob"] = m.groups()
        else:
            d["qk"]["params"], d["qk"]["temperature"] = m.groups()
    d["qk_logs"] = ["runs/quickstart_from_gh_download.log", "runs/quickstart_from_git_install.log",
                    "runs/quickstart_from_public_install.log"]

    py = (REPO / "pyproject.toml").read_text()
    for key, pat in (("dist", r'^name = "(.*)"'), ("version", r'^version = "(.*)"'),
                     ("python_req", r'^requires-python = "(.*)"')):
        m = w.grab(pat, py, f"pyproject.toml's {key}", re.M)
        d[key] = m.group(1) if m else ""

    d["gates"] = w._import("bench/gates.py", "w_gates").GATES
    d["rows"] = {r["id"]: r for r in w._import("bench/reproduce.py", "w_reproduce").ROWS}
    d["n_rows"] = len(d["rows"])

    d["serve"] = (REPO / "src/myna/serve.py").read_text()
    d["engine"] = (REPO / "src/myna/engine.py").read_text()
    none = w.grab(r'return \{"abstain": False, "confidence": None, '
                  r'"margin": None, "reason": None\}', d["engine"],
                  "engine.py's no-abstention return")
    d["abstain_none"] = none.group(0) if none else ""
    d["abstain_line"] = (d["engine"][:d["engine"].index(d["abstain_none"])].count("\n") + 1
                         if none else 0)
    d["routes"] = re.findall(r"^    (POST|DELETE) (\S+)", d["serve"], re.M)

    # the footer's date is the newest commit among the run artifacts the page reads: a stamp
    # taken from HEAD would be stale in the commit that carries it, so it is checked against
    # the figures' own sources rather than against the tree
    d["built"] = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "--",
         "runs/antiprior_off_s0.report.json", "runs/antiprior_on_s0c.report.json",
         "runs/v1b_kaggle_3600b.report.json", "runs/tier0_antiprior_off_s0.json",
         "runs/quickstart_from_gh_download.log", "runs/quickstart_from_git_install.log",
         "runs/quickstart_from_public_install.log"],
        capture_output=True, text=True).stdout.strip()
    if not d["built"]:
        w.fails.append("no commit date for the figure sources; the stamp cannot be checked")
    return d


# ---------------------------------------------------------------- assertions
def check_structure(w, d):
    h = w.html
    w.ok(h.startswith("<!DOCTYPE html>"), "page is not a standalone html document")
    w.ok("chart.js" not in h.lower() and "cdn.jsdelivr" not in h,
         "the page loads a charting CDN; a figure must not depend on a network")
    w.ok(f'href="{DEMO_URL}"' in h, f"live demo link {DEMO_URL} missing")
    w.ok(f'href="{REPO_URL}"' in h, "repository link missing")
    w.ok(f"pip install {d['dist']}=={d['version']}" in h,
         f"pip line is not `pip install {d['dist']}=={d['version']}`")
    w.ok(w.comma(d["params"]) in h, f"exact parameter count {w.comma(d['params'])} missing")
    c = d["cfg"]
    w.ok(f"{c['d_model']}d · {c['n_layers']} layers · {c['n_heads']} heads" in h,
         "the config line does not match runs/tier0_antiprior_off_s0.json model.cfg")
    w.ok(f"Python {w.esc(d['python_req'])}" in h,
         f"python requirement {d['python_req']} missing")
    w.ok(f"{int(d['params']) / 1e6:.1f}M" in h, "hero does not carry the derived M-scale form")


def check_quickstart_card(w, d):
    card = re.search(r'<div class="card" role="img".*?</div>\n</div>', w.html, re.S)
    if not w.ok(card is not None, "the released-bytes card is missing or its markup moved"):
        return
    body = card.group(0)
    printed = re.findall(r'translate="no">([^<]+)</span>', body)
    want = [d["qk"]["choice"], d["qk"]["confidence"], d["qk"]["weak_label"], d["qk"]["weak_prob"]]
    w.ok(printed[:4] == want, f"quickstart card reads {printed[:4]}, expected {want} in order")
    w.ok(d["qk"]["params"] in body and d["qk"]["temperature"] in body,
         "the card's params/temperature line moved")
    w.ok(d["qk"]["temperature"] == d["temperature"],
         f"quickstart temperature {d['qk']['temperature']} != artifact {d['temperature']}")
    w.ok(d["qk"]["params"] == str(d["params"]),
         f"quickstart params {d['qk']['params']} != tier0 {d['params']}")
    for logp in d["qk_logs"]:
        w.ok(logp in body, f"the card no longer cites {logp}")
        w.ok((REPO / logp).exists(), f"cited witness {logp} is not in the tree")


def check_arms(w, d):
    arms = re.findall(
        r'<li class="arm" data-arm="(\w+)">.*?'
        r'<div class="bar" style="width:([\d.]+)%"></div>.*?'
        r'<span class="arm-acc mono" translate="no">([^<]+)</span>.*?'
        r'<span class="arm-state (void|done)">(.*?)</span>.*?'
        r'([\d,]+) updates executed, ([\d,]+) requested on ([^·]+?) ·.*?'
        r'\+([\d.]+)</span> over the floor.*?'
        r'source <span translate="no">([^<]+)</span>', w.html, re.S)
    w.ok(len(arms) == 3, f"parsed {len(arms)} arm blocks, expected 3")
    tags = [a[0] for a in arms]
    w.ok(tags == ["off", "v1b", "on"], f"arm order is {tags}, expected off, v1b, on")
    for (tag, width, printed, state_cls, state_txt, executed, requested, gpu,
         marg, src) in arms:
        if tag not in d["acc"]:
            w.fails.append(f"arm {tag} is not a committed artifact")
            continue
        m = d["metrics"][tag]
        w.ok(printed == d["acc"][tag],
             f"{tag}: page prints {printed}, artifact macro.acc is {d['acc'][tag]}")
        w.ok(width == str(float(d["acc"][tag]) * 100),
             f"{tag}: bar width {width}% is not the same figure "
             f"({float(d['acc'][tag]) * 100}%) — a truncated axis is a claim")
        w.ok(marg == d["margin"][tag],
             f"{tag}: margin +{marg} is not the artifact's g1.margin_observed "
             f"{d['margin'][tag]} — a shortened margin is still a rounding")
        w.ok(executed == d["executed"][tag],
             f"{tag}: executed dose {executed} != last_step {m['last_step']} + 1")
        w.ok(requested == d["requested"][tag],
             f"{tag}: requested dose {requested} != {d['requested'][tag]}")
        w.ok(gpu.strip() == str(d["launch"][tag]["gpu"]),
             f"{tag}: device {gpu.strip()!r} != {d['launch'][tag]['gpu']!r} "
             f"from {d['launch'][tag]['path']}")
        w.ok(state_cls == ("void" if m.get("stopped") else "done"),
             f"{tag}: VOID state does not match metrics `stopped` ({m.get('stopped')!r})")
        if m.get("stopped"):
            w.ok("VOID" in state_txt and "§7.1" in state_txt,
                 f"{tag}: a truncated arm must print VOID and §7.1, not only a class name")
        else:
            w.ok("VOID" not in state_txt, f"{tag}: a complete run must not be called VOID")
        w.ok(src.startswith(d["witness_path"][tag]) and src.endswith("· macro.acc"),
             f"{tag}: source line {src!r} does not bind {d['witness_path'][tag]} · macro.acc")


def check_reference_values(w, d):
    tiles = re.findall(r'<h3>([^<]+) <span class="tag (\w+)">([^<]+)</span></h3>\s*'
                       r'<span class="big mono" translate="no">([^<]+)</span>', w.html)
    by_name = {t[0].strip(): t for t in tiles}
    w.ok(len(tiles) == 2, f"expected 2 reference tiles (floor, target), parsed {len(tiles)}")
    want = [("Majority floor", d["floor"], "measured", "runs/antiprior_off_s0.report.json · macro.majority"),
            ("G1 target", d["target"], "gated — not met", "runs/antiprior_off_s0.report.json · g1")]
    for name, value, label, source in want:
        if w.ok(name in by_name, f"reference tile {name!r} missing"):
            t = by_name[name]
            w.ok(t[3] == value, f"{name}: tile prints {t[3]}, artifact has {value}")
            w.ok(t[2] == label, f"{name}: honesty label is {t[2]!r}, must be {label!r}")
        w.ok(source in w.html, f"{name}: source line {source} missing")
    w.ok(d["g1_pass"] is False,
         "the artifact's g1.pass is no longer false — the page's label would be stale")
    w.ok(f'"pass": {json.dumps(d["g1_pass"])}' in w.html,
         'the page does not print the artifact\'s "pass": false')
    w.ok(f"+{d['margin_req']}" in w.html, f"the +{d['margin_req']} margin requirement is missing")
    w.ok(d["cells"] in w.html, f"the {d['cells']}-cell denominator is missing")
    w.ok(f"floor {w.fmt(d['floor'])}" in w.html and f"target {w.fmt(d['target'])}" in w.html,
         "the chart's floor/target markers do not print their values")
    w.ok(d["laya"] in w.html, f"the committed witness value {d['laya']} is missing")
    w.ok("runs/antiprior_off_s0.report.json · macro.both.laya" in w.html,
         "the laya figure is printed without its source key")
    w.ok(d["uniform"] not in w.html,
         "the uniform floor is printed but the page never explains it; leave it out")
    # the floor must not be painted as a pass
    floor_tile = re.search(r'<h3>Majority floor.*?</div>', w.html, re.S)
    if floor_tile:
        w.ok("0.7" not in floor_tile.group(0) or "target" in floor_tile.group(0),
             "the floor tile reads as a target")


def check_antiprior(w, d):
    pair = re.search(r"A sampler flag meant to break label priors.*?</p>", w.html, re.S)
    if not w.ok(pair is not None, "the anti-prior paragraph is gone"):
        return
    body = pair.group(0)
    w.ok(d["acc"]["on"] in body and d["acc"]["off"] in body,
         "the pair's two macros are not both printed")
    w.ok(f"&minus;{d['delta4']}" in body, f"the cost is not printed as −{d['delta4']}")
    w.ok(d["pair_diff"] in body, f"the computed option diff {d['pair_diff']!r} is not on the page")
    w.ok(f"--seed {d['seed']}" in body, "the shared seed is not printed")
    w.ok(f"--batch {d['batch']}" in body, "the shared batch is not printed")
    w.ok(d["executed"]["on"] in body, f"the executed dose {d['executed']['on']} is not printed")
    for path in (d["witness_path"]["on"], d["witness_path"]["off"], d["metrics_path"]["on"],
                 d["metrics_path"]["off"], d["launch"]["on"]["path"], d["launch"]["off"]["path"]):
        w.ok(path in body, f"the pair no longer cites {path}")


def check_gates(w, d):
    rows = re.findall(r'<tr class="gate" data-gate="(\w+)">\s*'
                      r'<td class="mono">(\w+)</td>\s*<td>(.*?)</td>\s*'
                      r'<td class="verdict"><span class="tag (\w+)">([^<]+)</span></td>\s*'
                      r'<td class="note">(.*?)</td>', w.html, re.S)
    w.ok(len(rows) == len(d["gates"]),
         f"table has {len(rows)} rows, gates.py has {len(d['gates'])}")
    for got, g in zip(rows, d["gates"]):
        gid, gcol, gname, tagcls, verdict, cond = got
        w.ok(gid == g["id"] and gcol == g["id"],
             f"row order: page has {gid}/{gcol}, gates.py has {g['id']}")
        # gates.py marks its own copy with markdown; the page renders it, so compare text
        w.ok(w.text(gname) == w.text(g["name"]).replace("*", ""),
             f"{g['id']}: name drifted from gates.py ({w.text(gname)!r})")
        w.ok(verdict == g["verdict"],
             f"{g['id']}: page says {verdict!r}, gates.py says {g['verdict']!r}")
        w.ok(w.text(cond) == w.text(g["condition"]).replace("`", ""),
             f"{g['id']}: condition drifted from gates.py")
        exp_cls = {"met": "measured", "not met": "notmet", "open": "open"}[g["verdict"]]
        w.ok(tagcls == exp_cls, f"{g['id']}: verdict class {tagcls}, expected {exp_cls}")
    tally = {v: sum(1 for g in d["gates"] if g["verdict"] == v)
             for v in ("met", "not met", "open")}
    w.ok(f"{tally['met']} met, {tally['not met']} not met, {tally['open']} open" in w.html,
         "the verdict tally no longer matches gates.py")
    word = "one two three four five six seven eight nine ten eleven twelve"
    w.ok(f"all {word.split()[len(d['gates']) - 1]} rows" in w.html,
         f"the heading's count does not match the {len(d['gates'])}-row table")
    w.ok("*" not in re.search(r"<tbody>.*?</tbody>", w.html, re.S).group(0),
         "a markdown marker from gates.py reached the page")
    w.ok("bench/gates.py" in w.html, "the table does not name bench/gates.py as its source")
    w.ok("make gates" in w.html, "the table does not point at `make gates`")


def check_demo(w, d):
    demo = re.search(r'<section id="demo">.*?</section>', w.html, re.S)
    if not w.ok(demo is not None, "the demo section is gone"):
        return
    # the capability items carry nested lists, so the boundaries are the markers, not `</li>`
    parts = re.split(r'<li class="cap" data-cap="([\w-]+)">', demo.group(0))
    caps = {parts[i]: parts[i + 1].split("\n  </ul>")[0] for i in range(1, len(parts) - 1, 2)}
    w.ok(len(caps) == 2, f"expected 2 demo capabilities, parsed {len(caps)}")
    if w.ok("scan-once" in caps, "the scan-once capability is missing"):
        w.ok("scan a state once and ask" in d["serve"],
             "serve.py no longer prints the scan-once wording the page quotes")
        w.ok("scan a state once and ask" in caps["scan-once"],
             "the capability copy no longer quotes serve.py's own phrase")
        w.ok("src/myna/serve.py" in caps["scan-once"], "scan-once is not sourced to serve.py")
        for method, path in d["routes"]:
            w.ok(f"{method} {path}" in caps["scan-once"], f"route {method} {path} not printed")
    if w.ok("abstain" in caps, "the abstention capability is missing"):
        body = caps["abstain"]
        w.ok("n/a" in body, "the abstention capability must render n/a, not a number")
        w.ok(d["abstain_none"] in body,
             f"the page does not quote engine.py's own return ({d['abstain_none']})")
        w.ok(f"src/myna/engine.py:{d['abstain_line']}" in body,
             f"the abstention citation must be src/myna/engine.py:{d['abstain_line']}")
        w.ok("--abstain-below" in d["serve"] and "--abstain-below" in body,
             "--abstain-below is not on both the page and serve.py")
    # the abstention caveat is a section-level honesty note, so it is checked at that scope
    w.ok("G5" in demo.group(0),
         "the demo section must name the gate its abstention floor does not satisfy")
    w.ok(d["sha"] in w.html, "the demo block does not name the checkpoint's sha256")
    w.ok(d["temperature"] in w.html, "the demo block does not print the temperature")


def check_reproduce(w, d):
    for row_id in ("antiprior-off-macro", "antiprior-on-macro", "v1b-kaggle-macro"):
        cmd = d["rows"][row_id]["cmd"]
        w.ok(cmd in w.html, f"registry command for {row_id} is not printed verbatim")
        for art in d["rows"][row_id]["witness"]:
            w.ok(art in w.html, f"{row_id}'s witness {art} is not cited on the page")
    w.ok("bench/reproduce.py" in w.html, "the reproduce block does not name the registry")
    w.ok(str(d["n_rows"]) in w.html, f"the registry's row count ({d['n_rows']}) is not printed")
    w.ok("uv run python bench/gen_site.py" in w.html, "the page does not name its own generator")
    w.ok("uv run python bench/site_witness.py" in w.html, "the page does not name its witness")
    for label in ("measured", "gated — not met", "not met"):
        w.ok(label in w.html, f"the honesty label {label!r} is missing")
    w.ok("make repro" in w.html and "make secrets" in w.html,
         "the reproduce block does not offer `make repro` and `make secrets`")
    w.ok(f"Figures as of {d['built']}" in w.html,
         f"the footer's date is not the newest commit among the figure sources ({d['built']})")


def check_no_unsourced_number(w, d):
    """Every visible number on the page must be one a file printed."""
    allowed = set()

    def allow(*values):
        for v in values:
            for tok in re.findall(r"\d[\d.]*", str(v)):
                allowed.add(tok.rstrip("."))

    allow(*AXIS)
    allow(d["built"])
    allow(*d["acc_short"].values())
    allow(d["params"], w.comma(d["params"]), d["temperature"], d["version"], d["python_req"],
          d["cells"], d["floor"], w.fmt(d["floor"]), d["target"], w.fmt(d["target"]),
          d["margin_req"], d["laya"], d["delta4"], d["uniform"], d["sha"], d["n_rows"],
          f"{int(d['params']) / 1e6:.1f}", d["dist"], d["pair_diff"], d["seed"], d["batch"])
    allow(*d["cfg"].values())
    for tag in d["acc"]:
        # no short margin form is allowed anywhere: the only margin the page may print is stored
        allow(d["acc"][tag], d["margin"][tag], d["executed"][tag],
              d["requested"][tag], d["metrics"][tag]["last_step"],
              w.comma(d["metrics"][tag]["last_step"]),
              d["metrics"][tag].get("steps_requested"), d["witness_path"][tag],
              d["metrics_path"][tag], d["launch"][tag]["path"])
        for flag, val in d["launch"][tag]["opts"].items():
            allow(f"{flag} {val}")
    allow(*d["qk"].values())
    allow(*d["qk_logs"])
    for g in d["gates"]:
        allow(g["id"], g["name"], g["condition"], g["verdict"])
    for row_id in ("antiprior-off-macro", "antiprior-on-macro", "v1b-kaggle-macro"):
        allow(d["rows"][row_id]["cmd"], d["rows"][row_id]["witness"])
    allow(d["abstain_none"], f"src/myna/engine.py:{d['abstain_line']}", "src/myna/serve.py",
          "src/myna/engine.py", "bench/gates.py", "bench/reproduce.py", "bench/gen_site.py",
          "bench/site_witness.py", "myna-weights", "antiprior_off_s0-weights",
          "antiprior_on_s0c-weights", "v1b-checkpoint", "data/decision-v2-pilot", DEMO_URL,
          REPO_URL, *[f"{m} {p}" for m, p in d["routes"]])

    text = w.strip_markup(w.html)
    citations = re.findall(r"§([\d.]+)", text)
    text = re.sub(r"§[\d.]+", " ", text)          # checked against SPEC.md below
    for tok in re.findall(r"\d[\d.]*", text):
        w.checks += 1
        if tok.rstrip(".") not in allowed:
            w.fails.append(f"unsourced number on the page: {tok}")

    spec_txt = (REPO / "SPEC.md").read_text()
    for s in sorted(set(citations)):
        w.ok(f"§{s}" in spec_txt, f"the page cites SPEC §{s}, which is not in SPEC.md")


def run(html):
    w = Witness(html)
    d = derive(w)
    if w.fails:
        # an artifact or source stopped saying what the page was generated from; comparing the
        # page against half-derived expectations would only bury that message
        w.fails.append("(the page was not compared: its sources moved first — see above)")
        return w
    check_structure(w, d)
    check_quickstart_card(w, d)
    check_arms(w, d)
    check_reference_values(w, d)
    check_antiprior(w, d)
    check_gates(w, d)
    check_demo(w, d)
    check_reproduce(w, d)
    check_no_unsourced_number(w, d)
    # a witness that quietly stops checking is worse than none
    if w.checks < 180:
        w.fails.append(f"check count fell to {w.checks}, floor is 180 — coverage shrank")
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default=str(REPO / "docs" / "index.html"),
                    help="the page to check (bench/mutation_site.py points it at a patched copy)")
    args = ap.parse_args()
    w = run(Path(args.html).read_text())
    print(f"{w.checks} checks, {len(w.fails)} failures")
    for line in w.fails:
        print(f"  FAIL {line}")
    if w.fails:
        raise SystemExit(1)
    print("docs/index.html: every printed figure re-derived from runs/*.report.json, "
          "runs/*.metrics.json, runs/*.train.log, runs/tier0_antiprior_off_s0.json, "
          "src/myna/{serve,engine}.py, bench/{gates,reproduce}.py and pyproject.toml")


if __name__ == "__main__":
    main()
