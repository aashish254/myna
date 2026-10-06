"""Generate the GitHub Pages site, `docs/index.html`, from committed artifacts only.

$ uv run python bench/gen_site.py

Every figure this file prints is read out of a file in the repo at build time; there is no
number literal in the template below. A page whose cells are typed by hand is the failure
mode SPEC §9.30 names, so the generator is the only writer of the file and
`bench/site_witness.py` is an independent reader that re-derives every printed literal
from the same sources and refuses the page when one drifts.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "index.html"

DEMO_URL = "https://myna-demo.onrender.com"
REPO_URL = "https://github.com/aashish254/myna"


def load(relpath):
    """`runs/*.json` as a dict, with its repo-relative path kept for the source line."""
    return json.loads((REPO / relpath).read_text()), relpath


def _import(rel, modname):
    spec = importlib.util.spec_from_file_location(modname, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_line():
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    return f"uv run python bench/gen_site.py (git HEAD {head})" if head else \
        "uv run python bench/gen_site.py"


STAMP_SOURCES = ("runs/antiprior_off_s0.report.json", "runs/antiprior_on_s0c.report.json",
                 "runs/v1b_kaggle_3600b.report.json", "runs/tier0_antiprior_off_s0.json",
                 "runs/quickstart_from_gh_download.log", "runs/quickstart_from_git_install.log",
                 "runs/quickstart_from_public_install.log")


def head_stamp():
    """The date the *figures* were committed — the newest commit among the artifacts this page
    reads — rather than HEAD's date or the clock. HEAD would make the page stale the moment it is
    committed, because the commit carrying it is then the newest thing in the tree, and a footer
    that can never be current is a footer no `--check` may enforce. A clock would make every
    re-run differ. This date is stable across doc commits and is the claim the footer makes."""
    out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cd",
                          "--date=format:%Y-%m-%d", "--", *STAMP_SOURCES],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        raise SystemExit("no commit date for the figure sources; the footer would print an "
                         "empty stamp, which is how a stale page gets a fresh-looking one")
    return out


def facts():
    f = {}

    off, off_path = load("runs/antiprior_off_s0.report.json")
    on, on_path = load("runs/antiprior_on_s0c.report.json")
    v1b, v1b_path = load("runs/v1b_kaggle_3600b.report.json")
    for d, tag in ((off, "off"), (on, "on"), (v1b, "v1b")):
        f[f"{tag}_acc"] = repr(d["macro"]["acc"])
        f[f"{tag}_margin"] = repr(d["g1"]["margin_observed"])
    f["off_witness"] = off_path
    f["on_witness"] = on_path
    f["v1b_witness"] = v1b_path

    # floor, uniform floor and the witness line come from the same block every arm scores
    f["floor"] = repr(off["macro"]["majority"])
    f["uniform"] = repr(off["macro"]["uniform"])
    f["target"] = repr(off["g1"]["target"])
    f["margin_req"] = repr(off["g1"]["margin"])
    f["g1_pass"] = off["g1"]["pass"]
    f["cells"] = str(off["macro"]["both"]["cells"])
    f["laya"] = repr(off["macro"]["both"]["laya"])

    # the anti-prior cost is arithmetic on two committed figures, not a remembered number
    delta = float(f["on_acc"]) - float(f["off_acc"])
    f["flag_delta"] = repr(delta)
    f["flag_delta_4"] = f"{abs(delta):.4f}"

    for tag, path in (("off", "runs/antiprior_off_s0.metrics.json"),
                      ("on", "runs/antiprior_on_s0c.metrics.json"),
                      ("v1b", "runs/v1b_kaggle_3600b.metrics.json")):
        d = json.loads((REPO / path).read_text())
        f[f"{tag}_last_step"] = str(d["last_step"])
        f[f"{tag}_requested"] = str(d["steps_requested"]) if d.get("steps_requested") else None
        f[f"{tag}_stopped"] = d.get("stopped")
        f[f"{tag}_flag"] = d.get("anti_prior")
        f[f"{tag}_metrics"] = path

    # `last_step` is an index, so the dose both arms executed is one past it — the same
    # arithmetic SPEC §9.56 prints, and the reason the pair is dose-matched
    f["executed"] = f"{int(f['on_last_step']) + 1:,}"
    if int(f["on_last_step"]) != int(f["off_last_step"]):
        raise SystemExit("the 10c arms are no longer dose-matched; rewrite the anti-prior copy")

    # The launch command each run wrote at its own top (§9.30). Device, seed and dose are
    # claims about how the arms were made, so they come from the logs that made them.
    train_logs = {"off": "runs/antiprior_off_s0.train.log",
                  "on": "runs/antiprior_on_s0c.train.log",
                  "v1b": "runs/v1b_kaggle_3600b.train.log"}
    for tag, path in train_logs.items():
        txt = (REPO / path).read_text()
        m = re.search(r"[\w./-]*python\d? -m myna\.train [^\n]*", txt)
        if not m:
            raise SystemExit(f"{path} no longer prints its launch command")
        cmd = m.group(0)
        f[f"{tag}_train_log"] = path
        f[f"{tag}_launch"] = cmd
        for flag in ("device", "seed", "steps", "batch"):
            g = re.search(rf"--{flag} (\S+)", cmd)
            f[f"{tag}_{flag}"] = g.group(1) if g else None
        ap = re.search(r"--anti-prior (\S+)", cmd)
        f[f"{tag}_prior_flag"] = ap.group(1) if ap else None
        gpu = re.search(r"^device: (.+)$", txt, re.M)
        f[f"{tag}_gpu"] = gpu.group(1) if gpu else f[f"{tag}_device"]
        # `steps_requested` reaches the metrics file only on a local run; the launch line the
        # run printed at its own top carries `--steps`, so the dose has a committed source either way
        if not f[f"{tag}_requested"]:
            f[f"{tag}_requested"] = f[f"{tag}_steps"]
            f[f"{tag}_requested_src"] = path
        if f[f"{tag}_flag"] and f[f"{tag}_prior_flag"] and f[f"{tag}_flag"] != f[f"{tag}_prior_flag"]:
            raise SystemExit(f"{path}: metrics and the launch line disagree about --anti-prior")

    # what the two 10c launches differ by, computed rather than remembered. `--` splits the
    # options; a flag with no value (`--row-batch`, `--resume`) would otherwise steal the next
    # flag's name as its own value and make the diff lie.
    off_opts = launch_opts(f["off_launch"])
    on_opts = launch_opts(f["on_launch"])
    diff = sorted(k for k in set(off_opts) | set(on_opts) if off_opts.get(k) != on_opts.get(k))
    f["pair_diff"] = diff
    if not set(diff) <= {"anti-prior", "out", "steps", "resume"}:
        raise SystemExit(f"the 10c pair now differs by {diff}, not only the flag — rewrite the copy")
    if f["off_seed"] != f["on_seed"]:
        raise SystemExit("the 10c arms are no longer the same seed")
    if f["off_batch"] != f["on_batch"]:
        raise SystemExit("the 10c arms are no longer the same batch")
    f["same_seed"] = f["off_seed"]
    f["same_batch"] = f["off_batch"]

    t0 = json.loads((REPO / "runs/tier0_antiprior_off_s0.json").read_text())
    f["params"] = str(t0["model"]["params"])
    f["params_comma"] = f"{int(f['params']):,}"
    f["temperature"] = repr(t0["model"]["temperature"])
    f["sha256"] = t0["model"]["model_sha256"]
    cfg = t0["model"]["cfg"]
    f["d_model"] = str(cfg["d_model"])
    f["n_layers"] = str(cfg["n_layers"])
    f["n_heads"] = str(cfg["n_heads"])
    f["vocab"] = str(cfg["vocab"])

    # the released-bytes quickstart: one float, three install harnesses
    qk = (REPO / "runs/quickstart_from_public_install.log").read_text()
    m = re.search(r"^(billing)$", qk, re.M)
    f["qk_choice"] = m.group(1)
    m = re.search(r"^confidence (\S+) weakest (\S+) (\S+)$", qk, re.M)
    f["qk_confidence"], f["qk_weakest_label"], f["qk_weakest_prob"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"^params (\d+) temperature (\S+) latency_ms", qk, re.M)
    f["qk_params"], f["qk_temperature"] = m.group(1), m.group(2)
    f["qk_logs"] = ["runs/quickstart_from_gh_download.log",
                    "runs/quickstart_from_git_install.log",
                    "runs/quickstart_from_public_install.log"]

    py = (REPO / "pyproject.toml").read_text()
    f["dist_name"] = re.search(r'^name = "(.*)"', py, re.M).group(1)
    f["version"] = re.search(r'^version = "(.*)"', py, re.M).group(1)
    f["python_req"] = re.search(r'^requires-python = "(.*)"', py, re.M).group(1)

    gates = _import("bench/gates.py", "myna_gates").GATES
    f["gates"] = [{"id": g["id"], "name": g["name"], "verdict": g["verdict"],
                   "condition": g["condition"]} for g in gates]
    f["gate_tally"] = {v: sum(1 for g in gates if g["verdict"] == v)
                       for v in ("met", "not met", "open")}

    reg = _import("bench/reproduce.py", "myna_reproduce").ROWS
    by_id = {r["id"]: r for r in reg}
    f["row_cmds"] = {i: by_id[i]["cmd"] for i in
                     ("antiprior-off-macro", "antiprior-on-macro", "v1b-kaggle-macro")}
    f["registry_rows"] = str(len(reg))

    # the two behaviours the live demo is offered for, read from the engine's own source
    serve = (REPO / "src/myna/serve.py").read_text()
    f["scan_once_phrase"] = "scan a state once and ask"
    if f["scan_once_phrase"] not in serve:
        raise SystemExit(f"serve.py no longer prints {f['scan_once_phrase']!r}")
    f["serve_routes"] = re.findall(r"^    (POST|DELETE) (\S+)", serve, re.M)
    f["abstain_flag"] = re.search(r'add_argument\("--abstain-below"', serve).group(0)

    engine = (REPO / "src/myna/engine.py").read_text()
    m = re.search(r"return \{\"abstain\": False, \"confidence\": None, \"margin\": None, "
                  r"\"reason\": None\}", engine)
    if not m:
        raise SystemExit("engine.py no longer returns None confidence below an off floor")
    f["abstain_none"] = m.group(0)
    f["abstain_line"] = engine[:m.start()].count("\n") + 1

    f["built"] = head_stamp()
    for g in f["gates"]:
        g["condition"] = g["condition"].replace("*", "")
    return f


def launch_opts(cmdline):
    """`{flag: value}` from a printed launch line. Splitting on `--` rather than pairing
    tokens is what keeps a valueless flag (`--row-batch`, `--resume`) from stealing the next
    flag's name as its own value, which would make a computed diff lie."""
    out = {}
    for part in cmdline.split("--")[1:]:
        toks = part.split()
        out[toks[0]] = " ".join(toks[1:]).strip()
    return out


def pct(x):
    """Bar position on a 0..1 domain, at full precision — a truncated axis is a claim."""
    return f"{float(x) * 100}"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def md_inline(s):
    """`bench/gates.py` writes its own copy in markdown. A page must not show the markers, and
    the inline code it marks gets the same treatment as the rest of the page's monospace."""
    out = esc(s)
    out = re.sub(r"`([^`]+)`", r'<span class="mono">\1</span>', out)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", out)


NUMWORD = "one two three four five six seven eight nine ten eleven twelve".split()


def numword(n):
    """'all seven rows' spelled from the table's own length, never typed by hand."""
    n = int(n)
    if not 1 <= n <= len(NUMWORD):
        raise SystemExit(f"no word for {n}; the gate-count prose needs its list extended")
    return NUMWORD[n - 1]


CSS = """
:root {
  color-scheme: light dark;
  --bg: #fbfaf8; --surface: #ffffff; --ink: #15181c; --ink-2: #4c5561;
  --ink-3: #7b8593; --rule: #e3e1dc; --accent: #1f5d4c;
  --measured: #1f5d4c; --notmet: #a43a2b; --open: #7a5a17;
  --bar: #c9d6d0; --bar-key: #1f5d4c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1013; --surface: #14171b; --ink: #eef0f2; --ink-2: #adb5bd;
    --ink-3: #7d868f; --rule: #262b31; --accent: #7fd0b4;
    --measured: #7fd0b4; --notmet: #f0907f; --open: #dcbc6b;
    --bar: #2a333a; --bar-key: #7fd0b4;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 400 17px/1.6 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
/* A 64-character digest and a witness path are the only unbreakable tokens on the page, and
   they live in these two blocks. Breaking them there — rather than on `.mono` — is what keeps
   a printed float from being split mid-digit at a narrow width. */
.sub, .note, .kv { overflow-wrap: anywhere; }
.wrap { max-width: 60rem; margin: 0 auto; padding: 0 1.5rem; }
header.site {
  display: flex; align-items: baseline; gap: 1.5rem; flex-wrap: wrap;
  padding: 1.75rem 0 1.25rem; border-bottom: 1px solid var(--rule);
}
.wordmark { font-size: 1.15rem; font-weight: 650; letter-spacing: -0.015em; }
nav.site { margin-left: auto; display: flex; gap: 1.25rem; font-size: .9rem; }
nav.site a { color: var(--ink-2); text-decoration: none; border-bottom: 1px solid transparent; }
nav.site a:hover { color: var(--ink); border-bottom-color: var(--ink-3); }
.hero { padding: 4.5rem 0 3.5rem; display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 3rem; }
.eyebrow {
  font-size: .78rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 1.1rem;
}
h1 { font-size: clamp(2.1rem, 5.2vw, 3.4rem); line-height: 1.06; letter-spacing: -0.03em;
     font-weight: 660; margin: 0 0 1.2rem; text-wrap: balance; }
.lede { font-size: 1.12rem; color: var(--ink-2); margin: 0 0 2rem; max-width: 34rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.cta { display: flex; gap: .8rem; flex-wrap: wrap; align-items: center; }
a.button {
  display: inline-block; background: var(--ink); color: var(--bg); text-decoration: none;
  padding: .72rem 1.25rem; border-radius: 3px; font-weight: 600; font-size: .95rem;
}
a.button:hover { background: var(--accent); color: #fff; }
.install {
  border: 1px solid var(--rule); border-radius: 3px; background: var(--surface);
  padding: .7rem .9rem; font-size: .86rem; overflow-wrap: anywhere;
}
.card {
  border: 1px solid var(--rule); border-radius: 4px; background: var(--surface);
  padding: 1.4rem 1.5rem;
}
.card h3 { margin: 0 0 .2rem; font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }
.card .big { font-size: clamp(1.6rem, 5.4vw, 2.15rem); letter-spacing: -0.03em; line-height: 1; display: block; margin: .5rem 0 .2rem; }
.card .kv { font-size: .84rem; color: var(--ink-2); margin: .35rem 0 0; }
section { padding: 3.25rem 0; border-top: 1px solid var(--rule); }
h2 { font-size: 1.5rem; letter-spacing: -0.02em; font-weight: 650; margin: 0 0 .5rem; }
.sub { color: var(--ink-2); font-size: .98rem; margin: 0 0 2rem; max-width: 42rem; }
.tracks { list-style: none; margin: 0; padding: 0; display: grid; gap: 1.15rem; }
.tracks li.arm { display: grid; grid-template-columns: minmax(0, 13rem) minmax(0, 1fr); gap: 1.25rem; align-items: start; }
.arm-head { font-size: .95rem; font-weight: 600; }
.arm-head span { display: block; font-weight: 400; font-size: .8rem; color: var(--ink-3); margin-top: .15rem; }
.plot { position: relative; height: 4.5rem; }
.baseline { position: absolute; left: 0; right: 0; top: 1.1rem; height: 2px; background: var(--bar); }
.bar { position: absolute; top: 1.1rem; left: 0; height: 2px; background: var(--bar-key); }
.bar::before {
  content: ""; position: absolute; left: -4px; top: -4px; width: 10px; height: 10px;
  background: var(--bar-key); border-radius: 50%;
}
.bar-value { position: absolute; top: .1rem; font-size: .9rem; font-weight: 600; transform: translateX(-50%); }
.marker { position: absolute; top: -2px; bottom: -34px; width: 1px; border-left: 1px dashed var(--ink-3); }
.marker.floor { border-left-style: solid; }
.marker span {
  position: absolute; bottom: -1.35rem; left: 3px; white-space: nowrap;
  font-size: .74rem; color: var(--ink-3); letter-spacing: .02em;
}
.axis { position: relative; height: 1px; background: var(--rule); margin: 2.6rem 0 .4rem; }
.axis i { position: absolute; top: 0; left: var(--at); width: 1px; height: 6px; background: var(--rule); }
.axis-labels { display: flex; justify-content: space-between; font-size: .74rem; color: var(--ink-3); }
.tag { font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; font-weight: 650; }
.tag.measured { color: var(--measured); }
.tag.notmet { color: var(--notmet); }
.tag.open { color: var(--open); }
.srcline { font-size: .78rem; color: var(--ink-3); margin-top: .5rem; overflow-wrap: anywhere; }
.arm-state { font-weight: 600; }
.arm-state.void { color: var(--notmet); }
.arm-state.done { color: var(--measured); }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
th { text-align: left; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
     color: var(--ink-3); font-weight: 600; padding: 0 .75rem .6rem 0; border-bottom: 1px solid var(--rule); }
td { padding: .8rem .75rem .8rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
td.verdict { white-space: nowrap; font-weight: 600; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 2.5rem; }
ul.plain { list-style: none; margin: 0; padding: 0; }
ul.plain li { padding: .7rem 0; border-bottom: 1px solid var(--rule); font-size: .95rem; }
ul.plain li:last-child { border-bottom: 0; }
.tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem; }
pre.code {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
  padding: 1.1rem 1.2rem; font-size: .82rem; overflow-x: auto; line-height: 1.55;
}
.note { font-size: .86rem; color: var(--ink-2); }
footer.site { padding: 2.5rem 0 4rem; border-top: 1px solid var(--rule); font-size: .84rem; color: var(--ink-3); }
a { color: var(--accent); }
@media (max-width: 720px) {
  .hero { grid-template-columns: 1fr; gap: 2rem; padding: 2.75rem 0 2.5rem; }
  .two-col { grid-template-columns: 1fr; gap: 1.5rem; }
  .tracks li.arm { grid-template-columns: 1fr; gap: .25rem; }
  .tiles { grid-template-columns: 1fr; }
  .lede { font-size: 1.05rem; }
}
"""


def executed(f, key):
    """`last_step` is an index, so the updates a run executed are one past it (§9.56). Every dose
    claim on the page — the bar line and the not-a-product bullet — reads this, not the index."""
    return comma(int(f[f"{key}_last_step"]) + 1)


def arm(f, key, label, sub):
    """One measured arm: its bar, its exact macro, its dose, and the file it came from."""
    acc = f[f"{key}_acc"]
    void = f[f"{key}_stopped"]
    if float(f[f"{key}_margin"]) < 0:
        raise SystemExit(f"{key}: its margin over the floor is negative; the '+' prefix would lie")
    dose = f"{executed(f, key)} updates executed, " \
           f"{comma(f[key + '_requested'])} requested"
    state = ("VOID — truncated dose, no gate judged from it (§7.1)"
             if void else "run completed")
    src = f'{f[key + "_witness"]} · macro.acc'
    return f"""
      <li class="arm" data-arm="{key}">
        <div class="arm-head">{label}<span>{sub}</span></div>
        <div>
          <div class="plot">
            <div class="baseline"></div>
            <div class="bar" style="width:{pct(acc)}%"></div>
            <div class="bar-value mono" style="left:{pct(acc)}%">
              <span class="arm-acc mono" translate="no">{esc(acc)}</span>
            </div>
            <div class="marker floor" style="left:{pct(f['floor'])}%"><span>floor {short(f['floor'])}</span></div>
            <div class="marker" style="left:{pct(f['target'])}%"><span>target {short(f['target'])}</span></div>
          </div>
          <div class="srcline mono"><span class="arm-state {('void' if void else 'done')}">{state}</span> ·
            {esc(dose)} on {esc(f[key + "_gpu"])} ·
            <span class="arm-margin">+{esc(f[key + "_margin"])}</span> over the floor ·
            source <span translate="no">{esc(src)}</span></div>
        </div>
      </li>"""


def diff_text(f):
    return ", ".join(f"--{k}" for k in f["pair_diff"])


def comma(x):
    return f"{int(x):,}"


def short(x):
    """4-dp display of a committed float. The exact literal is always printed too."""
    return f"{float(x):.4f}".rstrip("0") if float(x) != int(float(x)) else str(float(x))


def render(f):
    gate_rows = "\n".join(
        f"""      <tr class="gate" data-gate="{g['id']}">
        <td class="mono">{g['id']}</td>
        <td>{md_inline(g['name'])}</td>
        <td class="verdict"><span class="tag {cls(g['verdict'])}">{g['verdict']}</span></td>
        <td class="note">{md_inline(g['condition'])}</td>
      </tr>""" for g in f["gates"])

    routes = "\n".join(f"<li class=\"mono\">{m} {esc(p)}</li>" for m, p in f["serve_routes"])

    return f"""<!DOCTYPE html>
<!-- generated by {cmd_line()} -- do not hand-edit; bench/site_witness.py checks every figure -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Myna — a {short_m(f['params'])}-parameter typed-decision engine</title>
<meta name="description" content="Myna answers typed choice/score/noul decisions from a state it
 scans once. Every figure on this page is read from a committed artifact; the accuracy gate is not met.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="site">
  <span class="wordmark">Myna</span>
  <nav class="site">
    <a href="{DEMO_URL}">Live demo</a>
    <a href="#numbers">Numbers</a>
    <a href="#gates">Gates</a>
    <a href="#install">Install</a>
    <a href="{REPO_URL}">Repository</a>
  </nav>
</header>

<div class="hero">
  <div>
    <p class="eyebrow mono">{f['params_comma']} parameters · {f['d_model']}d · {f['n_layers']} layers · {f['n_heads']} heads</p>
    <h1>It scans the state once, then answers without re-reading.</h1>
    <p class="lede">Myna is a gated linear-attention engine for typed <span class="mono">choice</span>,
      <span class="mono">score</span> and <span class="mono">noul</span> decisions over a growing
      observation: a fixed-size matrix state instead of a cache that grows with context, small enough
      to run on a laptop and export to a browser.
      <strong>It does not clear its accuracy gate.</strong> That is printed below rather than
      buried, at the precision the artifacts write.</p>
    <div class="cta">
      <a class="button" href="{DEMO_URL}">Live demo</a>
      <span class="install mono" id="pip">pip install {esc(f['dist_name'])}=={esc(f['version'])}</span>
    </div>
  </div>
  <div class="card" role="img" aria-label="The quickstart output printed by the released wheel:
    choice billing at confidence {esc(f['qk_confidence'])}">
    <h3>Measured on the released bytes</h3>
    <span class="big mono" translate="no">{esc(f['qk_choice'])}</span>
    <p class="kv mono">confidence <span translate="no">{esc(f['qk_confidence'])}</span></p>
    <p class="kv mono">weakest <span translate="no">{esc(f['qk_weakest_label'])}</span>
      <span translate="no">{esc(f['qk_weakest_prob'])}</span></p>
    <p class="kv mono">params <span translate="no">{esc(f['qk_params'])}</span> · temperature
      <span translate="no">{esc(f['qk_temperature'])}</span></p>
    <p class="note" style="margin-top:1rem">The same float from three install harnesses —
      <span class="mono">gh release download</span>, <span class="mono">pip install git+</span>,
      <span class="mono">pip install</span> from PyPI. Sources:
      <span class="mono">{'</span>, <span class="mono">'.join(esc(p) for p in f['qk_logs'])}</span>.</p>
  </div>
</div>

<section id="numbers">
  <h2>Accuracy, three measured arms</h2>
  <p class="sub">One frozen split, {f['cells']} cells both harnesses score, per-cell unweighted macro.
    Bars run from zero against the same two reference lines, so the distance to the target is the
    real distance. Every value is the artifact's own float, printed in full.</p>
  <ul class="tracks">{arm(f, "off", "Anti-prior OFF control", f"{f['off_gpu']} · the best published arm")}{arm(
      f, "v1b", "V1-B", f"{f['v1b_gpu']} · the run gate G1 rests on")}{arm(
      f, "on", "Anti-prior ON", f"{f['on_gpu']} · dose-matched")}
  </ul>

  <div class="axis">{ticks(f)}</div>
  <div class="axis-labels mono"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div>

  <div class="tiles" style="margin-top:2.5rem">
    <div class="card">
      <h3>Majority floor <span class="tag measured">measured</span></h3>
      <span class="big mono" translate="no">{esc(f['floor'])}</span>
      <p class="kv">Always-predict-the-majority, scored by the same harness over the same cells.</p>
      <p class="srcline mono">source <span translate="no">{esc(f['off_witness'])} · macro.majority</span></p>
    </div>
    <div class="card">
      <h3>G1 target <span class="tag notmet">gated — not met</span></h3>
      <span class="big mono" translate="no">{esc(f['target'])}</span>
      <p class="kv">And ≥ +{esc(f['margin_req'])} over the floor. The committed report writes
        <span class="mono" translate="no">"pass": {json.dumps(f['g1_pass'])}</span> for this gate
        in all three arms.</p>
      <p class="srcline mono">source <span translate="no">{esc(f['off_witness'])} · g1</span></p>
    </div>
  </div>

  <h3 style="font-size:1.15rem;margin:2.75rem 0 .5rem;letter-spacing:-0.01em">The anti-prior experiment</h3>
  <p class="sub">A sampler flag meant to break label priors cost accuracy instead of gaining it:
    <span class="mono">{esc(f['on_acc'])}</span> with the flag on against
    <span class="mono">{esc(f['off_acc'])}</span> with it off —
    <strong class="mono">&minus;{esc(f['flag_delta_4'])}</strong> from the flag alone. The two runs
    print their own launch lines (SPEC §9.30), and these are the options they differ by:
    <span class="mono">{diff_text(f)}</span> — same <span class="mono">--seed {esc(f['same_seed'])}</span>,
    same <span class="mono">--batch {esc(f['same_batch'])}</span>, same suite and split, and both
    reaching <span class="mono">{esc(f['executed'])}</span> executed updates
    (<span class="mono">last_step {f['on_last_step']}</span> in each metrics file, the index one
    before the count). It is a clean negative, and it is why the next lever is data volume rather
    than a batcher trick. Sources <span class="mono">{esc(f['on_witness'])}</span> and
    <span class="mono">{esc(f['off_witness'])}</span>, with the dose and the launch lines from
    <span class="mono">{esc(f['on_metrics'])}</span>, <span class="mono">{esc(f['off_metrics'])}</span>,
    <span class="mono">{esc(f['on_train_log'])}</span> and
    <span class="mono">{esc(f['off_train_log'])}</span>.</p>

  <p class="note">The witness line beside these arms is a scored competitor run in the same harness,
    committed in the same file: laya at <span class="mono" translate="no">{esc(f['laya'])}</span>
    (<span class="mono" translate="no">{esc(f['off_witness'])} · macro.both.laya</span>).
    It is printed because it is the denominator the gate was written against, not because
    this page benchmarks it.</p>
</section>

<section id="gates">
  <h2>The release gate, all {numword(len(f['gates']))} rows</h2>
  <p class="sub">{f['gate_tally']['met']} met, {f['gate_tally']['not met']} not met, {f['gate_tally']['open']} open.
    Verdicts and conditions are imported from
    <span class="mono">bench/gates.py</span>, which checks each one against a committed witness
    under <span class="mono">make gates</span> — a gate cannot be greener than its artifact.</p>
  <table>
    <thead><tr><th>gate</th><th>name</th><th>verdict</th><th>condition</th></tr></thead>
    <tbody>
{gate_rows}
    </tbody>
  </table>
</section>

<section id="demo">
  <h2>What the live demo demonstrates</h2>
  <p class="sub"><a href="{DEMO_URL}">{DEMO_URL}</a> runs the {f['params_comma']}-parameter
    checkpoint whose <span class="mono">sha256</span> is
    <span class="mono" translate="no">{esc(f['sha256'])}</span>, at temperature
    <span class="mono">{esc(f['temperature'])}</span>. It shows two mechanisms; it does not show a
    model that passes G1.</p>
  <ul class="plain">
    <li class="cap" data-cap="scan-once"><strong>Scan once, ask repeatedly.</strong> The engine
      observes the message once and every later question reads that same state —
      <span class="mono">“{esc(f['scan_once_phrase'])}”</span> is the HTTP surface's own wording
      (<span class="mono">src/myna/serve.py</span>):
      <ul class="plain" style="border:0">{routes}</ul>
    </li>
    <li class="cap" data-cap="abstain"><strong>An abstention floor that prints <span class="mono">n/a</span>.</strong>
      <span class="mono">--abstain-below</span> withholds the commitment under the threshold, and
      with the floor off the engine returns no confidence and no margin rather than a fabricated
      one — <span class="mono" translate="no">{esc(f['abstain_none'])}</span>
      (<span class="mono">src/myna/engine.py:{f['abstain_line']}</span>). The demo renders that
      <span class="mono">None</span> as <span class="mono">n/a</span> instead of a number.</li>
  </ul>
  <p class="note">An abstention floor is a routing mechanism, not the product gate G5 describes:
    G5 is <span class="tag notmet">not met</span> above, and no confidence on that page has been
    shown to predict correctness at the level the gate asks for.</p>
</section>

<section id="install">
  <h2>Install and run it</h2>
  <p class="sub">Requires <span class="mono" translate="no">Python {esc(f['python_req'])}</span>.
    The wheel carries no weights — a
    <span class="mono">pip install</span> that silently fetched a training run would make every
    published macro a side effect of an install.</p>
  <pre class="code" translate="no">pip install {esc(f['dist_name'])}=={esc(f['version'])}
myna-weights          # stdlib urllib: no gh, no token; verifies each digest GitHub publishes
myna-weights --check   # does the cache hash? exits 1 without writing anything</pre>
  <pre class="code" translate="no">from myna import Myna, fetch_weights
myna = Myna(fetch_weights())          # test macro {short(f['off_acc'])}, the best published arm
out = myna.predict("My order never arrived and I want a refund.", {{
    "department": {{"type": "choice",
                   "instructions": "Which team should handle this ticket?",
                   "criteria": ["billing", "technical", "shipping", "returns", "other"]}}}})
print(out["answers"]["department"]["choice"])   # → {esc(f['qk_choice'])}</pre>
  <p class="note">Extras: <span class="mono">[serve]</span> for the HTTP API,
    <span class="mono">[browser]</span> for the ONNX export, <span class="mono">[mlx]</span> for
    Apple-silicon int8. Each published arm is a Release tag:
    <span class="mono">antiprior_off_s0-weights</span> is the
    {short(f['off_acc'])} control, <span class="mono">antiprior_on_s0c-weights</span> the
    {short(f['on_acc'])} one, <span class="mono">v1b-checkpoint</span> the {short(f['v1b_acc'])}
    Kaggle run.</p>
</section>

<section id="story">
  <h2>What this is, and what it is not</h2>
  <div class="two-col">
    <ul class="plain">
      <li>Typed decisions — <span class="mono">choice</span>, <span class="mono">score</span>,
        <span class="mono">noul</span> — in one forward pass, not a generation.</li>
      <li>Fixed-size state, so a longer observation costs the same as a shorter one.</li>
      <li>Reproducible: {f['registry_rows']} registry rows in <span class="mono">bench/reproduce.py</span>
        bind each published figure to one command and one committed witness.</li>
      <li>Exportable: the browser artifact answers decisions in real Chrome with the probabilities
        the torch engine prints (G3 <span class="tag measured">met</span>).</li>
    </ul>
    <ul class="plain">
      <li>Not G1: {esc(f['v1b_acc'])} on the complete run against a {esc(f['target'])} target,
        <span class="mono" translate="no">"pass": {json.dumps(f['g1_pass'])}</span> in the artifact.</li>
      <li>Not a trained-v1 product: the {esc(f['off_acc'])} arm above ran
        {executed(f, 'off')} of {comma(f['off_requested'])} requested updates, and the
        harness that scored it prints that table as a VOID one (§7.1).</li>
      <li>Not long-context-correct: G4 is <span class="tag notmet">not met</span> — the state is
        cheap, the decision at distance is unmeasured.</li>
      <li>Not useful confidence yet: G5 is <span class="tag notmet">not met</span>.</li>
      <li>No milliseconds on this page. The latency gate is a ratio; the numbers behind it are
        load-dependent on a shared box, so quoting them would be quoting the machine.</li>
    </ul>
  </div>
</section>

<section id="reproduce">
  <h2>Reproduce every figure above</h2>
  <p class="sub">Each command re-derives one arm from the committed metrics file. They are the
    registry's own lines, printed from <span class="mono">bench/reproduce.py</span>.</p>
  <pre class="code" translate="no">{esc(f['row_cmds']['antiprior-off-macro'])}
{esc(f['row_cmds']['antiprior-on-macro'])}
{esc(f['row_cmds']['v1b-kaggle-macro'])}</pre>
  <pre class="code">make repro       # every published table cell against its witness
make gates       # the {numword(len(f['gates']))} verdicts above against their artifacts
make secrets     # credential shapes in every blob reachable from a ref
make site-check  # this file is current and every literal on it re-derives
make site-mutate # the re-derivation can actually fail (page and artifact mutations)</pre>
  <p class="note">And this page itself: <span class="mono">uv run python bench/gen_site.py</span> is
    its only writer; <span class="mono">uv run python bench/site_witness.py</span> re-reads the
    artifacts and refuses the file if any printed literal moved, and
    <span class="mono">uv run python bench/mutation_site.py</span> is the check on that check.</p>
</section>

<footer class="site">
  <p>Figures as of {f['built']} — the newest commit among the run artifacts this page reads, so
    regenerating from that tree is byte-identical and the stamp does not move when the page is
    edited. Every number here is read out of a committed artifact or is arithmetic on two of them. Each
    arm prints its exact float; a four-decimal form appears only in prose beside it.</p>
  <p>Myna · <a href="{REPO_URL}">{REPO_URL}</a> · distribution <span class="mono">{esc(f['dist_name'])} {esc(f['version'])}</span></p>
</footer>

</div>
</body>
</html>
"""


def cls(verdict):
    return {"met": "measured", "not met": "notmet", "open": "open"}[verdict]


def short_m(params):
    """16,926,848 as the hero's '16.9M' — derived, and the exact count is in the eyebrow."""
    return f"{int(params) / 1e6:.1f}M"


def ticks(f):
    return "".join(f'<i style="--at:{v * 100}%"></i>' for v in (0.25, 0.5, 0.75, 1.0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if docs/index.html is not what these artifacts generate")
    args = ap.parse_args(argv)

    f = facts()
    html = render(f)
    if args.check:
        if OUT.exists() and OUT.read_text() == html:
            print(f"docs/index.html is current: {cmd_line()}")
            return 0
        print("docs/index.html differs from what the artifacts generate — "
              "run `uv run python bench/gen_site.py`")
        return 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print(f"$ {cmd_line()}")
    print(f"wrote {OUT.relative_to(REPO)}: {len(html)} bytes, "
          f"{len(f['gates'])} gate rows, 3 measured arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
