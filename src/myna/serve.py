"""HTTP serve API for the streaming engine.

    POST /v1/predict                      {state, questions}          -> answers
    POST /v1/sessions                     {state}                     -> {session_id}
    POST /v1/sessions/{id}/append         {text}                      -> {state_tokens}
    POST /v1/sessions/{id}/ask            {questions}                 -> answers
    DELETE /v1/sessions/{id}                                          -> {closed}

A session holds an Observation: the state was scanned once into the fixed-size
matrix cache, so /ask and /append cost the same no matter how long the
observation has grown. /v1/predict matches laya's Router.predict contract.

Run:  uv run --extra serve python -m myna.serve --ckpt runs/myna-v0 --port 8080
"""

# NB: no `from __future__ import annotations` here — FastAPI must resolve the
# request models by real type, and string annotations make it treat them as
# query parameters (HTTP 422).

import argparse
import uuid
from pathlib import Path

from .engine import Myna


def create_app(myna: Myna):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("serve extras missing: uv sync --extra serve") from e

    app = FastAPI(title="myna", version="0.1")
    sessions: dict[str, object] = {}

    class PredictIn(BaseModel):
        state: str
        questions: dict

    class SessionIn(BaseModel):
        state: str

    class AppendIn(BaseModel):
        text: str

    class AskIn(BaseModel):
        questions: dict

    def _get(sid: str):
        obs = sessions.get(sid)
        if obs is None:
            raise HTTPException(404, f"no session {sid}")
        return obs

    @app.get("/v1/health")
    def health():
        return {"ok": True, "params": myna.n_params, "temperature": myna.temperature,
                "abstain_below": myna.abstain_below}

    @app.post("/v1/predict")
    def predict(req: PredictIn):
        return myna.predict(req.state, req.questions)

    @app.post("/v1/sessions")
    def open_session(req: SessionIn):
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = myna.observe(req.state)
        return {"session_id": sid, "state_tokens": len(sessions[sid].ids)}

    @app.post("/v1/sessions/{sid}/append")
    def append(sid: str, req: AppendIn):
        obs = _get(sid)
        sessions[sid] = obs.append(req.text)
        return {"state_tokens": len(sessions[sid].ids)}

    @app.post("/v1/sessions/{sid}/ask")
    def ask(sid: str, req: AskIn):
        return _get(sid).ask(req.questions)

    @app.delete("/v1/sessions/{sid}")
    def close(sid: str):
        if sessions.pop(sid, None) is None:
            raise HTTPException(404, f"no session {sid}")
        return {"closed": sid}

    return app


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        prog="myna-serve",
        description="Serve the typed decision engine over HTTP: POST /v1/predict with "
                    "{state, questions}, or /v1/sessions to scan a state once and ask "
                    "it repeatedly.")
    ap.add_argument("--ckpt", default="runs/myna-v0", metavar="DIR",
                    help="directory holding model.pt and tokenizer.json (default "
                         "runs/myna-v0; fetch one with `myna-weights --dest DIR`)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--abstain-below", type=float, default=None, metavar="P",
                    help="refuse any decision whose top probability is under P and say why "
                         "(default: never abstain). /v1/health reports the value in force.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)

    # The default path only exists inside this repo, and `Myna` would answer with a
    # torch FileNotFoundError for a directory that simply has no weights in it. Check
    # that first, because a wrong `--ckpt` is the common mistake and a missing extra
    # is the rare one — and because this check is the half a reader can hit anywhere.
    ckpt = Path(args.ckpt)
    missing = [name for name in ("model.pt", "tokenizer.json") if not (ckpt / name).is_file()]
    if missing:
        raise SystemExit(
            f"{ckpt} has no {', '.join(missing)}. Fetch a published checkpoint with "
            f"`myna-weights --dest {args.ckpt}`, or name a directory that already has one "
            "with --ckpt.")

    # imported after parse_args so `--help` works on a box without the `serve`
    # extra: an optional dependency must not be what breaks the usage text
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "myna-serve needs the serve extra. Installed from git: "
            'pip install "myna[serve] @ git+https://github.com/aashish254/myna.git". '
            "In this checkout: uv sync --extra serve.")

    myna = Myna(args.ckpt, device=args.device, abstain_below=args.abstain_below)
    uvicorn.run(create_app(myna), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
