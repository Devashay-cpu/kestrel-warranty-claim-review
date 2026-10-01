"""Kestrel fraud-review API.   Start:  uvicorn app:app --reload     UI: http://127.0.0.1:8000/     Docs: /docs

Thin FastAPI wrapper around kestrel/service.py. The saved Phase 3 model is loaded ONCE at startup (never retrained).
No API key, no internet access and no paid service is needed.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from kestrel.service import ClaimError, Predictor, RISK_NOTE

log = logging.getLogger("kestrel.api")
UI_FILE = Path(__file__).resolve().parent / "static" / "index.html"
MAX_BODY_BYTES = 64 * 1024
EXAMPLE = {"claim_id": "DEMO-1", "submitted_at": "2026-10-02T14:30:00", "partner_id": "SP9001", "sku": "KH-AF-01",
           "claim_amount_inr": 1500, "days_since_purchase": 200, "photo_attached": "N", "partner_inspected": "N",
           "customer_prior_claims": 2, "inspector_note": None}


def create_app(bundle_path=None) -> FastAPI:
    """bundle_path: saved model bundle. Default: $KESTREL_BUNDLE or artifacts/phase3/final_model.joblib."""
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor, app.state.load_error = None, None
        try:
            path = bundle_path or os.environ.get("KESTREL_BUNDLE") or None
            app.state.predictor = Predictor.from_path(path)
        except Exception as e:                      # app still starts; /health says what is wrong
            app.state.load_error = f"{type(e).__name__}: {e}"
            log.error("model bundle could not be loaded: %s", app.state.load_error)
        yield

    app = FastAPI(title="Kestrel warranty fraud review", version="1.0", lifespan=lifespan,
                  description="Review-priority score for ONE warranty claim. " + RISK_NOTE)

    @app.get("/health")
    def health(request: Request):
        ok = request.app.state.predictor is not None
        body = {"status": "ok" if ok else "degraded", "model_loaded": ok}
        if not ok:
            body["problem"] = "Model bundle not loaded. Build it with `python scripts/run_phase3.py` (needs data/raw) or set KESTREL_BUNDLE."
        return body

    @app.get("/meta")
    def meta(request: Request):
        p = request.app.state.predictor
        if p is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded", "example": EXAMPLE})
        return {**p.meta(), "example": EXAMPLE}

    @app.post("/predict", openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"example": EXAMPLE}}}})
    async def predict(request: Request):
        p = request.app.state.predictor
        if p is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded. See /health."})
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body too large for a single claim."})
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(status_code=400, content={"detail": "Request body is not valid JSON.", "errors": []})
        try:
            return await run_in_threadpool(p.predict, payload)
        except ClaimError as e:
            return JSONResponse(status_code=422, content={"detail": "The claim could not be scored; please fix the listed fields.", "errors": e.errors})
        except Exception:                           # never leak a traceback to the caller
            log.exception("scoring failed")
            return JSONResponse(status_code=500, content={"detail": "Internal error while scoring this claim."})

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(UI_FILE, media_type="text/html")

    return app


app = create_app()
