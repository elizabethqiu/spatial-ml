"""
FastAPI server — exposes inference and scene graph endpoints to the frontend
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.correction_loop import apply_and_log
from model.scene_graph import SceneGraph

# inference engine
_engine = None
_graphs: dict[str, SceneGraph] = {}  # session_id -> SceneGraph


def get_engine():
    global _engine
    if _engine is None:
        from inference.run import SpatialInferenceEngine
        _engine = SpatialInferenceEngine(
            vision_model_path=os.getenv("VISION_MODEL_PATH", "remyxai/SpaceLLaVA"),
            load_reasoning=os.getenv("LOAD_REASONING", "1") != "0",
        )
    return _engine


app = FastAPI(title="spatial-ml API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models

class CorrectionRequest(BaseModel):
    session_id: str
    node_id: str
    label: str | None = None
    position_3d: list[float] | None = None
    depth_m: float | None = None
    delete: bool = False


class InferenceResponse(BaseModel):
    session_id: str
    summary: str
    cot_trace: str
    estimates: list[dict]
    safety_alerts: list[str]
    scene_graph: dict

# Routes

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    question: str = Form(default=""),
    session_id: str = Form(default=""),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    session_id = session_id or str(uuid.uuid4())
    graph = _graphs.setdefault(session_id, SceneGraph())

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    engine = get_engine()
    engine.scene_graph = graph
    result = engine.run_frame(image, question or None)

    _graphs[session_id] = engine.scene_graph

    return InferenceResponse(
        session_id=session_id,
        summary=result.get("summary", ""),
        cot_trace=result.get("cot_trace", ""),
        estimates=result.get("estimates", []),
        safety_alerts=result.get("safety_alerts", []),
        scene_graph=result.get("scene_graph", {}),
    )


@app.get("/graph/{session_id}")
def get_graph(session_id: str):
    graph = _graphs.get(session_id)
    if not graph:
        raise HTTPException(404, "Session not found.")
    return graph.to_dict()


@app.post("/correct")
def correct(req: CorrectionRequest):
    graph = _graphs.get(req.session_id)
    if not graph:
        raise HTTPException(404, "Session not found.")

    correction: dict[str, Any] = {}
    if req.label is not None:
        correction["label"] = req.label
    if req.position_3d is not None:
        correction["position_3d"] = req.position_3d
    if req.depth_m is not None:
        correction["depth_m"] = req.depth_m
    if req.delete:
        correction["delete"] = True

    result = apply_and_log(graph, req.node_id, correction, req.session_id)
    if not result["success"]:
        raise HTTPException(404, result["message"])
    return result


@app.delete("/graph/{session_id}")
def clear_graph(session_id: str):
    _graphs.pop(session_id, None)
    return {"cleared": session_id}
