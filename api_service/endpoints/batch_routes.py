"""
CivilityAI: Large-Scale Batch Content Analysis Endpoint.

Supports CSV upload for bulk moderation inference, computing throughput,
batch latency, and returning enriched moderation datasets.
"""

from __future__ import annotations

import csv
import io
import time
from typing import Any, Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from safety_ml.safety_inference import ContentSafetyEngine

batch_router = APIRouter(prefix="/batch", tags=["Batch Processing"])


@batch_router.post("/analyze")
async def analyze_batch_csv(
    file: UploadFile = File(...),
):
    """
    Accepts a CSV file containing user messages, conducts batched safety inference,
    and returns processing statistics along with the analyzed output.
    Expected CSV columns: 'message_body' or 'comment_text' or 'text'.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .csv file.",
        )

    content = await file.read()
    text_content = content.decode("utf-8", errors="replace")
    csv_reader = csv.DictReader(io.StringIO(text_content))

    # Detect text column
    fieldnames = csv_reader.fieldnames or []
    candidate_text_cols = ["message_body", "comment_text", "text", "message", "content"]
    text_col = next((col for col in candidate_text_cols if col in fieldnames), None)

    if text_col is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV must contain a message text column (e.g., 'message_body' or 'comment_text'). Found: {fieldnames}",
        )

    incoming_messages: List[Dict[str, str]] = list(csv_reader)
    if not incoming_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV contains no rows.",
        )

    engine = ContentSafetyEngine.get_singleton_instance()
    raw_texts = [row[text_col] for row in incoming_messages]

    # Measure batch inference throughput
    start_time = time.perf_counter()
    batch_results = engine.analyze_batch(raw_texts)
    processing_duration = time.perf_counter() - start_time

    total_messages = len(incoming_messages)
    throughput_rate = round(total_messages / max(processing_duration, 1e-6), 1)
    average_processing_latency = round((processing_duration / max(total_messages, 1)) * 1000, 2)

    # Enrich rows
    enriched_rows: List[Dict[str, Any]] = []
    for orig_row, assessment in zip(incoming_messages, batch_results):
        enriched = dict(orig_row)
        enriched["safety_risk_index"] = assessment["safety_risk_index"]
        enriched["moderation_action"] = assessment["moderation_action"]
        for cat, score in assessment["category_scores"].items():
            enriched[f"score_{cat}"] = score
        enriched_rows.append(enriched)

    return {
        "status": "success",
        "total_messages": total_messages,
        "processing_duration_seconds": round(processing_duration, 3),
        "throughput_rate_msgs_per_sec": throughput_rate,
        "average_latency_ms": average_processing_latency,
        "batch_sample_preview": enriched_rows[:10],
        "total_results": len(enriched_rows),
    }


@batch_router.post("/export-csv")
async def export_batch_analysis_csv(
    file: UploadFile = File(...),
):
    """
    Processes an uploaded CSV and streams the enriched CSV back to the client.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Must be a .csv file.")

    content = await file.read()
    text_content = content.decode("utf-8", errors="replace")
    csv_reader = csv.DictReader(io.StringIO(text_content))

    candidate_text_cols = ["message_body", "comment_text", "text", "message", "content"]
    text_col = next((col for col in (csv_reader.fieldnames or []) if col in candidate_text_cols), None)

    if not text_col:
        raise HTTPException(status_code=400, detail="Missing message column in CSV.")

    rows = list(csv_reader)
    engine = ContentSafetyEngine.get_singleton_instance()
    results = engine.analyze_batch([r[text_col] for r in rows])

    # Build CSV output stream
    output = io.StringIO()
    base_fields = list(csv_reader.fieldnames or [])
    extra_fields = ["safety_risk_index", "moderation_action"] + [f"score_{c}" for c in engine.category_columns]
    writer = csv.DictWriter(output, fieldnames=base_fields + extra_fields)
    writer.writeheader()

    for r, res in zip(rows, results):
        enriched = dict(r)
        enriched["safety_risk_index"] = res["safety_risk_index"]
        enriched["moderation_action"] = res["moderation_action"]
        for cat, score in res["category_scores"].items():
            enriched[f"score_{cat}"] = score
        writer.writerow(enriched)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=civility_moderation_results.csv"},
    )
