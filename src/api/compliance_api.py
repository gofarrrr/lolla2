"""
Compliance Validation API (placeholder)
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import re

router = APIRouter(prefix="/api/compliance", tags=["Compliance"]) 

class ComplianceInput(BaseModel):
    payload: Dict[str, Any]

@router.post("/validate")
async def validate(payload: ComplianceInput) -> Dict[str, Any]:
    data = payload.payload or {}
    text = str(data.get("text", ""))
    # Basic PII detection
    email_rx = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_rx = re.compile(r"\b\+?\d[\d\s().-]{7,}\b")
    emails = email_rx.findall(text)
    phones = phone_rx.findall(text)

    pii_found = bool(emails or phones)
    checks = [
        {"name": "pii_email", "status": "fail" if emails else "pass", "count": len(emails)},
        {"name": "pii_phone", "status": "fail" if phones else "pass", "count": len(phones)},
        {"name": "prompt_safety", "status": "pass"},
    ]
    return {"ok": not pii_found, "checks": checks}
