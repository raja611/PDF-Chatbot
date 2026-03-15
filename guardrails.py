"""
Input/output guardrails for the PDF chatbot.

- Input validation: length, empty checks, injection patterns
- File validation: type, size limits
- Output safety: PII detection, hallucination flag
- Rate limiting: per-session throttle
"""

import re
import time
import logging

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 2
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_QUERIES_PER_MINUTE = 20

_rate_limit_store = {}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(a|an)\s+",
    r"system\s*:\s*",
    r"<\s*script",
    r"\{\{.*\}\}",
    r"ADMIN_OVERRIDE",
]

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


class GuardrailResult:
    def __init__(self, passed: bool, message: str = "", code: str = ""):
        self.passed = passed
        self.message = message
        self.code = code

    def __bool__(self):
        return self.passed


def ok():
    return GuardrailResult(True)


def fail(message: str, code: str):
    return GuardrailResult(False, message, code)


# -- Input guardrails ----------------------------------------------------------

def validate_query(text: str | None) -> GuardrailResult:
    if not text or not text.strip():
        return fail("Message cannot be empty.", "EMPTY_INPUT")

    text = text.strip()

    if len(text) < MIN_QUERY_LENGTH:
        return fail(
            f"Message too short (min {MIN_QUERY_LENGTH} characters).",
            "INPUT_TOO_SHORT",
        )

    if len(text) > MAX_QUERY_LENGTH:
        return fail(
            f"Message too long (max {MAX_QUERY_LENGTH} characters).",
            "INPUT_TOO_LONG",
        )

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Prompt injection attempt detected: {text[:100]}")
            return fail(
                "Your message was flagged as potentially unsafe.",
                "INJECTION_DETECTED",
            )

    return ok()


def validate_file(file) -> GuardrailResult:
    if file is None or file.filename == "":
        return fail("No file selected.", "NO_FILE")

    filename = file.filename.lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return fail(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            "INVALID_FILE_TYPE",
        )

    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if size_mb > MAX_FILE_SIZE_MB:
        return fail(
            f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.",
            "FILE_TOO_LARGE",
        )

    if size_mb == 0:
        return fail("File is empty.", "EMPTY_FILE")

    return ok()


# -- Rate limiting -------------------------------------------------------------

def check_rate_limit(session_id: str) -> GuardrailResult:
    now = time.time()
    window = 60

    if session_id not in _rate_limit_store:
        _rate_limit_store[session_id] = []

    timestamps = _rate_limit_store[session_id]
    timestamps[:] = [t for t in timestamps if now - t < window]

    if len(timestamps) >= MAX_QUERIES_PER_MINUTE:
        return fail(
            f"Rate limit exceeded. Max {MAX_QUERIES_PER_MINUTE} queries per minute.",
            "RATE_LIMITED",
        )

    timestamps.append(now)
    return ok()


# -- Output guardrails ---------------------------------------------------------

def scan_output_for_pii(text: str) -> dict:
    """Returns dict of PII types found. Empty dict = clean."""
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = len(matches)
    return found


def redact_pii(text: str) -> str:
    """Replace detected PII with [REDACTED]."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text


def check_output(text: str) -> tuple[str, dict]:
    """
    Scan and optionally redact output.
    Returns (cleaned_text, metadata).
    """
    pii = scan_output_for_pii(text)
    meta = {"pii_detected": bool(pii)}

    if pii:
        logger.warning(f"PII detected in output: {pii}")
        text = redact_pii(text)
        meta["pii_types"] = list(pii.keys())
        meta["pii_redacted"] = True

    return text, meta
