#!/usr/bin/env python3
"""
Gmail Auto-Classifier
Reads unread INBOX emails, classifies them via OpenAI GPT-4o-mini,
removes the INBOX label and applies the corresponding custom Gmail label.

Designed to run every minute via cron on a low-RAM VPS (~360 MB).
"""

import os
import sys
import logging
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & logging setup  (done before any heavy imports)
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports – only pulled in when actually needed
# ---------------------------------------------------------------------------
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI, APITimeoutError, APIError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.error("OPENAI_API_KEY not set. Add it to your .env file.")
    sys.exit(1)

BODY_MAX_CHARS = 1_500
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TIMEOUT = 20  # seconds – keep it tight for a cron job

# ---------------------------------------------------------------------------
# Category → Gmail Label ID mapping
# Replace the placeholder values with your real Label IDs from Gmail.
# You can find them with: GET https://gmail.googleapis.com/gmail/v1/users/me/labels
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    # ↓↓↓ Reemplaza cada valor con el Label ID real obtenido de Gmail ↓↓↓
    "Job_Offers":    "Label_8119458192147360879",
    "Contributions": "Label_5886649300345820358",
    "Invoices":      "Label_5126766089745091903",
    "Therapy":       "Label_6518647735862115722",
    "Mentorship":    "Label_8542356387351197241",
    "BIT_Corporate": "Label_1496558045047244139",
    "Uncategorized": "Label_2231438737774944686",
}

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an email classifier. Respond with ONLY the exact category name, nothing else.

Categories and rules:
- Job_Offers: ONLY if sender/domain is LinkedIn or Computrabajo AND the email contains a job offer, vacancy, or interview. Exclude networking, ads, newsletters, or motivational content.
- Contributions: ONLY if it's about payroll payments, settlements, EPS contributions, pension, ARL, or payroll certificates. Exclude general EPS info or ads.
- Invoices: ONLY if it's transactional and real (invoices, receipts, vouchers, debits/credits, charges). Typical entities: Nequi, Lulo, Davivienda, PSE, Movistar, WOM, AWS. Exclude bank ads or credit card offers.
- Therapy: ONLY if it's about personal psychological therapy sessions, appointments, or payments. Exclude general wellness, coaching, or third-party therapy.
- Mentorship: ONLY for confirmations of technical mentoring sessions with students or workshop invitations from Julian Sandoval.
- BIT_Corporate: ONLY for BIT institutional emails, contracts, job offers from BIT, or internal BIT information. Exclude mentoring or student agendas.
- Uncategorized: ALWAYS use this if none of the above rules are met exactly.

Reply with exactly one of: Job_Offers, Contributions, Invoices, Therapy, Mentorship, BIT_Corporate, Uncategorized"""


# ---------------------------------------------------------------------------
# Gmail authentication
# ---------------------------------------------------------------------------
def get_gmail_service():
    """Return an authenticated Gmail API service object."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                log.error("credentials.json not found at %s", CREDENTIALS_FILE)
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------
def get_inbox_messages(service) -> list[dict]:
    """Return list of unread INBOX message stubs (id + threadId only)."""
    try:
        result = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX"], q="is:unread", maxResults=50)
            .execute()
        )
        return result.get("messages", [])
    except HttpError as exc:
        log.error("Failed to list messages: %s", exc)
        return []


def get_message_details(service, msg_id: str) -> dict | None:
    """Fetch full message and return a dict with subject and body snippet."""
    try:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )
    except HttpError as exc:
        log.error("Failed to fetch message %s: %s", msg_id, exc)
        return None

    headers = msg.get("payload", {}).get("headers", [])
    subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(no subject)")

    # Extract plain-text body walking the payload parts
    body = _extract_body(msg.get("payload", {}))

    return {"subject": subject, "body": body[:BODY_MAX_CHARS]}


def _extract_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    import base64

    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace") if data else ""

    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _extract_body(part)
            if text:
                return text

    return ""


def apply_label(service, msg_id: str, label_id: str) -> bool:
    """Remove INBOX label and add the custom classification label."""
    try:
        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={
                "addLabelIds": [label_id],
                "removeLabelIds": ["INBOX"],
            },
        ).execute()
        return True
    except HttpError as exc:
        log.error("Failed to modify labels for message %s: %s", msg_id, exc)
        return False


# ---------------------------------------------------------------------------
# OpenAI classification
# ---------------------------------------------------------------------------
def classify_email(client: OpenAI, subject: str, body: str) -> str:
    """Call GPT-4o-mini and return exactly one category name."""
    user_content = f"Subject: {subject}\n\nBody:\n{body}"
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=10,       # category name is at most 15 chars
            temperature=0,       # fully deterministic
            timeout=OPENAI_TIMEOUT,
        )
        category = response.choices[0].message.content.strip()
        # Safety fallback: if model hallucinates an unknown category
        if category not in LABEL_MAP:
            log.warning("Unknown category '%s' returned – falling back to Uncategorized", category)
            category = "Uncategorized"
        return category

    except APITimeoutError:
        log.warning("OpenAI timeout for message with subject '%s' – skipping", subject)
        return "Uncategorized"
    except APIError as exc:
        log.error("OpenAI API error: %s – skipping message", exc)
        return "Uncategorized"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=== Gmail classifier started ===")

    service = get_gmail_service()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    messages = get_inbox_messages(service)
    if not messages:
        log.info("No unread INBOX messages found. Exiting.")
        return

    log.info("Found %d message(s) to process.", len(messages))

    processed = errors = 0

    for stub in messages:
        msg_id = stub["id"]
        details = get_message_details(service, msg_id)
        if not details:
            errors += 1
            continue

        subject = details["subject"]
        log.info("Processing: %s", subject)

        category = classify_email(openai_client, subject, details["body"])
        label_id = LABEL_MAP[category]

        success = apply_label(service, msg_id, label_id)
        if success:
            log.info("  → Classified as '%s' (label: %s)", category, label_id)
            processed += 1
        else:
            errors += 1

    log.info("=== Done. Processed: %d | Errors: %d ===", processed, errors)


if __name__ == "__main__":
    main()
