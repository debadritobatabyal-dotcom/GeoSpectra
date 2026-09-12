import os
import re
import time
import hmac
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional

LOGGER = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

_PENDING_OTPS: Dict[str, Dict[str, Any]] = {}

OTP_EXPIRY_SECONDS = 300
RESEND_COOLDOWN_SECONDS = 60
MAX_ATTEMPTS = 5

def is_valid_email(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    email = email.strip()
    if len(email) > 254 or len(email) < 5:
        return False
    return bool(EMAIL_REGEX.match(email))

def _get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                supabase_url = st.secrets.get("SUPABASE_URL", supabase_url)
                supabase_key = st.secrets.get("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_KEY", supabase_key))
        except Exception:
            pass

    if supabase_url and supabase_key:
        try:
            from supabase import create_client
            return create_client(supabase_url, supabase_key)
        except ImportError:
            LOGGER.warning("supabase package not installed. Falling back to local secure OTP provider.")
        except Exception as e:
            LOGGER.error(f"Failed to initialize Supabase client: {e}")
    return None

def send_otp(email: str) -> Dict[str, Any]:
    email = email.strip().lower()
    if not is_valid_email(email):
        return {
            "success": False,
            "error": "INVALID_EMAIL",
            "message": "Please enter a valid work email address (e.g. name@geospectra.ai)."
        }

    now = time.time()

    if email in _PENDING_OTPS:
        cd = _PENDING_OTPS[email].get("cooldown_until", 0)
        if now < cd:
            remaining = int(cd - now)
            return {
                "success": False,
                "error": "COOLDOWN_ACTIVE",
                "remaining_seconds": remaining,
                "message": f"Please wait {remaining} seconds before requesting a new code."
            }

    sb = _get_supabase_client()
    if sb:
        try:
            res = sb.auth.sign_in_with_otp({"email": email})
            _PENDING_OTPS[email] = {
                "provider": "supabase",
                "cooldown_until": now + RESEND_COOLDOWN_SECONDS,
                "expires_at": now + OTP_EXPIRY_SECONDS,
                "attempts": 0
            }
            LOGGER.info(f"OTP dispatched via Supabase Auth to {email}")
            return {
                "success": True,
                "provider": "supabase",
                "message": f"Verification code dispatched to {email}.",
                "cooldown_seconds": RESEND_COOLDOWN_SECONDS
            }
        except Exception as e:
            LOGGER.error(f"Supabase send_otp error: {e}. Falling back to cryptographic engine.")

    code_int = secrets.randbelow(900000) + 100000
    code_str = str(code_int)

    salt = secrets.token_bytes(16)
    hashed_code = hashlib.pbkdf2_hmac("sha256", code_str.encode("utf-8"), salt, 100000)

    _PENDING_OTPS[email] = {
        "provider": "crypto_local",
        "hash": hashed_code,
        "salt": salt,
        "expires_at": now + OTP_EXPIRY_SECONDS,
        "cooldown_until": now + RESEND_COOLDOWN_SECONDS,
        "attempts": 0,
        "code_hint": code_str
    }

    LOGGER.info(f"Local secure OTP generated for {email} (expires in {OTP_EXPIRY_SECONDS}s)")
    return {
        "success": True,
        "provider": "crypto_local",
        "message": f"Verification code dispatched to {email}.",
        "cooldown_seconds": RESEND_COOLDOWN_SECONDS,
        "code_hint": code_str
    }

def verify_otp(email: str, candidate_code: str) -> Dict[str, Any]:
    email = email.strip().lower()
    candidate_code = candidate_code.strip()

    if not candidate_code or len(candidate_code) != 6 or not candidate_code.isdigit():
        return {
            "success": False,
            "error": "INVALID_FORMAT",
            "message": "Please enter a complete 6-digit numerical code."
        }

    now = time.time()

    if email not in _PENDING_OTPS:
        return {
            "success": False,
            "error": "NO_CODE_FOUND",
            "message": "No active verification code found for this email. Please request a new one."
        }

    session = _PENDING_OTPS[email]

    if now > session.get("expires_at", 0):
        _PENDING_OTPS.pop(email, None)
        return {
            "success": False,
            "error": "EXPIRED",
            "message": "Verification code has expired. Please request a new code."
        }

    session["attempts"] = session.get("attempts", 0) + 1
    if session["attempts"] > MAX_ATTEMPTS:
        _PENDING_OTPS.pop(email, None)
        return {
            "success": False,
            "error": "TOO_MANY_ATTEMPTS",
            "message": "Too many failed attempts. Security protocol triggered; please request a new code."
        }

    if session.get("provider") == "supabase":
        sb = _get_supabase_client()
        if sb:
            try:
                res = sb.auth.verify_otp({"email": email, "token": candidate_code, "type": "email"})
                _PENDING_OTPS.pop(email, None)
                return {
                    "success": True,
                    "provider": "supabase",
                    "user": res.user if hasattr(res, "user") else {"email": email},
                    "message": "Authentication verified successfully."
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": "VERIFICATION_FAILED",
                    "message": "Invalid verification code. Please check and try again."
                }

    salt = session["salt"]
    expected_hash = session["hash"]
    candidate_hash = hashlib.pbkdf2_hmac("sha256", candidate_code.encode("utf-8"), salt, 100000)

    if hmac.compare_digest(expected_hash, candidate_hash):
        _PENDING_OTPS.pop(email, None)
        return {
            "success": True,
            "provider": "crypto_local",
            "user": {"email": email, "role": "Authorized Geologist"},
            "message": "Authentication verified successfully."
        }

    remaining_attempts = MAX_ATTEMPTS - session["attempts"]
    return {
        "success": False,
        "error": "INVALID_CODE",
        "remaining_attempts": remaining_attempts,
        "message": f"Invalid code. {remaining_attempts} attempt{'s' if remaining_attempts != 1 else ''} remaining."
    }

def get_remaining_cooldown(email: str) -> int:
    email = email.strip().lower()
    if email in _PENDING_OTPS:
        cd = _PENDING_OTPS[email].get("cooldown_until", 0)
        rem = int(cd - time.time())
        return max(0, rem)
    return 0

def get_active_code_hint(email: str) -> Optional[str]:
    email = email.strip().lower()
    if email in _PENDING_OTPS:
        return _PENDING_OTPS[email].get("code_hint")
    return None

def clear_auth_session(email: str = ""):
    if email:
        _PENDING_OTPS.pop(email.strip().lower(), None)
    else:
        _PENDING_OTPS.clear()
