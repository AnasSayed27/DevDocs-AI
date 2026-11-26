import html
import secrets

from fastapi import HTTPException


def verify_api_key(api_key: str, expected_key: str) -> bool:
    """Validate an API key in constant time using secrets.compare_digest."""
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def wrap_prompt(user_input: str) -> str:
    """Escape XML/HTML delimiters in user input and wrap it in <user_input> tags."""
    escaped = html.escape(user_input, quote=False)
    return f"<user_input>{escaped}</user_input>"


wrap_user_input = wrap_prompt
