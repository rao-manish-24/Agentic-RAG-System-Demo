import httpx
from openai import APIConnectionError, AuthenticationError, RateLimitError

from agentic_rag.documents import DocumentError
from agentic_rag.presentation import safe_error


def test_errors_never_echo_api_details_or_secret_values() -> None:
    request = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(401, request=request)
    error = AuthenticationError("private-test-placeholder", response=response, body=None)
    assert "authentication" in safe_error(error)
    assert "private-test-placeholder" not in safe_error(error)
    assert "private-test-placeholder" not in safe_error(RuntimeError("private-test-placeholder"))
    rate = RateLimitError("details", response=response, body=None)
    assert "quota" in safe_error(rate)
    assert "network" in safe_error(APIConnectionError(request=request))
    assert safe_error(DocumentError("No extractable text")) == "No extractable text"
