from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from agentic_rag.documents import DocumentError


def safe_error(error: Exception) -> str:
    if isinstance(error, DocumentError):
        return str(error)
    if isinstance(error, AuthenticationError):
        return "OpenAI authentication failed. Check the API key in your environment."
    if isinstance(error, RateLimitError):
        return "OpenAI rate/quota limit reached. Check billing or wait and retry."
    if isinstance(error, APIConnectionError):
        return "Could not connect to OpenAI. Check your network and retry."
    if isinstance(error, APIStatusError):
        return "OpenAI could not complete the request. Retry or check model availability."
    return "The request could not be completed. Check the file/configuration and try again."
