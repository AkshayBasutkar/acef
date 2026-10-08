"""
LLM Client — unified Gemini / Groq client.

Both providers expose OpenAI-compatible chat completions endpoints, so a
single OpenAI SDK client works for both.  Provider selection is a config
value; switching requires only changing LLM_PROVIDER (and the corresponding
API key) in .env — no code changes.
"""

import logging

from openai import (
    OpenAI,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    APIStatusError,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when an external LLM call fails, times out, or returns invalid content."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ── System prompts per query type ────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {
    "simple": (
        "You are a helpful, direct assistant. Answer the user's question "
        "clearly and concisely."
    ),
    "code": (
        "You are a helpful coding assistant. Use the provided code context "
        "(functions, files, and their relationships) to answer precisely. "
        "Reference specific file and function names where relevant. If the "
        "context doesn't fully cover the question, say so explicitly."
    ),
    "memory": (
        "You are a helpful assistant with access to relevant past "
        "conversation and document excerpts below. Use them to answer "
        "accurately, and mention the source document or prior turn when "
        "relevant. If the context doesn't answer the question, say so."
    ),
}


class LLMClient:
    """
    One class, two providers.  Initialised once at app startup and shared
    across all pipeline invocations.
    """

    def __init__(self, settings=None):
        settings = settings or get_settings()
        self.settings = settings
        self.provider = settings.llm_provider

        if settings.llm_provider == "gemini":
            api_key = settings.gemini_api_key or "placeholder_gemini_key"
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings.gemini_base_url,
                timeout=30.0,
                max_retries=2,
            )
            self.model = settings.gemini_model
        elif settings.llm_provider == "groq":
            api_key = settings.groq_api_key or "placeholder_groq_key"
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings.groq_base_url,
                timeout=30.0,
                max_retries=2,
            )
            self.model = settings.groq_model
        else:
            raise ValueError(
                f"Unknown llm_provider: {settings.llm_provider!r}. "
                f"Expected 'gemini' or 'groq'."
            )

        self._max_context_chars = settings.max_context_chars
        logger.info(
            "LLMClient initialised — provider=%s  model=%s",
            settings.llm_provider,
            self.model,
        )

    def is_configured(self) -> bool:
        """Return True if the active provider has a valid (non-placeholder) API key."""
        if self.provider == "gemini":
            key = self.settings.gemini_api_key
            return bool(key and key.strip() and key != "placeholder_gemini_key")
        elif self.provider == "groq":
            key = self.settings.groq_api_key
            return bool(key and key.strip() and key != "placeholder_groq_key")
        return False

    def generate(
        self,
        query: str,
        query_type: str,
        context_chunks: list[str] | None = None,
    ) -> str:
        """
        Build a prompt from the query, its type-specific system prompt, and
        any retrieved context chunks, then call the LLM and return the text
        response.

        Parameters
        ----------
        query : str
            The user's original question.
        query_type : str
            One of "simple", "code", "memory" — selects the system prompt.
        context_chunks : list[str] | None
            Pre-formatted context strings from GraphRAG or MemRAG.  May be
            empty or None for SIMPLE queries.

        Returns
        -------
        str
            The LLM's response text.
        """
        system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["simple"])

        # Build user message — with or without retrieved context (I-11)
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            context_text = context_text[: self._max_context_chars]
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
        else:
            if query_type in ("code", "memory"):
                user_message = (
                    f"Note: No relevant {query_type} context was found in the indexed index.\n\n"
                    f"Question: {query}"
                )
            else:
                user_message = query

        logger.debug(
            "LLM request — model=%s  type=%s  context_len=%d",
            self.model,
            query_type,
            len(context_chunks or []),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
        except AuthenticationError as e:
            logger.error("LLM authentication failed for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                f"LLM authentication failed for provider '{self.provider}'. Check API key.",
                status_code=503,
            ) from e
        except RateLimitError as e:
            logger.error("LLM rate limit / quota exceeded for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                f"LLM rate limit or quota exceeded for provider '{self.provider}'.",
                status_code=429,
            ) from e
        except APITimeoutError as e:
            logger.error("LLM request timed out for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                "LLM request timed out after 30 seconds.",
                status_code=503,
            ) from e
        except APIConnectionError as e:
            logger.error("LLM connection error for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                f"Could not connect to LLM provider '{self.provider}'.",
                status_code=503,
            ) from e
        except APIStatusError as e:
            err_code = getattr(e, "status_code", 500)
            logger.error("LLM API status error (%s) for provider '%s': %s", err_code, self.provider, e)
            status_code = 429 if err_code == 429 else 503
            raise LLMServiceError(
                f"LLM provider '{self.provider}' returned error status {err_code}.",
                status_code=status_code,
            ) from e
        except APIError as e:
            logger.error("LLM API error for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                f"LLM provider '{self.provider}' error: {getattr(e, 'message', str(e))}",
                status_code=503,
            ) from e
        except TimeoutError as e:
            logger.error("LLM timeout for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                "LLM request timed out.",
                status_code=503,
            ) from e
        except Exception as e:
            if isinstance(e, LLMServiceError):
                raise
            logger.error("Unexpected LLM error for provider '%s': %s", self.provider, e)
            raise LLMServiceError(
                f"LLM call failed: {str(e)}",
                status_code=503,
            ) from e

        if not response or not getattr(response, "choices", None):
            raise LLMServiceError("LLM response contained no choices.", status_code=503)

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason in ("content_filter", "safety"):
            raise LLMServiceError("LLM response was blocked by safety/content filter.", status_code=503)

        result = choice.message.content
        if result is None or not result.strip():
            raise LLMServiceError("LLM returned empty content.", status_code=503)

        logger.debug("LLM response — length=%d chars", len(result))
        return result
