"""
LLM Client — unified Gemini / Groq client.

Both providers expose OpenAI-compatible chat completions endpoints, so a
single OpenAI SDK client works for both.  Provider selection is a config
value; switching requires only changing LLM_PROVIDER (and the corresponding
API key) in .env — no code changes.
"""

import logging

from openai import OpenAI

from config.settings import get_settings

logger = logging.getLogger(__name__)

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

        if settings.llm_provider == "gemini":
            api_key = settings.gemini_api_key or "placeholder_gemini_key"
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings.gemini_base_url,
            )
            self.model = settings.gemini_model
        elif settings.llm_provider == "groq":
            api_key = settings.groq_api_key or "placeholder_groq_key"
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings.groq_base_url,
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
            # Simple character-based truncation to stay within budget
            context_text = context_text[: self._max_context_chars]
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
        else:
            # When retrieval is empty, omit context block entirely and prompt with fallback note
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )

        result = response.choices[0].message.content
        logger.debug("LLM response — length=%d chars", len(result or ""))
        return result or ""
