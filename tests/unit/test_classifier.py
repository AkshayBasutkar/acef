"""
Unit tests for core/classifier.py — strategy-pattern query classifier.

Covers:
  - HeuristicClassifier: delegates correctly to query_analyzer.classify()
  - LLMClassifier: parses all three labels, handles noisy output, falls back
    gracefully on timeout / API error / unparseable response
  - get_classifier factory: returns correct backend for each mode, handles
    unknown mode, handles missing llm_client for "llm" mode
  - ClassificationContext: dataclass defaults are safe
  - Pipeline integration: accepts classifier= kwarg, logs classifier_mode
"""

from unittest.mock import MagicMock, patch
import pytest

from core.classifier import (
    ClassificationContext,
    HeuristicClassifier,
    LLMClassifier,
    LocalClassifier,
    get_classifier,
)
from core.schemas import QueryType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_llm_client(model: str = "gemini-2.0-flash", provider: str = "gemini") -> MagicMock:
    client = MagicMock()
    client.model = model
    client.provider = provider
    client.client = MagicMock()  # the underlying OpenAI SDK instance
    return client


# ── ClassificationContext ─────────────────────────────────────────────────────

class TestClassificationContext:
    def test_defaults_are_safe(self):
        ctx = ClassificationContext()
        assert ctx.session_turn_count == 0
        assert ctx.known_symbols == frozenset()
        assert ctx.doc_similarity_fn is None
        assert ctx.last_turn_type is None

    def test_fields_set_correctly(self):
        ctx = ClassificationContext(
            session_turn_count=5,
            known_symbols=frozenset({"login", "logout"}),
            last_turn_type=QueryType.CODE,
        )
        assert ctx.session_turn_count == 5
        assert "login" in ctx.known_symbols
        assert ctx.last_turn_type == QueryType.CODE


# ── HeuristicClassifier ───────────────────────────────────────────────────────

class TestHeuristicClassifier:
    def test_delegates_to_query_analyzer(self):
        """HeuristicClassifier must call the original classify() function."""
        classifier = HeuristicClassifier()
        ctx = ClassificationContext()
        # Queries with unambiguous strong signals
        assert classifier.classify("What does main() do?", ctx) == QueryType.CODE
        assert classifier.classify("What is the capital of France?", ctx) == QueryType.SIMPLE
        # "you said" is a strong MEMORY phrase in the heuristic list
        assert classifier.classify("You said the deadline was Friday.", ctx) == QueryType.MEMORY

    def test_passes_context_fields(self):
        """Context fields (known_symbols, doc_similarity_fn, etc.) are forwarded."""
        classifier = HeuristicClassifier()
        ctx = ClassificationContext(
            known_symbols=frozenset({"auth", "login"}),
            session_turn_count=4,
            last_turn_type=QueryType.CODE,
        )
        # Short follow-up in long session with CODE last turn → CODE
        result = classifier.classify("why?", ctx)
        assert result == QueryType.CODE


# ── LLMClassifier ─────────────────────────────────────────────────────────────

class TestLLMClassifier:
    def _make_classifier(self, provider: str = "groq") -> tuple[LLMClassifier, MagicMock]:
        llm_client = _make_llm_client(provider=provider)
        classifier = LLMClassifier(llm_client, provider=provider)
        return classifier, llm_client.client

    # ── Label parsing ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("raw,expected", [
        ("CODE", QueryType.CODE),
        ("MEMORY", QueryType.MEMORY),
        ("SIMPLE", QueryType.SIMPLE),
        # Case-insensitive
        ("code", QueryType.CODE),
        ("Memory", QueryType.MEMORY),
        ("simple", QueryType.SIMPLE),
    ])
    def test_parses_clean_response(self, raw, expected):
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response(raw)
        result = classifier.classify("Any query", ClassificationContext())
        assert result == expected

    def test_parses_noisy_response_with_label_embedded(self):
        """The label may be embedded in noise — regex must extract it."""
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response(
            "The answer is CODE."
        )
        assert classifier.classify("test", ClassificationContext()) == QueryType.CODE

    def test_parses_label_with_surrounding_whitespace(self):
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response("  MEMORY  ")
        assert classifier.classify("test", ClassificationContext()) == QueryType.MEMORY

    # ── Fallback behaviour ─────────────────────────────────────────────────

    def test_falls_back_to_heuristic_on_unparseable_response(self):
        """If the LLM returns an unrecognised label, heuristic takes over."""
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response("banana")
        ctx = ClassificationContext()
        # "What is Paris?" should be SIMPLE via heuristic
        result = classifier.classify("What is Paris?", ctx)
        assert result == QueryType.SIMPLE

    def test_falls_back_to_heuristic_on_timeout(self):
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.side_effect = TimeoutError("5s timeout")
        ctx = ClassificationContext()
        # Falls back to heuristic; "main()" is CODE
        result = classifier.classify("What does main() do?", ctx)
        assert result == QueryType.CODE

    def test_falls_back_to_heuristic_on_api_error(self):
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.side_effect = Exception("Rate limited")
        ctx = ClassificationContext()
        result = classifier.classify("What is 2 + 2?", ctx)
        assert result == QueryType.SIMPLE  # heuristic → SIMPLE

    def test_falls_back_to_heuristic_on_empty_response(self):
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response("")
        ctx = ClassificationContext()
        result = classifier.classify("What is Paris?", ctx)
        assert result == QueryType.SIMPLE

    # ── Model selection ────────────────────────────────────────────────────

    def test_uses_default_lightweight_model_for_groq(self):
        llm_client = _make_llm_client(model="llama-3.3-70b", provider="groq")
        classifier = LLMClassifier(llm_client, model_override="", provider="groq")
        assert classifier._model == "llama-3.1-8b-instant"

    def test_uses_default_lightweight_model_for_gemini(self):
        llm_client = _make_llm_client(model="gemini-2.0-flash", provider="gemini")
        classifier = LLMClassifier(llm_client, model_override="", provider="gemini")
        assert classifier._model == "gemini-2.0-flash-lite"

    def test_model_override_is_respected(self):
        llm_client = _make_llm_client(provider="groq")
        classifier = LLMClassifier(llm_client, model_override="gemma2-9b-it", provider="groq")
        assert classifier._model == "gemma2-9b-it"

    def test_uses_correct_api_parameters(self):
        """Classification call must use temperature=0.0 and a small max_tokens."""
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response("CODE")
        classifier.classify("What does auth() do?", ClassificationContext())

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] <= 10
        assert call_kwargs["timeout"] <= 10.0   # must be short

    def test_system_prompt_is_sent(self):
        """The classifier system prompt must be included in every call."""
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response("SIMPLE")
        classifier.classify("Hello", ClassificationContext())

        messages = mock_openai.chat.completions.create.call_args[1]["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "CODE" in system_msgs[0]["content"]
        assert "MEMORY" in system_msgs[0]["content"]
        assert "SIMPLE" in system_msgs[0]["content"]

    def test_query_is_user_message(self):
        """The raw query must appear as the user turn."""
        classifier, mock_openai = self._make_classifier()
        mock_openai.chat.completions.create.return_value = _make_openai_response("CODE")
        query = "Explain dependency injection"
        classifier.classify(query, ClassificationContext())

        messages = mock_openai.chat.completions.create.call_args[1]["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs[0]["content"] == query


# ── LocalClassifier ───────────────────────────────────────────────────────────

class TestLocalClassifier:
    def test_raises_value_error_with_no_path(self):
        with pytest.raises(ValueError, match="CLASSIFIER_LOCAL_MODEL_PATH"):
            LocalClassifier("")

    def test_raises_not_implemented_with_path(self):
        with pytest.raises(NotImplementedError):
            LocalClassifier("/some/model/path")


# ── get_classifier factory ────────────────────────────────────────────────────

class TestGetClassifier:
    def _make_settings(self, mode: str, model: str = "", path: str = "",
                       provider: str = "groq") -> MagicMock:
        s = MagicMock()
        s.classifier_mode = mode
        s.classifier_model = model
        s.classifier_local_model_path = path
        s.llm_provider = provider
        return s

    def test_returns_heuristic_by_default(self):
        settings = self._make_settings("heuristic")
        result = get_classifier(settings)
        assert isinstance(result, HeuristicClassifier)

    def test_returns_heuristic_for_empty_mode(self):
        settings = self._make_settings("")
        result = get_classifier(settings)
        assert isinstance(result, HeuristicClassifier)

    def test_returns_heuristic_for_unknown_mode(self):
        settings = self._make_settings("neural_net_v7")
        result = get_classifier(settings)
        assert isinstance(result, HeuristicClassifier)

    def test_returns_llm_classifier_when_configured(self):
        settings = self._make_settings("llm", provider="groq")
        llm_client = _make_llm_client(provider="groq")
        result = get_classifier(settings, llm_client)
        assert isinstance(result, LLMClassifier)

    def test_falls_back_to_heuristic_when_llm_mode_but_no_client(self):
        settings = self._make_settings("llm")
        result = get_classifier(settings, llm_client=None)
        assert isinstance(result, HeuristicClassifier)

    def test_llm_classifier_uses_model_override(self):
        settings = self._make_settings("llm", model="custom-model", provider="groq")
        llm_client = _make_llm_client(provider="groq")
        classifier = get_classifier(settings, llm_client)
        assert isinstance(classifier, LLMClassifier)
        assert classifier._model == "custom-model"

    def test_local_mode_raises_not_implemented_with_path(self):
        settings = self._make_settings("local", path="/models/classifier")
        with pytest.raises(NotImplementedError):
            get_classifier(settings)

    def test_local_mode_raises_value_error_without_path(self):
        settings = self._make_settings("local", path="")
        with pytest.raises(ValueError):
            get_classifier(settings)


# ── Pipeline integration ──────────────────────────────────────────────────────

class TestPipelineWithClassifier:
    """Verify Pipeline wires the classifier correctly."""

    def _make_mock_pipeline(self, classifier):
        from core.pipeline import Pipeline

        mock_cache = MagicMock()
        mock_graph_rag = MagicMock()
        mock_mem_rag = MagicMock()
        mock_llm = MagicMock()

        mock_cache.lookup.return_value = None
        mock_mem_rag.session_turn_count.return_value = 0
        mock_mem_rag.last_turn_type.return_value = None
        mock_llm.generate.return_value = "Mock answer"

        pipe = Pipeline(
            cache=mock_cache,
            graph_rag=mock_graph_rag,
            mem_rag=mock_mem_rag,
            llm_client=mock_llm,
            classifier=classifier,
        )
        return pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm

    def test_injected_classifier_is_used(self):
        """An injected classifier mock must receive the query."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = QueryType.SIMPLE

        pipe, _, graph_rag, mem_rag, llm = self._make_mock_pipeline(mock_classifier)
        from core.schemas import QueryRequest
        req = QueryRequest(query="What is 2 + 2?")
        pipe.handle_query(req)

        mock_classifier.classify.assert_called_once()
        call_query = mock_classifier.classify.call_args[0][0]
        assert call_query == "What is 2 + 2?"

    def test_code_classification_routes_to_graph_rag(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = QueryType.CODE

        pipe, _, graph_rag, mem_rag, _ = self._make_mock_pipeline(mock_classifier)
        from core.schemas import QueryRequest
        pipe.handle_query(QueryRequest(query="What does login do?"))

        graph_rag.retrieve.assert_called_once()
        mem_rag.retrieve.assert_not_called()

    def test_memory_classification_routes_to_mem_rag(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = QueryType.MEMORY

        pipe, _, graph_rag, mem_rag, _ = self._make_mock_pipeline(mock_classifier)
        from core.schemas import QueryRequest
        pipe.handle_query(QueryRequest(query="What did you say earlier?"))

        mem_rag.retrieve.assert_called_once()
        graph_rag.retrieve.assert_not_called()

    def test_simple_classification_skips_retrieval(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = QueryType.SIMPLE

        pipe, _, graph_rag, mem_rag, _ = self._make_mock_pipeline(mock_classifier)
        from core.schemas import QueryRequest
        pipe.handle_query(QueryRequest(query="What is the capital of France?"))

        graph_rag.retrieve.assert_not_called()
        mem_rag.retrieve.assert_not_called()

    def test_default_classifier_is_heuristic_without_explicit_injection(self):
        """When no classifier is passed, Pipeline must create a HeuristicClassifier
        (default mode) without crashing."""
        from core.pipeline import Pipeline

        mock_cache = MagicMock()
        mock_graph_rag = MagicMock()
        mock_mem_rag = MagicMock()
        mock_llm = MagicMock()
        mock_cache.lookup.return_value = None
        mock_mem_rag.session_turn_count.return_value = 0
        mock_mem_rag.last_turn_type.return_value = None
        mock_llm.generate.return_value = "answer"

        pipe = Pipeline(
            cache=mock_cache,
            graph_rag=mock_graph_rag,
            mem_rag=mock_mem_rag,
            llm_client=mock_llm,
        )
        assert isinstance(pipe.classifier, HeuristicClassifier)
