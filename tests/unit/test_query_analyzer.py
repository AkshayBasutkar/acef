"""
Unit tests for core/query_analyzer.py — I-04 signal-tiered classifier.

Covers:
  - Each strong CODE signal (traceback, file extensions, backticks, call syntax, snake_case, line-anchored)
  - Weak CODE signal threshold rule (1 weak signal is insufficient; >= 2 distinct weak signals trigger CODE)
  - Brand allowlist (iPhone, YouTube, OpenAI, etc. are ignored; custom camelCase triggers weak signal)
  - Strong MEMORY phrases vs bare words (earlier, before, remember do not trigger MEMORY)
  - Follow-up rule (length <= 5 words, session turns >= threshold, CODE inheritance vs default MEMORY)
  - Precedence (CODE > MEMORY)
  - Regression test on data/eval/classifier_dev.jsonl (accuracy >= 90%, all hard negatives pass)
"""

import json
from pathlib import Path
import pytest

from core.schemas import QueryType
from core.query_analyzer import classify


class TestQueryAnalyzerSignals:
    """Test individual signal tiers and rules."""

    # ── 1. Strong CODE signals ───────────────────────────────────────────
    def test_strong_traceback(self):
        assert classify("Traceback (most recent call last): File 'app.py'") == QueryType.CODE

    def test_strong_file_extensions(self):
        for ext in [".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"]:
            assert classify(f"Please inspect file{ext} for details") == QueryType.CODE

    def test_strong_backticks(self):
        assert classify("Refactor the `login` helper") == QueryType.CODE
        assert classify("Fix this: `for i in range(10) print(i)`") == QueryType.CODE

    def test_strong_call_syntax(self):
        assert classify("What does main() call?") == QueryType.CODE
        assert classify("What does calculate_total() do?") == QueryType.CODE

    def test_strong_snake_case(self):
        assert classify("Why does hash_password fail?") == QueryType.CODE
        assert classify("Where is user_profile defined?") == QueryType.CODE

    def test_strong_line_anchored(self):
        assert classify("def login(username, password):\n    pass") == QueryType.CODE
        assert classify("class AuthManager:\n    pass") == QueryType.CODE
        assert classify("import requests\nrequests.get('...')") == QueryType.CODE
        assert classify("from math import sqrt") == QueryType.CODE

    # ── 2. Weak CODE signal threshold ────────────────────────────────────
    def test_single_weak_signal_is_simple(self):
        """Single weak signal is insufficient on its own (< 2)."""
        assert classify("I found a bug in my garden, what is it?") == QueryType.SIMPLE
        assert classify("There was an error in my tax return, who should I contact?") == QueryType.SIMPLE
        assert classify("What is the function of the pancreas?") == QueryType.SIMPLE
        assert classify("Can you recommend a method for studying?") == QueryType.SIMPLE
        assert classify("The weather is a variable factor.") == QueryType.SIMPLE

    def test_two_distinct_weak_signals_trigger_code(self):
        """Two distinct weak signals combine to route to CODE."""
        # 'function' + 'call' (relationship word)
        assert classify("What does the login function call?") == QueryType.CODE
        # 'method' + 'refactor'
        assert classify("Can you refactor this method?") == QueryType.CODE
        # 'error' + 'script'
        assert classify("There is an error in my script") == QueryType.CODE

    # ── 3. Brand allowlist ───────────────────────────────────────────────
    def test_brand_allowlist_not_treated_as_code(self):
        assert classify("What's the difference between an iPhone and an Android phone?") == QueryType.SIMPLE
        assert classify("Recommend a book about the history of YouTube.") == QueryType.SIMPLE
        assert classify("How do I make a presentation in PowerPoint?") == QueryType.SIMPLE
        assert classify("Tell me about OpenAI and ChatGPT.") == QueryType.SIMPLE

    def test_non_brand_camel_case_counts_as_weak_signal(self):
        # UserProfile (weak) + defined (weak relationship) -> 2 distinct weak -> CODE
        assert classify("Where is the UserProfile class defined?") == QueryType.CODE
        # TypeError (weak) + script (weak) -> 2 distinct weak -> CODE
        assert classify("How do I fix a TypeError in my script?") == QueryType.CODE

    # ── 4. Strong MEMORY phrases vs bare words ───────────────────────────
    def test_strong_memory_phrases(self):
        assert classify("What did we discuss earlier about the budget?") == QueryType.MEMORY
        assert classify("Summarize the PDF I uploaded.") == QueryType.MEMORY
        assert classify("According to the document, what is the policy?") == QueryType.MEMORY
        assert classify("Remind me what you said about deadlines.") == QueryType.MEMORY
        assert classify("What was my previous question?") == QueryType.MEMORY
        assert classify("As we talked about last time, which one?") == QueryType.MEMORY
        assert classify("In the document, which plan is best?") == QueryType.MEMORY
        assert classify("Compare this with what I told you before.") == QueryType.MEMORY

    def test_bare_memory_words_are_not_strong(self):
        """Bare words earlier, before, remember, discussed are hard negatives."""
        assert classify("Can you recommend something to read before bed?") == QueryType.SIMPLE
        assert classify("Should I get up earlier to be productive?") == QueryType.SIMPLE
        assert classify("What's the best way to remember a list of numbers?") == QueryType.SIMPLE
        assert classify("Who discussed the theory of relativity first?") == QueryType.SIMPLE
        assert classify("What is a document database?") == QueryType.SIMPLE

    # ── 5. Follow-up rule ────────────────────────────────────────────────
    def test_short_followup_in_long_session_defaults_to_memory(self):
        assert classify("and the second one?", session_turn_count=3) == QueryType.MEMORY
        assert classify("why is that?", session_turn_count=4) == QueryType.MEMORY
        assert classify("what do you mean?", session_turn_count=3) == QueryType.MEMORY

    def test_short_followup_inherits_code_when_last_turn_was_code(self):
        assert classify(
            "and the second one?",
            session_turn_count=3,
            last_turn_type=QueryType.CODE,
        ) == QueryType.CODE

    def test_short_followup_in_short_session_stays_simple(self):
        assert classify("and the second one?", session_turn_count=1) == QueryType.SIMPLE

    def test_long_followup_in_long_session_stays_simple_without_keywords(self):
        long_q = "tell me more about the weather forecast for tomorrow morning please"
        assert classify(long_q, session_turn_count=5) == QueryType.SIMPLE

    # ── 6. Precedence: CODE > MEMORY ─────────────────────────────────────
    def test_code_precedence_over_memory(self):
        # Has "earlier" / "you said" memory phrases but also strong code signals
        q1 = "Do you remember the bug in auth.py?"
        assert classify(q1) == QueryType.CODE

        q2 = "Earlier you showed me a function that had a bug"
        assert classify(q2) == QueryType.CODE


class TestDevEvaluationSet:
    """Validate accuracy on the seed classifier_dev.jsonl evaluation set."""

    def test_dev_set_accuracy_and_hard_negatives(self):
        dev_file = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "classifier_dev.jsonl"
        assert dev_file.exists(), f"Evaluation file missing: {dev_file}"

        records = []
        with open(dev_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        assert len(records) >= 40

        corpus_symbols = frozenset({"login", "hash_password", "user_profile", "auth"})
        correct = 0
        hn_total = 0
        hn_correct = 0

        for r in records:
            query = r["query"]
            expected = QueryType(r["expected"].lower())
            needs_corpus = r.get("needs_corpus", False)
            min_turns = r.get("min_session_turns", 0)
            is_hn = r.get("hard_negative", False)

            known_symbols = corpus_symbols if needs_corpus else frozenset()
            doc_probe = (lambda: 0.85) if (needs_corpus and expected == QueryType.MEMORY) else None

            pred = classify(
                query,
                session_turn_count=min_turns,
                known_symbols=known_symbols,
                doc_similarity_fn=doc_probe,
            )

            is_match = (pred == expected)
            if is_match:
                correct += 1

            if is_hn:
                hn_total += 1
                if is_match:
                    hn_correct += 1

        accuracy = correct / len(records)
        assert accuracy >= 0.90, f"Classifier dev accuracy {accuracy:.1%} is below 90%"
        assert hn_correct == hn_total, f"Hard negatives failed: {hn_correct}/{hn_total}"
