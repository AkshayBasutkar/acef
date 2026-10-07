"""
Unit tests for core/query_analyzer.py — M4 acceptance check.

Tests all four example classifications from §7.2, plus additional edge cases.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.schemas import QueryType
from core.query_analyzer import classify


class TestQueryAnalyzer:
    """Test the heuristic classifier against the acceptance criteria."""

    # ── §7.2 acceptance examples ─────────────────────────────────────────
    def test_code_query_bug_in_function(self):
        """'Fix the bug in my login function' → CODE"""
        assert classify("Fix the bug in my login function") == QueryType.CODE

    def test_memory_query_discussed_earlier(self):
        """'What did we discuss about the pricing document earlier?' → MEMORY"""
        result = classify("What did we discuss about the pricing document earlier?")
        assert result == QueryType.MEMORY

    def test_simple_query_capital(self):
        """'What is the capital of France?' → SIMPLE"""
        assert classify("What is the capital of France?") == QueryType.SIMPLE

    def test_memory_short_followup_in_long_session(self):
        """A 2-word query in a session with 4 prior turns → MEMORY"""
        assert classify("and that?", session_turn_count=4) == QueryType.MEMORY

    # ── Additional CODE classifications ──────────────────────────────────
    def test_code_traceback(self):
        assert classify("I got a traceback in my script") == QueryType.CODE

    def test_code_class_keyword(self):
        assert classify("Explain this class definition") == QueryType.CODE

    def test_code_py_extension(self):
        assert classify("What does main.py do?") == QueryType.CODE

    def test_code_refactor(self):
        assert classify("How do I refactor this method?") == QueryType.CODE

    def test_code_import_statement(self):
        assert classify("Why does this import fail?") == QueryType.CODE

    # ── Additional MEMORY classifications ────────────────────────────────
    def test_memory_previously(self):
        assert classify("What did you previously say about X?") == QueryType.MEMORY

    def test_memory_in_the_document(self):
        assert classify("What's in the document about pricing?") == QueryType.MEMORY

    def test_memory_remind_me(self):
        assert classify("Remind me what the summary was") == QueryType.MEMORY

    def test_memory_the_pdf(self):
        assert classify("Summarize the pdf we uploaded") == QueryType.MEMORY

    # ── SIMPLE classifications ───────────────────────────────────────────
    def test_simple_general_knowledge(self):
        assert classify("How far is the moon?") == QueryType.SIMPLE

    def test_simple_math(self):
        assert classify("What is 2 + 2?") == QueryType.SIMPLE

    # ── Short follow-up NOT in long session ──────────────────────────────
    def test_short_followup_in_short_session_is_simple(self):
        """Short query but session is not 'long' yet → stays SIMPLE."""
        assert classify("and that?", session_turn_count=1) == QueryType.SIMPLE

    def test_long_query_in_long_session_is_simple(self):
        """Long query (>8 words) even in long session → SIMPLE (no keyword match)."""
        long_q = "tell me more about the weather forecast for tomorrow morning please"
        assert classify(long_q, session_turn_count=5) == QueryType.SIMPLE

    # ── Priority: CODE > MEMORY when both match ─────────────────────────
    def test_code_wins_over_memory(self):
        """If both CODE and MEMORY keywords present, CODE wins (first match)."""
        q = "Earlier you showed me a function that had a bug"
        # "function" and "bug" → CODE match first
        assert classify(q) == QueryType.CODE
