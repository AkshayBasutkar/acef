"""Tests for retrieval-needs routing boundaries."""

import json
from pathlib import Path

from core.schemas import QueryType
from core.query_analyzer import classify


class TestSimpleBoundary:
    """Programming subject matter is SIMPLE unless local retrieval is needed."""

    def test_generic_programming_questions_are_simple(self):
        queries = [
            "How do we define a function in Python?",
            "Explain dependency injection.",
            "How do I invert a binary tree?",
            "What is the time complexity of quicksort?",
            "How do I write unit tests?",
            "What is a closure in JavaScript?",
            "How do I fix a TypeError in my script?",
        ]
        for query in queries:
            assert classify(query) == QueryType.SIMPLE

    def test_self_contained_code_is_simple(self):
        queries = [
            "Fix this: `for i in range(10) print(i)`",
            "def login(username, password):\n    pass",
            "import requests\nrequests.get('https://example.com')",
            "Traceback (most recent call last): File 'app.py', line 2",
        ]
        for query in queries:
            assert classify(query) == QueryType.SIMPLE

    def test_nontechnical_lookalikes_are_simple(self):
        queries = [
            "What is the function of the pancreas?",
            "There was an error in my tax return.",
            "What class should I take to learn pottery?",
            "What is a document database?",
            "How can I remember a list of numbers?",
        ]
        for query in queries:
            assert classify(query) == QueryType.SIMPLE


class TestCodeBoundary:
    """CODE is reserved for facts obtainable from the indexed repository."""

    def test_explicit_local_codebase_scope_routes_to_code(self):
        queries = [
            "Explain authentication in this codebase.",
            "How does our repository handle caching?",
            "Review the current project's source code architecture.",
            "Where is validation implemented in the local repo?",
        ]
        for query in queries:
            assert classify(query) == QueryType.CODE

    def test_repository_navigation_routes_to_code(self):
        queries = [
            "Where is UserProfile defined?",
            "Which functions call hash_password?",
            "Find the implementation of the login handler.",
            "Trace the call graph for authentication.",
        ]
        for query in queries:
            assert classify(query) == QueryType.CODE

    def test_project_file_request_routes_to_code(self):
        queries = [
            "Explain what utils.py does.",
            "Inspect auth.ts for the validation flow.",
            "Refactor handlers.go to remove the dependency.",
        ]
        for query in queries:
            assert classify(query) == QueryType.CODE

    def test_known_corpus_symbol_routes_to_code(self):
        symbols = frozenset({"login", "hash_password"})
        assert classify("What does login call?", known_symbols=symbols) == QueryType.CODE
        assert classify("Why does hash_password fail?", known_symbols=symbols) == QueryType.CODE

    def test_generic_form_remains_simple_even_if_word_matches_symbol(self):
        symbols = frozenset({"login"})
        query = "How do I define a login function in Python?"
        assert classify(query, known_symbols=symbols) == QueryType.SIMPLE


class TestMemoryBoundary:
    """MEMORY requires stored document or conversation evidence."""

    def test_explicit_prior_turn_dependency_routes_to_memory(self):
        queries = [
            "What did we discuss earlier about the budget?",
            "Remind me what you said about deadlines.",
            "What was my previous question?",
            "What option did we agree on?",
            "Do you remember what I chose?",
        ]
        for query in queries:
            assert classify(query) == QueryType.MEMORY

    def test_explicit_uploaded_document_dependency_routes_to_memory(self):
        queries = [
            "Summarize the PDF I uploaded.",
            "According to the document, what is the refund policy?",
            "In my notes, which plan includes support?",
        ]
        for query in queries:
            assert classify(query) == QueryType.MEMORY

    def test_document_probe_routes_implicit_document_question(self):
        assert classify(
            "What are the pricing tiers?",
            doc_similarity_fn=lambda: 0.85,
        ) == QueryType.MEMORY

    def test_low_document_similarity_does_not_route_to_memory(self):
        assert classify(
            "What are the pricing tiers?",
            doc_similarity_fn=lambda: 0.1,
        ) == QueryType.SIMPLE

    def test_short_followup_uses_session_context(self):
        assert classify("and the second one?", session_turn_count=3) == QueryType.MEMORY
        assert classify(
            "and its callers?",
            session_turn_count=3,
            last_turn_type=QueryType.CODE,
        ) == QueryType.CODE

    def test_code_subject_recall_is_memory(self):
        assert classify("What did you say earlier about auth.py?") == QueryType.MEMORY

    def test_current_repository_fact_beats_incidental_recall_wording(self):
        query = "You mentioned login earlier; where is it defined in this repository?"
        assert classify(query) == QueryType.CODE


class TestDevEvaluationSet:
    def test_dev_set_accuracy_and_hard_negatives(self):
        dev_file = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "classifier_dev.jsonl"
        records = [json.loads(line) for line in dev_file.read_text(encoding="utf-8").splitlines() if line]
        assert len(records) >= 40

        corpus_symbols = frozenset({"login", "hash_password", "user_profile", "auth"})
        correct = 0
        hard_negative_total = 0
        hard_negative_correct = 0

        for record in records:
            expected = QueryType(record["expected"].lower())
            needs_corpus = record.get("needs_corpus", False)
            known_symbols = corpus_symbols if needs_corpus and expected == QueryType.CODE else frozenset()
            doc_probe = (
                (lambda: 0.85)
                if needs_corpus and expected == QueryType.MEMORY
                else None
            )
            predicted = classify(
                record["query"],
                session_turn_count=record.get("min_session_turns", 0),
                known_symbols=known_symbols,
                doc_similarity_fn=doc_probe,
                last_turn_type=(
                    QueryType(record["last_turn_type"])
                    if record.get("last_turn_type")
                    else None
                ),
            )
            matched = predicted == expected
            correct += int(matched)
            if record.get("hard_negative", False):
                hard_negative_total += 1
                hard_negative_correct += int(matched)

        assert correct / len(records) >= 0.90
        assert hard_negative_correct == hard_negative_total
