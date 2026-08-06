import unittest
from unittest.mock import patch

import numpy as np

from rag.src.conversation import ConversationState
from rag.src.intent import QueryIntent, classify_intent
from rag.src.preprocess import QueryPreprocessor
from rag.src.search import (
    create_query_preprocessor,
    create_result_selection_answer,
    get_relevant_notices,
    hydrate_result_notice,
    is_recent_sort_request,
    search_notices,
    should_answer_without_selection,
    sort_notices_by_published_at,
)
from rag.src.router import QueryRoute


class FakeModel:
    def encode(self, text, normalize_embeddings=True):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class QueryPreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = QueryPreprocessor({
            "배알골": "배OO 교수가 담당하는 알고리즘 과목",
            "알골": "알고리즘 과목",
        })

    def test_normalizes_attached_notice_question(self):
        processed = self.preprocessor.process("인턴공지알려줘")

        self.assertEqual(processed.normalized, "인턴 공지 알려줘")
        self.assertIn("인턴", processed.keywords)

    def test_expands_course_alias(self):
        normalized = self.preprocessor.normalize("배알골 기말 어디서봄?")

        self.assertIn("배OO 교수가 담당하는 알고리즘 과목", normalized)
        self.assertIn("어디서 봄", normalized)

    def test_reports_resolved_alias_to_caller(self):
        processed = self.preprocessor.process("배알골 기말 어디서봄?")

        self.assertEqual(len(processed.resolved_aliases), 1)
        self.assertEqual(processed.resolved_aliases[0].alias, "배알골")
        self.assertIn("알고리즘", processed.resolved_aliases[0].meaning)

    def test_prefers_longer_alias_before_overlapping_alias(self):
        processed = self.preprocessor.process("배알골 시험")

        self.assertEqual(
            [match.alias for match in processed.resolved_aliases],
            ["배알골"],
        )

    def test_does_not_expand_aliases_inside_resolved_meaning(self):
        preprocessor = QueryPreprocessor({
            "혜자구": "자료구조및프로그래밍 자료구조 이혜영",
            "혜영": "이혜영",
        })

        processed = preprocessor.process("혜자구 시험언제냐")

        self.assertEqual(
            processed.normalized,
            "자료구조및프로그래밍 자료구조 이혜영 시험언제냐",
        )
        self.assertEqual(
            [match.alias for match in processed.resolved_aliases],
            ["혜자구"],
        )

    def test_does_not_expand_alias_inside_longer_korean_word(self):
        preprocessor = QueryPreprocessor({"디어": "박재영"})

        processed = preprocessor.process("멀티미디어실 어디야?")

        self.assertEqual(processed.normalized, "멀티미디어실 어디야?")
        self.assertEqual(processed.resolved_aliases, ())

    def test_expands_alias_at_start_of_attached_question(self):
        preprocessor = QueryPreprocessor({
            "혜자구": "자료구조및프로그래밍 자료구조 이혜영",
        })

        processed = preprocessor.process("혜자구시험언제냐")

        self.assertEqual(
            processed.normalized,
            "자료구조및프로그래밍 자료구조 이혜영시험언제냐",
        )
        self.assertEqual(
            [match.alias for match in processed.resolved_aliases],
            ["혜자구"],
        )

    def test_builds_preprocessor_from_alias_rows(self):
        preprocessor = create_query_preprocessor([
            {"id": 1, "alias": "과사", "meaning": "컴퓨터공학과 학과사무실"},
        ])

        self.assertEqual(
            preprocessor.normalize("과사 어디야?"),
            "컴퓨터공학과 학과사무실 어디야?",
        )

    def test_fallback_removes_colloquial_when_question_stopword(self):
        preprocessor = QueryPreprocessor({
            "곤골": "알고리즘분석 김상곤",
        })

        with patch("rag.src.preprocess._get_kiwi", return_value=None):
            processed = preprocessor.process("곤골 기말 언제냐")

        self.assertEqual(
            processed.normalized,
            "알고리즘분석 김상곤 기말 언제냐",
        )
        self.assertEqual(
            processed.keywords,
            ("알고리즘분석", "김상곤", "기말"),
        )


class IntentTests(unittest.TestCase):
    def test_classifies_supported_routes(self):
        cases = {
            "다른 건 없어?": QueryIntent.MORE_RESULTS,
            "신청기간 널널한 거 없어?": QueryIntent.DEADLINE_RELAXED,
            "곧 마감인 공지 있어?": QueryIntent.DEADLINE_URGENT,
            "지금 신청 가능한 장학금 있어?": QueryIntent.DEADLINE_URGENT,
            "배알골 기말 어디서 봄?": QueryIntent.EXAM_LOCATION,
            "자료구조및프로그래밍 시험언제냐": QueryIntent.EXAM_LOCATION,
            "이 공지 요약해줘": QueryIntent.NOTICE_SUMMARY,
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(classify_intent(question), expected)

    def test_short_question_uses_context_as_follow_up(self):
        self.assertEqual(
            classify_intent("서류는?", has_context=True),
            QueryIntent.FOLLOW_UP,
        )
        self.assertEqual(
            classify_intent("서류는?", has_context=False),
            QueryIntent.GENERAL_SEARCH,
        )
        self.assertEqual(
            classify_intent("상금은?", has_context=True),
            QueryIntent.FOLLOW_UP,
        )


class ConversationStateTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = QueryPreprocessor()
        self.state = ConversationState()

    def test_more_results_reuses_query_and_excludes_shown_notices(self):
        initial_query = self.preprocessor.process("인턴공지 알려줘")
        initial_resolution = self.state.resolve(
            initial_query,
            QueryIntent.GENERAL_SEARCH,
        )
        self.state.record_results(
            initial_resolution,
            [{"notice": {"id": 1}}, {"notice": {"id": 2}}],
        )

        more_query = self.preprocessor.process("다른 건 없어?")
        resolution = self.state.resolve(
            more_query,
            QueryIntent.MORE_RESULTS,
        )

        self.assertEqual(resolution.search_question, "인턴 공지 알려줘")
        self.assertEqual(resolution.exclude_notice_ids, (1, 2))

    def test_follow_up_builds_standalone_search_question(self):
        self.state.last_search_query = "장학금 신청 언제까지야?"
        query = self.preprocessor.process("서류는?")

        resolution = self.state.resolve(query, QueryIntent.FOLLOW_UP)

        self.assertIn("장학금 신청", resolution.search_question)
        self.assertIn("서류는?", resolution.search_question)

    def test_more_results_without_context_asks_for_topic(self):
        query = self.preprocessor.process("다른 건 없어?")

        resolution = self.state.resolve(query, QueryIntent.MORE_RESULTS)

        self.assertIsNotNone(resolution.clarification)

    def test_selects_numbered_candidate_and_keeps_it_active(self):
        query = self.preprocessor.process("대회 공지 알려줘")
        resolution = self.state.resolve(query, QueryIntent.GENERAL_SEARCH)
        results = [
            {"notice": {"id": 10, "title": "첫 번째 대회"}},
            {"notice": {"id": 20, "title": "두 번째 대회"}},
            {"notice": {"id": 30, "title": "세 번째 대회"}},
        ]
        self.state.record_results(resolution, results)

        selected, error = self.state.select_candidate("2번 공지가 궁금해")

        self.assertIsNone(error)
        self.assertEqual(selected["notice"]["id"], 20)
        self.assertEqual(self.state.active_result["notice"]["id"], 20)
        self.assertEqual(self.state.referenced_notice_ids, [20])

    def test_keeps_original_answer_question_while_user_selects_notice(self):
        query = self.preprocessor.process("알고리즘 시험 언제야?")
        resolution = self.state.resolve(query, QueryIntent.EXAM_LOCATION)

        self.state.record_results(
            resolution,
            [{"notice": {"id": 10, "title": "기말고사 일정"}}],
            answer_question="알고리즘 시험 언제야?",
        )
        self.state.select_candidate("1번")

        self.assertEqual(
            self.state.pending_answer_question,
            "알고리즘 시험 언제야?",
        )

    def test_rejects_candidate_number_out_of_range(self):
        self.state.candidate_results = [
            {"notice": {"id": 10, "title": "첫 번째 대회"}},
        ]

        selected, error = self.state.select_candidate("3번")

        self.assertIsNone(selected)
        self.assertIn("1번부터 1번", error)

    def test_selects_candidate_by_ordinal_or_title(self):
        self.state.candidate_results = [
            {"notice": {"id": 10, "title": "AI 아이디어 경진대회"}},
            {"notice": {"id": 20, "title": "소프트웨어 공모전"}},
        ]

        ordinal, _ = self.state.select_candidate("두 번째 알려줘")
        title, _ = self.state.select_candidate("AI 아이디어 경진대회 알려줘")

        self.assertEqual(ordinal["notice"]["id"], 20)
        self.assertEqual(title["notice"]["id"], 10)


class SearchTests(unittest.TestCase):
    def test_rejects_specific_query_with_low_keyword_coverage(self):
        results = [{
            "hybrid_score": 0.77,
            "keyword_score": 0.4,
            "matched_keywords": ["컴퓨터", "공학"],
            "notice": {"id": 1, "published_at": "2026-03-02"},
        }]

        relevant = get_relevant_notices(
            results,
            required_keywords=[
                "2027",
                "컴퓨터",
                "공학",
                "해외여행",
                "지원금",
            ],
        )

        self.assertEqual(relevant, [])

    def test_accepts_specific_query_with_enough_keyword_coverage(self):
        results = [{
            "hybrid_score": 0.85,
            "keyword_score": 0.8,
            "matched_keywords": ["신청", "장학금"],
            "notice": {"id": 1, "published_at": "2026-05-22"},
        }]

        relevant = get_relevant_notices(
            results,
            required_keywords=["신청", "가능", "장학금"],
        )

        self.assertEqual([result["notice"]["id"] for result in relevant], [1])

    def test_relevant_notices_are_returned_newest_first(self):
        results = [
            {
                "hybrid_score": 0.91,
                "keyword_score": 1.0,
                "notice": {"id": 1, "published_at": "2024-03-01"},
            },
            {
                "hybrid_score": 0.89,
                "keyword_score": 1.0,
                "notice": {"id": 2, "published_at": "2026-07-01"},
            },
            {
                "hybrid_score": 0.88,
                "keyword_score": 1.0,
                "notice": {"id": 3, "published_at": None},
            },
        ]

        relevant = get_relevant_notices(results)

        self.assertEqual(
            [result["notice"]["id"] for result in relevant],
            [2, 1, 3],
        )

    def test_recognizes_recent_sort_follow_up(self):
        self.assertTrue(is_recent_sort_request("최신순으로 알려줄래?"))
        self.assertTrue(is_recent_sort_request("작성일 순으로 다시 보여줘"))
        self.assertFalse(is_recent_sort_request("신청 방법 알려줘"))

    def test_recent_sort_keeps_undated_notices_last(self):
        results = [
            {"notice": {"id": 1, "published_at": None}},
            {"notice": {"id": 2, "published_at": "2025-01-01"}},
        ]

        sorted_results = sort_notices_by_published_at(results)

        self.assertEqual(
            [result["notice"]["id"] for result in sorted_results],
            [2, 1],
        )

    def test_result_selection_answer_only_lists_titles(self):
        results = [
            {
                "hybrid_score": 0.9,
                "notice": {
                    "id": 1,
                    "title": "AI 경진대회",
                    "content": "사용자에게 아직 보여주면 안 되는 긴 본문",
                },
            },
            {
                "hybrid_score": 0.8,
                "notice": {
                    "id": 2,
                    "title": "소프트웨어 공모전",
                    "content": "또 다른 긴 본문",
                },
            },
        ]

        answer = create_result_selection_answer(results)

        self.assertIn("관련 공지 2개", answer)
        self.assertIn("1. AI 경진대회", answer)
        self.assertIn("2. 소프트웨어 공모전", answer)
        self.assertNotIn("긴 본문", answer)
        self.assertNotIn("0.9", answer)
        self.assertIn("1번", answer)

    def test_excludes_previously_shown_notice_ids(self):
        notices = [
            {"id": 1, "title": "인턴 공지", "content": "채용 인턴"},
            {"id": 2, "title": "추가 인턴 공지", "content": "인턴 모집"},
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )

        results = search_notices(
            model=FakeModel(),
            question="인턴 공지",
            notices=notices,
            notice_embeddings=embeddings,
            exclude_notice_ids=[1],
        )

        self.assertEqual([result["notice"]["id"] for result in results], [2])

    def test_hydrates_partial_chunk_result_with_full_notice(self):
        partial_result = {
            "hybrid_score": 0.9,
            "notice": {"id": 10, "title": "시험 일정", "content": "일부 청크"},
        }

        class FakeNoticeRepository:
            def fetch_notice(self, notice_id):
                self.notice_id = notice_id
                return {
                    "id": notice_id,
                    "title": "시험 일정",
                    "content": "전체 시험 일정표",
                }

        repository = FakeNoticeRepository()
        hydrated = hydrate_result_notice(partial_result, repository)

        self.assertEqual(repository.notice_id, 10)
        self.assertEqual(hydrated["notice"]["content"], "전체 시험 일정표")
        self.assertEqual(hydrated["hybrid_score"], 0.9)

    def test_exam_route_skips_notice_selection(self):
        self.assertTrue(
            should_answer_without_selection(QueryRoute.EXAM_NOTICE_SEARCH)
        )
        self.assertFalse(
            should_answer_without_selection(QueryRoute.NOTICE_SEARCH)
        )


if __name__ == "__main__":
    unittest.main()
