from __future__ import annotations

import os
import logging
from dataclasses import dataclass, replace
from typing import Any

from .config import GEMINI_EMBEDDING_MODEL_NAME
from .conversation import ConversationState
from .db import (
    AliasRepositoryError,
    ChunkRepository,
    ChunkRepositoryError,
    NoticeRepository,
    NoticeRepositoryError,
    get_alias_repository,
    get_chunk_repository,
    get_notice_repository,
    get_rag_search_source,
)
from .intent import QueryIntent
from .llm import generate_answer, generate_general_answer
from .preprocess import DEFAULT_PREPROCESSOR, QueryPreprocessor
from .retriever import search_notice_chunks
from .router import QueryRoute, get_current_datetime, plan_question, route_to_intent
from .search import (
    KEYWORD_WEIGHT,
    MAX_RESULT_CHOICES,
    SEMANTIC_WEIGHT,
    TOP_K,
    create_query_preprocessor,
    embed_notices,
    get_relevant_notices,
    hydrate_result_notice,
    is_recent_sort_request,
    resolve_relative_time_expression,
    search_notices,
    should_answer_without_selection,
    sort_notices_by_published_at,
)


RESET_PATTERNS = {"초기화", "처음부터", "대화 리셋", "검색 리셋"}
OPEN_NOTICE_TIME_KEYWORDS = {
    "가능",
    "금주",
    "기간",
    "다음",
    "다음달",
    "다음주",
    "당일",
    "마감",
    "모레",
    "모집",
    "신청",
    "오늘",
    "이번",
    "이번달",
    "이번주",
    "접수",
    "주간",
    "지금",
    "현재",
}
logger = logging.getLogger("uvicorn.error")


class ChatbotConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    answer: str
    state: dict
    sources: list[dict]
    selection_required: bool = False
    has_more: bool = False
    page: int = 1
    page_count: int = 1


class ChatbotService:
    """CLI와 HTTP API가 함께 사용하는 한 번의 챗봇 요청 처리기입니다."""

    def __init__(
        self,
        *,
        model: Any,
        preprocessor: QueryPreprocessor,
        search_source: str,
        notice_repository: NoticeRepository,
        chunk_repository: ChunkRepository | None = None,
        notices: list[dict] | None = None,
        notice_embeddings: Any = None,
        retrieval_mode: str = "hybrid",
    ) -> None:
        if retrieval_mode != "hybrid":
            raise ChatbotConfigurationError(
                "임계값 기반 공지 판정은 "
                "RAG_RETRIEVAL_MODE=hybrid만 지원합니다."
            )
        self.model = model
        self.preprocessor = preprocessor
        self.search_source = search_source
        self.notice_repository = notice_repository
        self.chunk_repository = chunk_repository
        self.notices = notices or []
        self.notice_embeddings = notice_embeddings
        self.retrieval_mode = retrieval_mode

    @classmethod
    def create_default(cls) -> "ChatbotService":
        try:
            search_source = get_rag_search_source()
            retrieval_mode = os.getenv(
                "RAG_RETRIEVAL_MODE",
                "hybrid",
            ).lower()
            if retrieval_mode != "hybrid":
                raise ValueError(
                    "임계값 기반 공지 판정은 "
                    "RAG_RETRIEVAL_MODE=hybrid만 지원합니다."
                )

            notice_repository = get_notice_repository()
            chunk_repository = (
                get_chunk_repository() if search_source == "chunks" else None
            )
            from .embedding import GeminiEmbeddingModel

            model = GeminiEmbeddingModel()
        except Exception as error:
            raise ChatbotConfigurationError(
                f"챗봇 초기화에 실패했습니다: {error}"
            ) from error

        try:
            alias_rows = get_alias_repository().fetch_aliases()
            preprocessor = create_query_preprocessor(alias_rows)
        except AliasRepositoryError:
            preprocessor = DEFAULT_PREPROCESSOR

        notices = []
        notice_embeddings = None
        if search_source == "notices":
            try:
                notices = notice_repository.fetch_notices()
            except NoticeRepositoryError as error:
                raise ChatbotConfigurationError(str(error)) from error
            if not notices:
                raise ChatbotConfigurationError("저장된 공지가 없습니다.")
            notice_embeddings = embed_notices(
                model=model,
                notices=notices,
            )

        logger.info(
            "[chat.init] retrieval_mode=%s search_source=%s model=%s",
            retrieval_mode,
            search_source,
            GEMINI_EMBEDDING_MODEL_NAME if model is not None else "none",
        )

        return cls(
            model=model,
            preprocessor=preprocessor,
            search_source=search_source,
            notice_repository=notice_repository,
            chunk_repository=chunk_repository,
            notices=notices,
            notice_embeddings=notice_embeddings,
            retrieval_mode=retrieval_mode,
        )

    def handle_message(
        self,
        message: str,
        state_snapshot: dict | None = None,
        selected_notice_id: int | str | None = None,
        load_more: bool = False,
        candidate_page: int | None = None,
    ) -> ChatResult:
        question = message.strip()
        if (
            not question
            and selected_notice_id is None
            and not load_more
            and candidate_page is None
        ):
            return self._result("질문을 입력해주세요.", ConversationState())

        if question in RESET_PATTERNS:
            return self._result(
                "대화 검색 상태를 초기화했습니다.",
                ConversationState(),
            )

        conversation = self._restore_state(state_snapshot or {})

        if question:
            logger.info(
                "[chat.question] raw=%r has_context=%s active_notice_id=%r",
                self._log_text(question),
                conversation.has_context,
                self._active_notice_id(conversation),
            )

        if candidate_page is not None:
            return self._show_candidate_page(conversation, candidate_page)

        if load_more:
            return self._show_candidate_page(
                conversation,
                conversation.candidate_page + 1,
            )

        if selected_notice_id is not None:
            selected_result = conversation.select_candidate_by_id(
                selected_notice_id
            )
            if selected_result is None:
                return self._result(
                    "현재 검색 결과에 없는 공지입니다. 다시 검색해주세요.",
                    conversation,
                )

            selected_result = self._hydrate(selected_result)
            conversation.active_result = selected_result
            answer_question = (
                conversation.pending_answer_question
                or conversation.last_search_query
                or str(selected_result.get("notice", {}).get("title") or "공지")
            )
            answer = generate_answer(
                question=answer_question,
                resolved_question=answer_question,
                relevant_results=[selected_result],
                answer_mode="summary",
            )
            return self._result(answer, conversation, [selected_result])

        selected_result, selection_error = conversation.select_candidate(question)
        if selection_error:
            return self._result(selection_error, conversation)

        if selected_result:
            selected_result = self._hydrate(selected_result)
            conversation.active_result = selected_result
            answer_question = (
                conversation.pending_answer_question
                or conversation.last_search_query
                or question
            )
            answer = generate_answer(
                question=question,
                resolved_question=answer_question,
                relevant_results=[selected_result],
                answer_mode="summary",
            )
            return self._result(answer, conversation, [selected_result])

        if (
            conversation.has_candidates
            and conversation.active_result is None
            and is_recent_sort_request(question)
        ):
            conversation.candidate_results = sort_notices_by_published_at(
                conversation.candidate_results
            )
            visible_count = max(
                len(conversation.shown_notice_ids),
                min(MAX_RESULT_CHOICES, len(conversation.candidate_results)),
            )
            visible_results = conversation.candidate_results[:visible_count]
            visible_ids = self._notice_ids(visible_results)
            conversation.shown_notice_ids = visible_ids
            conversation.referenced_notice_ids = visible_ids
            return self._result(
                self._create_card_selection_answer(
                    visible_results,
                    total_count=len(conversation.candidate_results),
                    page=1,
                    page_count=self._page_count(conversation.candidate_results),
                ),
                conversation,
                visible_results,
                selection_required=True,
                has_more=self._has_hidden_candidates(conversation),
                page=1,
                page_count=self._page_count(conversation.candidate_results),
            )

        processed_query = self.preprocessor.process(question)
        answer_question = processed_query.normalized

        logger.info(
            "[chat.preprocess] normalized=%r aliases=%s",
            self._log_text(processed_query.normalized),
            [
                f"{match.alias}->{match.meaning}"
                for match in processed_query.resolved_aliases
            ],
        )

        plan = plan_question(
            question=processed_query.normalized,
            router_context=conversation.build_router_context(),
        )
        conversation.add_message("user", processed_query.normalized)
        query_route = plan.route
        logger.info(
            "[chat.router] route=%s search_query=%r confidence=%.3f source=%s",
            plan.route.value,
            self._log_text(plan.search_query),
            plan.confidence,
            plan.source,
        )

        if plan.route == QueryRoute.MORE_NOTICE_SEARCH:
            return self._show_candidate_page(
                conversation,
                conversation.candidate_page + 1,
            )

        if plan.route == QueryRoute.SELECTED_NOTICE_ANSWER:
            if conversation.active_result is None:
                return self._result(
                    "먼저 궁금한 공지를 검색하고 선택해주세요.",
                    conversation,
                    conversation.candidate_results,
                    selection_required=conversation.has_candidates,
                    has_more=self._has_hidden_candidates(conversation),
                )
            previous_search_query = conversation.last_search_query
            resolved_follow_up = (
                f"이전 검색 대상: {previous_search_query}\n"
                f"현재 후속 질문: {processed_query.normalized}"
                if previous_search_query
                else processed_query.normalized
            )
            answer = generate_answer(
                question=question,
                resolved_question=resolved_follow_up,
                relevant_results=[conversation.active_result],
                answer_mode="focused",
            )
            return self._result(
                answer,
                conversation,
                [conversation.active_result],
            )

        if plan.route == QueryRoute.GENERAL_CHAT:
            return self._result(
                generate_general_answer(plan.search_query),
                conversation,
            )

        if plan.route == QueryRoute.CLARIFICATION:
            return self._result(
                plan.clarification
                or "어떤 종류의 공지를 찾는지 조금 더 알려주세요.",
                conversation,
            )

        intent = route_to_intent(plan.route)
        resolved_search_query = resolve_relative_time_expression(
            plan.search_query
        )
        processed_query = replace(
            processed_query,
            normalized=resolved_search_query,
        )

        resolution = conversation.resolve(processed_query, intent)
        logger.info(
            "[chat.resolve] search_question=%r intent=%s excluded_ids=%s",
            self._log_text(resolution.search_question),
            resolution.intent.value,
            list(resolution.exclude_notice_ids),
        )
        if resolution.clarification:
            return self._result(resolution.clarification, conversation)

        search_results = self._search(
            resolution.search_question,
            query_route=query_route,
            exclude_notice_ids=resolution.exclude_notice_ids,
        )
        self._log_search_results("search", search_results)
        is_time_only_open_search = self._is_time_only_open_search(
            resolution.search_question,
            query_route,
        )
        if is_time_only_open_search:
            # 미래 마감일 조건을 이미 통과한 결과다. "이번 주", "신청 가능"
            # 같은 시간 표현은 공지 본문에 없을 수 있으므로 의미/키워드
            # 임계값으로 다시 제거하지 않는다.
            relevant_results = sort_notices_by_published_at(search_results)
        else:
            relevant_results = get_relevant_notices(
                results=search_results,
                required_keywords=self.preprocessor.extract_keywords(
                    resolution.search_question
                ),
            )
        relevant_ids = set(self._notice_ids(relevant_results))
        logger.info(
            "[chat.threshold] searched=%d selected=%d selected_ids=%s",
            len(search_results),
            len(relevant_results),
            list(relevant_ids),
        )
        self._log_search_results(
            "candidate",
            relevant_results,
            passed_ids=relevant_ids,
        )

        if not relevant_results:
            if query_route == QueryRoute.OPEN_NOTICE_SEARCH:
                answer = (
                    "현재 신청 가능한 공지를 확인하지 못했습니다. "
                    "공지의 마감일 정보가 아직 등록되지 않았을 수도 있습니다."
                )
            elif resolution.intent == QueryIntent.MORE_RESULTS:
                answer = "현재 저장된 공지 중 추가 결과가 없습니다."
            else:
                answer = "관련 공지를 찾지 못했습니다."
            return self._result(answer, conversation)

        displayed_results = relevant_results[:MAX_RESULT_CHOICES]
        has_more = len(relevant_results) > len(displayed_results)
        conversation.record_results(
            resolution,
            relevant_results,
            answer_question=answer_question,
        )
        conversation.shown_notice_ids = self._notice_ids(displayed_results)
        conversation.referenced_notice_ids = self._notice_ids(displayed_results)

        if (
            should_answer_without_selection(query_route)
            or (
                len(displayed_results) == 1
                and not is_time_only_open_search
            )
        ):
            direct_results = [self._hydrate(result) for result in displayed_results]
            conversation.candidate_results = direct_results
            conversation.active_result = direct_results[0]
            answer = generate_answer(
                question=question,
                resolved_question=resolution.search_question,
                relevant_results=direct_results,
                answer_mode="focused",
            )
            return self._result(answer, conversation, direct_results)

        return self._result(
            self._create_card_selection_answer(
                displayed_results,
                total_count=len(relevant_results),
                page=1,
                page_count=self._page_count(relevant_results),
            ),
            conversation,
            displayed_results,
            selection_required=True,
            has_more=has_more,
            page=1,
            page_count=self._page_count(relevant_results),
        )

    def _show_candidate_page(
        self,
        conversation: ConversationState,
        page: int,
    ) -> ChatResult:
        """최초 검색 후보를 재검색 없이 3개 단위 페이지로 보여줍니다."""
        if not conversation.has_candidates:
            return self._result(
                "먼저 궁금한 공지를 검색해주세요.",
                conversation,
            )

        page_count = self._page_count(conversation.candidate_results)
        current_page = max(1, min(page, page_count))
        start = (current_page - 1) * MAX_RESULT_CHOICES
        visible_results = conversation.candidate_results[
            start : start + MAX_RESULT_CHOICES
        ]
        visible_ids = self._notice_ids(visible_results)
        conversation.candidate_page = current_page
        conversation.shown_notice_ids = visible_ids
        conversation.referenced_notice_ids = visible_ids

        return self._result(
            self._create_card_selection_answer(
                visible_results,
                total_count=len(conversation.candidate_results),
                page=current_page,
                page_count=page_count,
            ),
            conversation,
            visible_results,
            selection_required=True,
            has_more=current_page < page_count,
            page=current_page,
            page_count=page_count,
        )

    @staticmethod
    def _has_hidden_candidates(conversation: ConversationState) -> bool:
        return (
            len(conversation.shown_notice_ids)
            < len(conversation.candidate_results)
        )

    def _is_time_only_open_search(
        self,
        question: str,
        query_route: QueryRoute | None,
    ) -> bool:
        if query_route != QueryRoute.OPEN_NOTICE_SEARCH:
            return False

        keywords = self.preprocessor.extract_keywords(question)
        return bool(keywords) and all(
            keyword in OPEN_NOTICE_TIME_KEYWORDS
            for keyword in keywords
        )

    @staticmethod
    def _create_card_selection_answer(
        results: list[dict],
        total_count: int | None = None,
        page: int = 1,
        page_count: int = 1,
    ) -> str:
        total = total_count if total_count is not None else len(results)
        count_text = (
            f"관련 공지 총 {total}개 중 {len(results)}개를 보여드렸습니다."
            if total > len(results)
            else f"관련 공지 {total}개를 찾았습니다."
        )
        page_text = f" ({page}/{page_count}페이지)" if page_count > 1 else ""
        return f"{count_text}{page_text} 궁금한 공지 카드를 선택해주세요."

    @staticmethod
    def _page_count(results: list[dict]) -> int:
        return max(1, (len(results) + MAX_RESULT_CHOICES - 1) // MAX_RESULT_CHOICES)

    @staticmethod
    def _notice_ids(results: list[dict]) -> list[Any]:
        return [
            notice_id
            for result in results
            if (notice_id := result.get("notice", {}).get("id")) is not None
        ]

    @staticmethod
    def _active_notice_id(conversation: ConversationState) -> Any | None:
        if conversation.active_result is None:
            return None
        return conversation.active_result.get("notice", {}).get("id")

    @staticmethod
    def _log_text(value: Any, limit: int = 300) -> str:
        return " ".join(str(value or "").split())[:limit]

    @classmethod
    def _log_search_results(
        cls,
        stage: str,
        results: list[dict],
        passed_ids: set[Any] | None = None,
    ) -> None:
        for rank, result in enumerate(results, start=1):
            notice = result.get("notice") or {}
            notice_id = notice.get("id")
            logger.info(
                "[chat.%s] rank=%d id=%r title=%r published_at=%r "
                "hybrid=%s semantic=%s keyword=%s filter_status=%s",
                stage,
                rank,
                notice_id,
                cls._log_text(notice.get("title")),
                notice.get("published_at"),
                cls._log_score(result.get("hybrid_score")),
                cls._log_score(result.get("semantic_score")),
                cls._log_score(result.get("keyword_score")),
                (
                    "pending"
                    if passed_ids is None
                    else "passed"
                    if notice_id in passed_ids
                    else "rejected"
                ),
            )

    @staticmethod
    def _log_score(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "n/a"

    def _search(
        self,
        question: str,
        *,
        query_route: QueryRoute | None,
        exclude_notice_ids: tuple | list | set,
    ) -> list[dict]:
        if self.search_source == "chunks":
            if self.chunk_repository is None:
                raise ChatbotConfigurationError(
                    "공지 청크 검색 저장소가 준비되지 않았습니다."
                )
            return search_notice_chunks(
                model=self.model,
                question=question,
                repository=self.chunk_repository,
                top_k=TOP_K,
                semantic_weight=SEMANTIC_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
                deadline_from=(
                    get_current_datetime().isoformat()
                    if query_route == QueryRoute.OPEN_NOTICE_SEARCH
                    else None
                ),
                exclude_notice_ids=exclude_notice_ids,
                preprocessor=self.preprocessor,
            )

        return search_notices(
            model=self.model,
            question=question,
            notices=self.notices,
            notice_embeddings=self.notice_embeddings,
            top_k=TOP_K,
            exclude_notice_ids=exclude_notice_ids,
            preprocessor=self.preprocessor,
        )

    def _hydrate(self, result: dict) -> dict:
        if self.search_source != "chunks":
            return result
        return hydrate_result_notice(result, repository=self.notice_repository)

    def _restore_state(self, snapshot: dict) -> ConversationState:
        conversation = ConversationState(
            last_search_query=snapshot.get("last_search_query"),
            referenced_notice_ids=list(
                snapshot.get("referenced_notice_ids") or []
            )[:100],
            shown_notice_ids=list(snapshot.get("shown_notice_ids") or [])[:100],
            pending_answer_question=snapshot.get("pending_answer_question"),
            router_context=list(snapshot.get("router_context") or [])[-6:],
            candidate_page=max(1, int(snapshot.get("candidate_page") or 1)),
        )

        candidate_ids = list(snapshot.get("candidate_notice_ids") or [])[:100]
        conversation.candidate_results = self._restore_results(candidate_ids)

        active_notice_id = snapshot.get("active_notice_id")
        if active_notice_id is not None:
            restored = self._restore_results([active_notice_id])
            conversation.active_result = restored[0] if restored else None
        return conversation

    def _restore_results(self, notice_ids: list[Any]) -> list[dict]:
        results = []
        seen = set()
        for notice_id in notice_ids:
            if notice_id in seen:
                continue
            seen.add(notice_id)
            notice = self.notice_repository.fetch_notice(notice_id)
            if notice is not None:
                results.append({"notice": notice})
        return results

    @staticmethod
    def _sources(results: list[dict] | None) -> list[dict]:
        sources = []
        seen = set()
        for result in results or []:
            notice = result.get("notice") or {}
            source_id = notice.get("id")
            if source_id in seen:
                continue
            seen.add(source_id)
            sources.append({
                "id": source_id,
                "title": notice.get("title") or "제목 없음",
                "publishedAt": notice.get("published_at"),
                "url": notice.get("url"),
            })
        return sources

    def _result(
        self,
        answer: str,
        conversation: ConversationState,
        results: list[dict] | None = None,
        selection_required: bool = False,
        has_more: bool = False,
        page: int = 1,
        page_count: int = 1,
    ) -> ChatResult:
        conversation.add_message("assistant", answer[:400])
        return ChatResult(
            answer=answer,
            state=conversation.snapshot(),
            sources=self._sources(results),
            selection_required=selection_required,
            has_more=has_more,
            page=page,
            page_count=page_count,
        )
