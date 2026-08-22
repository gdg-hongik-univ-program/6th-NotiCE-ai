from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

if __package__:
    from .embedding import GeminiEmbeddingModel
    from .config import EMBEDDING_MODEL_NAME
    from .conversation import ConversationState
    from .db import (
        AliasRepositoryError,
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
    from .router import (
        QueryRoute,
        get_current_datetime,
        plan_question,
        route_to_intent,
    )
else:
    from embedding import GeminiEmbeddingModel
    from config import EMBEDDING_MODEL_NAME
    from conversation import ConversationState
    from db import (
        AliasRepositoryError,
        ChunkRepositoryError,
        NoticeRepository,
        NoticeRepositoryError,
        get_alias_repository,
        get_chunk_repository,
        get_notice_repository,
        get_rag_search_source,
    )
    from intent import QueryIntent
    from llm import generate_answer, generate_general_answer
    from preprocess import DEFAULT_PREPROCESSOR, QueryPreprocessor
    from retriever import search_notice_chunks
    from router import (
        QueryRoute,
        get_current_datetime,
        plan_question,
        route_to_intent,
    )


# =========================================================
# 기본 설정
# =========================================================

# 검색할 최대 공지 개수
TOP_K = 15

# 첫 응답에서 사용자가 고를 수 있도록 보여줄 최대 공지 수
MAX_RESULT_CHOICES = 3

# 하이브리드 검색 가중치
SEMANTIC_WEIGHT = 0.85
KEYWORD_WEIGHT = 0.15

# 정확 키워드가 충분히 맞으면 의미 점수가 낮아도 관련 공지로 판단
MIN_KEYWORD_SCORE = 0.65

# 키워드 문자열이 달라도 의미가 충분히 유사하다고 판단되면 관련 공지로 판단
MIN_SEMANTIC_SCORE = 0.7

# 의미 점수 차이가 이 값보다 큰 공지는 제외
MAX_SEMANTIC_SCORE_GAP = 0.05

# 구체적인 질문은 전체 핵심어 중 절반 이상이 실제 공지에 등장해야 함
MIN_SPECIFIC_QUERY_KEYWORDS = 2
MIN_KEYWORD_COVERAGE = 0.5

MIN_KEYWORD_BACKUP_SEMANTIC = 0.66

# 최고 의미 점수가 이 값보다 낮으면 관련 공지 없음 (임시 추가)
MIN_SEARCH_TOP_SEMANTIC_SCORE = 0.7

RECENT_SORT_PATTERNS = (
    "최신순",
    "최신 순",
    "최근순",
    "최근 순",
    "날짜순",
    "날짜 순",
    "작성일순",
    "작성일 순",
)

# =========================================================
# 데이터 불러오기
# =========================================================

def load_notices() -> list[dict]:
    """환경 설정에 따라 Supabase 또는 샘플 JSON에서 공지를 불러옵니다."""
    return get_notice_repository().fetch_notices()


def create_notice_text(notice: dict) -> str:
    """
    공지의 제목, 카테고리, 작성일, 본문을
    하나의 검색용 문자열로 합칩니다.
    """
    return (
        f"제목: {notice.get('title', '')}\n"
        f"카테고리: {notice.get('category') or ''}\n"
        f"작성일: {notice.get('published_at', '')}\n"
        f"내용: {notice.get('content', '')}"
    )


# =========================================================
# 키워드 검색
# =========================================================

def normalize_text(text: str) -> str:
    """키워드 비교를 위해 대소문자와 공백을 정리합니다."""
    return " ".join(text.lower().split())

def extract_keywords(
    question: str,
    preprocessor: QueryPreprocessor = DEFAULT_PREPROCESSOR,
) -> list[str]:
    """사용자 질문에서 키워드 검색에 사용할 단어를 추출합니다."""
    return preprocessor.extract_keywords(question)

def calculate_keyword_score(
    question: str,
    notice: dict,
    keywords: list[str] | None = None,
    preprocessor: QueryPreprocessor = DEFAULT_PREPROCESSOR,
) -> tuple[float, list[str]]:
    """
    질문 키워드가 공지 제목/카테고리/본문에 얼마나 직접 등장하는지 계산합니다.

    제목과 카테고리에 등장한 키워드는 본문보다 조금 더 높은 점수를 줍니다.
    """
    if keywords is None:
        keywords = extract_keywords(question, preprocessor=preprocessor)

    if not keywords:
        return 0.0, []

    title_text = normalize_text(str(notice.get("title", "")))
    category_text = normalize_text(str(notice.get("category", "")))
    content_text = normalize_text(str(notice.get("content", "")))
    url_text = normalize_text(str(notice.get("url", "")))

    matched_keywords = []
    score = 0.0

    for keyword in keywords:
        keyword_score = 0.0

        if keyword in title_text:
            keyword_score += 1.0

        if keyword in category_text:
            keyword_score += 0.8

        if keyword in content_text:
            keyword_score += 0.6

        if keyword in url_text:
            keyword_score += 0.2

        if keyword_score > 0:
            matched_keywords.append(keyword)
            score += min(keyword_score, 1.0)

    return score / len(keywords), matched_keywords


# =========================================================
# 임베딩 생성
# =========================================================

def embed_notices(
    model: SentenceTransformer,
    notices: list[dict],
) -> np.ndarray:
    """모든 공지를 임베딩 벡터로 변환합니다."""
    notice_texts = [
        f"passage: {create_notice_text(notice)}"
        for notice in notices
    ]

    embeddings = model.encode(
        notice_texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


# =========================================================
# 공지 검색
# =========================================================

def search_notices(
    model: SentenceTransformer,
    question: str,
    notices: list[dict],
    notice_embeddings: np.ndarray,
    top_k: int = TOP_K,
    exclude_notice_ids: tuple | list | set | None = None,
    preprocessor: QueryPreprocessor = DEFAULT_PREPROCESSOR,
) -> list[dict]:
    """
    의미 검색과 키워드 검색을 함께 사용해 관련 공지를 반환합니다.
    """
    question_embedding = model.encode(
        f"query: {question}",
        normalize_embeddings=True,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32,
    )

    results = []
    excluded_ids = set(exclude_notice_ids or ())
    keywords = extract_keywords(question, preprocessor=preprocessor)

    # 모든 벡터가 정규화되어 있으므로 내적 결과가 코사인 유사도와 같음
    semantic_scores = notice_embeddings @ question_embedding

    for index, notice in enumerate(notices):
        if notice.get("id") in excluded_ids:
            continue

        semantic_score = float(semantic_scores[index])
        keyword_score, matched_keywords = calculate_keyword_score(
            question=question,
            notice=notice,
            keywords=keywords,
            preprocessor=preprocessor,
        )
        hybrid_score = (
            semantic_score * SEMANTIC_WEIGHT
            + keyword_score * KEYWORD_WEIGHT
        )

        results.append({
            "score": hybrid_score,
            "hybrid_score": hybrid_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "matched_keywords": matched_keywords,
            "notice": notice,
        })

    results.sort(key=lambda result: result["hybrid_score"], reverse=True)

    return results[: min(top_k, len(results))]


def get_relevant_notices(
    results: list[dict],
    min_keyword_score: float = MIN_KEYWORD_SCORE,
    min_semantic_score: float = MIN_SEMANTIC_SCORE,
    min_search_top_semantic_score: float = MIN_SEARCH_TOP_SEMANTIC_SCORE,
    max_semantic_score_gap: float = MAX_SEMANTIC_SCORE_GAP,
    min_keyword_backup_semantic: float = MIN_KEYWORD_BACKUP_SEMANTIC,
    required_keywords: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """
    검색 결과 중 질문과 관련성이 충분한 공지들을 반환합니다.

    1. 검색 결과 1위 점수가 너무 낮으면 전체 검색 실패
    2. 1위 점수와 차이가 크지 않은 공지들을 함께 선택
    """
    if not results:
        return []

    top_keyword_score = max(
        result["keyword_score"] for result in results
    )
    
    top_semantic_score = max(
        result["semantic_score"] for result in results
    )
    
    # 키워드 점수가 충분히 높거나 의미 점수가 충분히 높으면 관련 공지로 판단
    has_reliable_search_signal = (
        top_keyword_score >= min_keyword_score
        or top_semantic_score >= min_search_top_semantic_score
    )
    
    # 하이브리드 점수와 키워드 점수가 모두 낮으면 관련 공지 없음
    if not has_reliable_search_signal:
        return []

    relevant_results = []
    unique_required_keywords = {
        normalize_text(keyword)
        for keyword in (required_keywords or ())
        if normalize_text(keyword)
    }

    for result in results:
        semantic_score_gap = top_semantic_score - result["semantic_score"]
        
        has_strong_semantic_match = (
            result["semantic_score"] >= min_semantic_score
            and semantic_score_gap <= max_semantic_score_gap
        )

        has_supported_keyword_match = (
            result["keyword_score"] >= min_keyword_score
            and result["semantic_score"] >= min_keyword_backup_semantic
        )
        
        if (
            has_supported_keyword_match
            or has_strong_semantic_match
        ):
            if len(unique_required_keywords) >= MIN_SPECIFIC_QUERY_KEYWORDS:
                matched_keywords = {
                    normalize_text(keyword)
                    for keyword in result.get("matched_keywords", [])
                }
                keyword_coverage = (
                    len(unique_required_keywords & matched_keywords)
                    / len(unique_required_keywords)
                )

                if keyword_coverage < MIN_KEYWORD_COVERAGE:
                    continue

            relevant_results.append(result)

    return sort_notices_by_published_at(relevant_results)


def sort_notices_by_published_at(results: list[dict]) -> list[dict]:
    """관련도 순서를 날짜가 같은 공지의 보조 기준으로 유지하며 최신순 정렬합니다."""
    return sorted(
        results,
        key=lambda result: str(
            result.get("notice", {}).get("published_at") or ""
        ),
        reverse=True,
    )


def is_recent_sort_request(question: str) -> bool:
    """현재 검색 결과를 최신순으로 다시 보여달라는 요청인지 확인합니다."""
    normalized = " ".join(question.lower().split())
    return any(pattern in normalized for pattern in RECENT_SORT_PATTERNS)

def resolve_relative_time_expression(
    question: str,
    now=None,
) -> str:
    """상대 시간 표현을 검색 가능한 절대 표현으로 변환합니다."""
    current = get_current_datetime(now)
    year = current.year
    month = current.month

    resolved = question

    # 연도 표현
    resolved = resolved.replace("올해", f"{year}년")
    resolved = resolved.replace("작년", f"{year - 1}년")
    resolved = resolved.replace("내년", f"{year + 1}년")

    # 현재 학기 계산
    if 3 <= month <= 8:
        current_semester_year = year
        current_semester = 1
    elif 9 <= month <= 12:
        current_semester_year = year
        current_semester = 2
    else:
        current_semester_year = year - 1
        current_semester = 2

    # 이번 학기
    if(
        "이번학기" in resolved
        or "이번 학기" in resolved
        or "지금학기" in resolved
        or "지금 학기" in resolved
        or "현재학기" in resolved
        or "현재 학기" in resolved
    ):

        # 졸업 질문은 실제 졸업 월로 변환
        if "졸업" in resolved:
            if 3 <= month <= 8:
                replacement = f"{year}년 8월"
            elif 9 <= month <= 12:
                replacement = f"{year + 1}년 2월"
            else:
                replacement = f"{year}년 2월"
        else:
            replacement = (
                f"{current_semester_year}-{current_semester}학기"
            )

        resolved = resolved.replace("이번학기", replacement)
        resolved = resolved.replace("이번 학기", replacement)
        resolved = resolved.replace("지금학기", replacement)
        resolved = resolved.replace("지금 학기", replacement)
        resolved = resolved.replace("현재학기", replacement)
        resolved = resolved.replace("현재 학기", replacement)

    # 지난 학기
    if (
        "지난학기" in resolved 
        or "지난 학기" in resolved
        or "저번학기" in resolved
        or "저번 학기" in resolved
        or "이전학기" in resolved
        or "이전 학기" in resolved
    ):
        if current_semester == 1:
            previous_year = current_semester_year - 1
            previous_semester = 2
        else:
            previous_year = current_semester_year
            previous_semester = 1

        replacement = f"{previous_year}-{previous_semester}학기"

        resolved = resolved.replace("지난학기", replacement)
        resolved = resolved.replace("지난 학기", replacement)
        resolved = resolved.replace("저번학기", replacement)
        resolved = resolved.replace("저번 학기", replacement)
        resolved = resolved.replace("이전학기", replacement)
        resolved = resolved.replace("이전 학기", replacement)

    # 다음 학기
    if "다음학기" in resolved or "다음 학기" in resolved:
        if current_semester == 1:
            next_year = current_semester_year
            next_semester = 2
        else:
            next_year = current_semester_year + 1
            next_semester = 1

        replacement = f"{next_year}-{next_semester}학기"

        resolved = resolved.replace("다음학기", replacement)
        resolved = resolved.replace("다음 학기", replacement)

    return resolved

def create_result_selection_answer(results: list[dict]) -> str:
    """본문을 노출하지 않고 선택 가능한 공지 제목 목록을 만듭니다."""
    lines = [f"관련 공지 {len(results)}개를 찾았습니다.", ""]

    for index, result in enumerate(results, start=1):
        notice = result["notice"]
        title = notice.get("title") or "제목 없음"
        lines.append(f"{index}. {title}")

    lines.extend([
        "",
        "궁금한 공지의 번호나 제목을 입력해주세요. 예: 1번",
    ])
    return "\n".join(lines)


def create_query_preprocessor(alias_rows: list[dict]) -> QueryPreprocessor:
    aliases = {
        str(row["alias"]): str(row["meaning"])
        for row in alias_rows
    }
    return QueryPreprocessor(aliases=aliases)


def hydrate_result_notice(
    result: dict,
    repository: NoticeRepository,
) -> dict:
    """선택된 검색 결과의 부분 청크를 공지 전체 본문으로 교체합니다."""
    notice_id = result.get("notice", {}).get("id")

    if notice_id is None:
        return result

    full_notice = repository.fetch_notice(notice_id)

    if full_notice is None:
        return result

    return {
        **result,
        "notice": full_notice,
    }


def should_answer_without_selection(route: QueryRoute | None) -> bool:
    """사용자 질문에 바로 답해야 하는 검색 경로인지 반환합니다."""
    return route == QueryRoute.EXAM_NOTICE_SEARCH


# =========================================================
# 결과 출력
# =========================================================

def print_and_record_answer(
    conversation: ConversationState,
    answer: str,
    context_message: str | None = None,
) -> None:
    """챗봇 답변을 출력하고 Router용 대화 기록을 저장합니다."""
    conversation.add_message(
        "assistant",
        context_message or answer,
    )
    print("\n===== 챗봇 답변 =====")
    print(answer)

def print_search_failure(results: list[dict]) -> None:
    """관련 공지를 찾지 못했을 때 검색 정보를 출력합니다."""

    if not results:
        return

    top_score = results[0]["hybrid_score"]
    print(f"최고 하이브리드 점수: {top_score:.4f}")
    print(f"의미 검색 점수: {results[0]['semantic_score']:.4f}")
    print(f"키워드 검색 점수: {results[0]['keyword_score']:.4f}")

    if len(results) >= 2:
        second_score = results[1]["hybrid_score"]
        score_gap = top_score - second_score

        print(f"2위 하이브리드 점수: {second_score:.4f}")
        print(f"1위와 2위 점수 차이: {score_gap:.4f}")
        
def print_search_results(
    title: str,
    results: list[dict],
    passed_notice_ids: set | None = None,
) -> None:
    """임계값 실험을 위해 검색 결과와 점수를 출력합니다."""
    print(f"\n===== {title} =====")

    if not results:
        print("결과 없음")
        return

    for rank, result in enumerate(results, start=1):
        notice = result.get("notice", {})
        notice_id = notice.get("id")
        
        status = ""
        if passed_notice_ids is not None:
            status = (
                " [✅ 통과]"
                if notice_id in passed_notice_ids
                else " [❌ 탈락]"
            )

        print(
            f"{rank}.{status} {notice.get('title', '제목 없음')}\n"
            f"   hybrid={result.get('hybrid_score', 0.0):.4f} "
            f"semantic={result.get('semantic_score', 0.0):.4f} "
            f"keyword={result.get('keyword_score', 0.0):.4f}\n"
            f"   matched_keywords={result.get('matched_keywords', [])}"
        )        


# =========================================================
# 프로그램 실행
# =========================================================

def main() -> None:

    notices = []
    notice_embeddings = None
    chunk_repository = None

    try:
        search_source = get_rag_search_source()
        notice_repository = get_notice_repository()

        if search_source == "chunks":
            chunk_repository = get_chunk_repository()
    except (NoticeRepositoryError, ChunkRepositoryError) as error:
        print(f"챗봇 실행 설정을 확인해주세요: {error}")
        return

    print("Gemini 임베딩 모델을 불러오는 중입니다.")

    try:
        model = GeminiEmbeddingModel()
    except Exception as error:
        print(f"Gemini 임베딩 모델을 불러오지 못했습니다: {error}")
        return

    try:
        alias_rows = get_alias_repository().fetch_aliases()
        preprocessor = create_query_preprocessor(alias_rows)
        print(f"은어 사전 {len(alias_rows)}개를 불러왔습니다.")
    except AliasRepositoryError as error:
        preprocessor = DEFAULT_PREPROCESSOR
        print(f"은어 사전을 불러오지 못해 기본 검색으로 진행합니다: {error}")

    if search_source == "chunks":
        print("Supabase notice_chunks RPC 검색 모드입니다.")
    else:
        notices = notice_repository.fetch_notices()

        if not notices:
            print("저장된 공지가 없습니다.")
            return

        print(
            f"{notice_repository.source_name}에서 "
            f"공지 {len(notices)}개를 불러왔습니다."
        )
        print("Gemini 공지 임베딩을 생성합니다.")

        notice_embeddings = embed_notices(
            model=model,
            notices=notices,
        )

        print("임베딩 생성 완료")
        print(f"임베딩 배열 크기: {notice_embeddings.shape}")

    conversation = ConversationState()

    while True:
        try:
            question = input(
                "\n질문을 입력하세요. 종료하려면 exit 입력: "
            ).strip()
        except EOFError:
            print("\n입력이 종료되어 프로그램을 종료합니다.")
            break

        if question.lower() == "exit":
            print("프로그램을 종료합니다.")
            break

        if not question:
            print("질문을 입력해주세요.")
            continue

        selected_result, selection_error = conversation.select_candidate(question)

        if selection_error:
            print(f"\n===== 챗봇 답변 =====\n{selection_error}")
            continue

        if selected_result:
            if search_source == "chunks":
                try:
                    selected_result = hydrate_result_notice(
                        selected_result,
                        repository=notice_repository,
                    )
                    conversation.active_result = selected_result
                except NoticeRepositoryError as error:
                    print(f"공지 전체 본문을 불러오지 못했습니다: {error}")

            answer_question = (
                conversation.pending_answer_question
                or conversation.last_search_query
                or question
            )
            answer = generate_answer(
                question=question,
                resolved_question=answer_question,
                relevant_results=[selected_result],
                answer_mode="focused",
            )
            notice = selected_result.get("notice", {})
            print_and_record_answer(
                conversation,
                answer,
                context_message=(
                    f"'{notice.get('title') or '제목 없음'}' 공지를 선택해 답변함"
                ),
            )
            continue

        if (
            conversation.has_candidates
            and conversation.active_result is None
            and is_recent_sort_request(question)
        ):
            sorted_results = sort_notices_by_published_at(
                conversation.candidate_results
            )
            conversation.candidate_results = sorted_results
            answer = create_result_selection_answer(sorted_results)
            print_and_record_answer(
                conversation,
                answer,
                context_message = "이전 검색 결과를 최신순으로 다시 정렬해 제시함",
            )
            continue

        processed_query = preprocessor.process(question)
        
        if processed_query.resolved_aliases:
            resolved_text = ", ".join(
                f"{match.alias} → {match.meaning}"
                for match in processed_query.resolved_aliases
            )
            print(f"은어 해석: {resolved_text}")

        query_route = None

        router_context = conversation.build_router_context()
        plan = plan_question(
            question=processed_query.normalized,
            router_context=router_context,
        )
        
        conversation.add_message(
            "user",
            processed_query.normalized,
        )
            
        query_route = plan.route
        route_source = "LLM" if plan.source == "llm" else "기본 규칙"
        print(f"질문 경로: {plan.route.value} ({route_source})")

        if plan.route == QueryRoute.SELECTED_NOTICE_ANSWER:
            if conversation.active_result is None:
                print(
                    "\n===== 챗봇 답변 =====\n"
                    "먼저 궁금한 공지를 검색하고 선택해주세요."
                )
                continue
            
            previous_search_query = conversation.last_search_query

            if previous_search_query:
                follow_up_question = (
                    f"이전 검색 대상: {previous_search_query}\n"
                    f"현재 후속 질문: {processed_query.normalized}"
                )
            else:
                follow_up_question = processed_query.normalized

            answer = generate_answer(
                question=question,
                resolved_question=follow_up_question,
                relevant_results=[conversation.active_result],
                answer_mode="focused",
            )
            notice = conversation.active_result.get("notice", {})
            print_and_record_answer(
                conversation,
                answer,
                context_message=(
                    f"[선택된 공지] '{notice.get('title') or '제목 없음'}'에 대해 "
                    f"사용자의 질문에 답변함"
                ),
            )
            continue

        if plan.route == QueryRoute.GENERAL_CHAT:
            answer = generate_general_answer(plan.search_query)
            
            print_and_record_answer(conversation, answer)
            continue

        if plan.route == QueryRoute.CLARIFICATION:
            clarification = plan.clarification or (
                "어떤 종류의 공지를 찾는지 조금 더 알려주세요."
            )
            print_and_record_answer(
                conversation,
                clarification,
                context_message="사용자에게 질문을 더 구체적으로 알려달라고 요청함",
            )
            continue

        intent = route_to_intent(plan.route)
        resolved_search_query = resolve_relative_time_expression(plan.search_query)
        processed_query = replace(
            processed_query,
            normalized=resolved_search_query,
        )

        resolution = conversation.resolve(processed_query, intent)
        resolved_question = resolution.search_question
        
        keywords = preprocessor.extract_keywords(
            resolution.search_question
        )
        print(f"추출 키워드: {keywords}")

        if intent == QueryIntent.MORE_RESULTS:
            answer_question = resolution.search_question

        if resolution.clarification:
            print(f"\n===== 챗봇 답변 =====\n{resolution.clarification}")
            continue

        if resolution.search_question != question:
            print(f"질문 해석: {resolution.search_question}")

        try:
            if search_source == "chunks":
                search_results = search_notice_chunks(
                    model=model,
                    question=resolution.search_question,
                    repository=chunk_repository,
                    top_k=TOP_K,
                    semantic_weight=SEMANTIC_WEIGHT,
                    keyword_weight=KEYWORD_WEIGHT,
                    deadline_from=(
                        get_current_datetime().isoformat()
                        if query_route == QueryRoute.OPEN_NOTICE_SEARCH
                        else None
                    ),
                    exclude_notice_ids=resolution.exclude_notice_ids,
                    preprocessor=preprocessor,
                )
            else:
                search_results = search_notices(
                    model=model,
                    question=resolution.search_question,
                    notices=notices,
                    notice_embeddings=notice_embeddings,
                    top_k=TOP_K,
                    exclude_notice_ids=resolution.exclude_notice_ids,
                    preprocessor=preprocessor,
                )
        except ChunkRepositoryError as error:
            print(f"\n청크 검색을 사용할 수 없습니다: {error}")
            print(
                "DB 준비 전에는 RAG_SEARCH_SOURCE=notices로 실행해주세요."
            )
            continue

        relevant_results = get_relevant_notices(
            results=search_results,
            required_keywords=preprocessor.extract_keywords(
                resolution.search_question
            ),
        )
        
        passed_notice_ids = {
            result.get("notice", {}).get("id")
            for result in relevant_results
        }
        
        print_search_results(
            "필터링 전 검색 결과",
            search_results,
            passed_notice_ids=passed_notice_ids,
        )
        
        print_search_results(
            "필터링 후 관련 공지",
            relevant_results,
        )
        
        if not relevant_results:
            if resolution.intent == QueryIntent.MORE_RESULTS:
                print_and_record_answer(
                    conversation,
                    "현재 저장된 공지 중 추가 결과가 없습니다.",
                    context_message = "이전 검색 주제에서 추가 공지를 찾지 못함"
                )
                continue

            if query_route == QueryRoute.OPEN_NOTICE_SEARCH:
                answer = (
                    "현재 신청 가능한 공지를 확인하지 못했습니다. "
                    "크롤링 파이프라인의 notices.deadline 적재 상태를  "
                    "확인해주세요."
                )
                print_and_record_answer(
                    conversation,
                    answer,
                    context_message = "현재 신청 가능한 공지를 찾지 못함"
                )
                continue
            print_and_record_answer(
                conversation,
                "관련 공지를 찾지 못했습니다.",
                context_message = "현재 질문과 관련된 공지를 찾지 못함"
            )
            print_search_failure(search_results)
            continue

        displayed_results = relevant_results[:MAX_RESULT_CHOICES]
        conversation.record_results(
            resolution,
            displayed_results,
            answer_question=resolved_question,
        )

        if should_answer_without_selection(query_route):
            direct_results = displayed_results

            if search_source == "chunks":
                try:
                    direct_results = [
                        hydrate_result_notice(
                            result,
                            repository=notice_repository,
                        )
                        for result in displayed_results
                    ]
                except NoticeRepositoryError as error:
                    print(f"공지 전체 본문을 불러오지 못했습니다: {error}")

            conversation.candidate_results = direct_results
            conversation.active_result = direct_results[0]
            answer = generate_answer(
                question=question,
                resolved_question=resolved_question,
                relevant_results=direct_results,
                answer_mode="focused",
            )
            notice = direct_results[0]["notice"]
            print_and_record_answer(
                conversation,
                answer,
                context_message=(
                    f"'{notice.get('title') or '제목 없음'}' 공지에 대해 바로 답변함"
                ),
            )
            continue

        if len(displayed_results) > 1:
            answer = create_result_selection_answer(displayed_results)
            
            titles = [
                result["notice"].get("title") or "제목 없음"
                for result in displayed_results
            ]
            print_and_record_answer(
                conversation,
                answer,
                context_message=(
                    "관련 공지 목록을 제시함: "
                    + " / ".join(titles)
                ),
            )
            continue

        active_result = displayed_results[0]

        if search_source == "chunks":
            try:
                active_result = hydrate_result_notice(
                    active_result,
                    repository=notice_repository,
                )
            except NoticeRepositoryError as error:
                print(f"공지 전체 본문을 불러오지 못했습니다: {error}")

        conversation.active_result = active_result
        notice = active_result.get("notice", {})

        answer = generate_answer(
            question=question,
            resolved_question=resolved_question,
            relevant_results=[active_result],
            answer_mode="focused",
        )
        
        print_and_record_answer(
            conversation,
            answer,
            context_message=(
                f"'{notice.get('title') or '제목 없음'}' 공지에 대해 답변함"
            ),
        )

if __name__ == "__main__":
    main()
