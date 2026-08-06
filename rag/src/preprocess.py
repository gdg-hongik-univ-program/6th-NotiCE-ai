from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

try:
    from kiwipiepy import Kiwi
except ImportError:  # pragma: no cover - fallback is tested without the package
    Kiwi = None


SPACING_REPLACEMENTS = {
    "인턴공지": "인턴 공지",
    "장학금공지": "장학금 공지",
    "졸업공지": "졸업 공지",
    "채용공지": "채용 공지",
    "공지알려줘": "공지 알려줘",
    "공지있어": "공지 있어",
    "공지없어": "공지 없어",
    "신청기간": "신청 기간",
    "신청방법": "신청 방법",
    "지원방법": "지원 방법",
    "제출서류": "제출 서류",
    "시험장소": "시험 장소",
    "중간장소": "중간 장소",
    "기말장소": "기말 장소",
    "중간시험": "중간 시험",
    "기말시험": "기말 시험",
    "알고리즘중간": "알고리즘 중간",
    "알고리즘기말": "알고리즘 기말",
    "중간어디": "중간 어디",
    "기말어디": "기말 어디",
    "시험어디": "시험 어디",
    "어디서봄": "어디서 봄",
    "어디서봐": "어디서 봐",
    "다른건": "다른 건",
}

STOPWORDS = {
    "공지",
    "관련",
    "알려줘",
    "알려주세요",
    "알리",
    "뭐야",
    "뭔가요",
    "뭐",
    "무엇",
    "무슨",
    "언제",
    "언제야",
    "언제냐",
    "언제까지",
    "언제까지야",
    "언제인가요",
    "어디",
    "어디야",
    "어딘가요",
    "어떻게",
    "있어",
    "있나요",
    "해줘",
    "해주세요",
    "나는",
    "제가",
}

KOREAN_SUFFIXES = (
    "까지",
    "부터",
    "에서",
    "에게",
    "으로",
    "하고",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "로",
    "와",
    "과",
    "도",
    "만",
    "요",
)


@dataclass(frozen=True)
class ResolvedAlias:
    alias: str
    meaning: str


@dataclass(frozen=True)
class ProcessedQuery:
    raw: str
    normalized: str
    keywords: tuple[str, ...]
    resolved_aliases: tuple[ResolvedAlias, ...] = ()


@lru_cache(maxsize=1)
def _get_kiwi() -> Any | None:
    if Kiwi is None:
        return None

    return Kiwi()


def strip_korean_suffix(token: str) -> str:
    """질문 키워드 끝에 붙은 대표적인 한국어 조사를 제거합니다."""
    for suffix in KOREAN_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]

    return token


class QueryPreprocessor:
    """검색 전에 사용자 질문을 정규화하고 핵심 키워드를 추출합니다."""

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        self.aliases = dict(aliases or {})
        sorted_aliases = sorted(self.aliases, key=len, reverse=True)
        self._alias_pattern = (
            re.compile(
                r"(?<![0-9A-Za-z가-힣])(?:"
                + "|".join(re.escape(alias) for alias in sorted_aliases)
                + r")"
            )
            if sorted_aliases
            else None
        )

    def resolve_aliases(
        self,
        question: str,
    ) -> tuple[str, tuple[ResolvedAlias, ...]]:
        normalized = " ".join(question.strip().split())
        resolved_aliases = []
        seen_aliases = set()

        def replace_alias(match: re.Match[str]) -> str:
            alias = match.group(0)
            meaning = self.aliases[alias]

            if alias not in seen_aliases:
                resolved_aliases.append(
                    ResolvedAlias(alias=alias, meaning=meaning)
                )
                seen_aliases.add(alias)

            return meaning

        if self._alias_pattern is not None:
            normalized = self._alias_pattern.sub(replace_alias, normalized)

        for before, after in sorted(
            SPACING_REPLACEMENTS.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            normalized = normalized.replace(before, after)

        return (
            " ".join(normalized.split()),
            tuple(resolved_aliases),
        )

    def normalize(self, question: str) -> str:
        normalized, _ = self.resolve_aliases(question)
        return normalized

    def _extract_keywords_from_normalized(self, normalized: str) -> list[str]:
        kiwi = _get_kiwi()

        if kiwi is None:
            tokens = re.findall(r"[0-9a-zA-Z가-힣]+", normalized.lower())
        else:
            tokens = [
                token.form.lower()
                for token in kiwi.tokenize(normalized)
                if token.tag.startswith(("N", "V"))
                or token.tag in {"SL", "SH", "SN"}
            ]

        keywords = []

        for token in tokens:
            keyword = strip_korean_suffix(token)

            if keyword in STOPWORDS:
                continue

            if len(keyword) < 2 and not keyword.isdigit():
                continue

            if keyword not in keywords:
                keywords.append(keyword)

        return keywords

    def extract_keywords(self, question: str) -> list[str]:
        normalized = self.normalize(question)
        return self._extract_keywords_from_normalized(normalized)

    def process(self, question: str) -> ProcessedQuery:
        normalized, resolved_aliases = self.resolve_aliases(question)
        return ProcessedQuery(
            raw=question,
            normalized=normalized,
            keywords=tuple(self._extract_keywords_from_normalized(normalized)),
            resolved_aliases=resolved_aliases,
        )


DEFAULT_PREPROCESSOR = QueryPreprocessor()
