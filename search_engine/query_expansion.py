from __future__ import annotations

import re
from typing import Dict, Tuple


KOREAN_FASHION_LEXICON: Dict[str, Tuple[str, ...]] = {
    "블랙": ("black",),
    "검정": ("black",),
    "화이트": ("white",),
    "흰색": ("white",),
    "아이보리": ("ivory", "cream"),
    "베이지": ("beige",),
    "브라운": ("brown",),
    "카멜": ("camel", "tan"),
    "네이비": ("navy", "dark blue"),
    "블루": ("blue",),
    "그린": ("green",),
    "카키": ("khaki", "olive"),
    "레드": ("red",),
    "핑크": ("pink",),
    "실버": ("silver", "metallic"),
    "골드": ("gold", "metallic"),
    "광택": ("glossy", "shiny", "lustrous"),
    "유광": ("glossy", "shiny"),
    "무광": ("matte",),
    "미니멀": ("minimal", "minimalist", "clean"),
    "심플": ("simple", "minimal"),
    "가죽": ("leather", "faux leather"),
    "레더": ("leather", "faux leather"),
    "라이더": ("biker", "rider"),
    "피방": ("pc cafe", "gaming", "comfortable casual", "hoodie", "sweatshirt", "joggers", "sweatpants", "sneakers"),
    "피씨방": ("pc cafe", "gaming", "comfortable casual", "hoodie", "sweatshirt", "joggers", "sweatpants", "sneakers"),
    "피시방": ("pc cafe", "gaming", "comfortable casual", "hoodie", "sweatshirt", "joggers", "sweatpants", "sneakers"),
    "PC방": ("pc cafe", "gaming", "comfortable casual", "hoodie", "sweatshirt", "joggers", "sweatpants", "sneakers"),
    "pc방": ("pc cafe", "gaming", "comfortable casual", "hoodie", "sweatshirt", "joggers", "sweatpants", "sneakers"),
    "겜방": ("gaming", "comfortable casual", "hoodie", "sweatshirt", "joggers", "sweatpants", "sneakers"),
    "자켓": ("jacket", "outerwear"),
    "재킷": ("jacket", "outerwear"),
    "아우터": ("outerwear", "jacket", "coat"),
    "코트": ("coat", "outerwear"),
    "패딩": ("puffer", "down jacket"),
    "후드": ("hoodie", "hooded sweatshirt", "sweatshirt"),
    "후디": ("hoodie", "hooded sweatshirt", "sweatshirt"),
    "셔츠": ("shirt", "top"),
    "블라우스": ("blouse", "top"),
    "니트": ("knit", "sweater"),
    "스웨터": ("sweater", "knit"),
    "티셔츠": ("t-shirt", "tee", "top"),
    "반팔": ("short sleeve", "tee", "top"),
    "긴팔": ("long sleeve", "top"),
    "스커트": ("skirt",),
    "치마": ("skirt",),
    "미디": ("midi",),
    "미니": ("mini",),
    "플리츠": ("pleated", "pleats"),
    "원피스": ("dress", "one-piece"),
    "드레스": ("dress",),
    "팬츠": ("pants", "trousers"),
    "바지": ("pants", "trousers"),
    "츄리닝": ("sweatpants", "joggers", "track pants", "training pants", "sport casual"),
    "추리닝": ("sweatpants", "joggers", "track pants", "training pants", "sport casual"),
    "트레이닝복": ("sweatpants", "joggers", "track pants", "training pants", "sport casual"),
    "조거": ("joggers", "sweatpants", "relaxed trousers"),
    "데님": ("denim", "jeans"),
    "청바지": ("jeans", "denim"),
    "편한": ("comfortable", "casual", "relaxed fit", "basic"),
    "편하게": ("comfortable", "casual", "relaxed fit", "basic"),
    "꾸안꾸": ("comfortable casual", "minimal", "daily", "relaxed fit"),
    "데이트": ("date outfit", "smart casual", "shirt", "blouse", "knit", "cardigan", "trousers", "dress", "loafers"),
    "소개팅": ("first date", "smart casual", "shirt", "blouse", "knit", "cardigan", "trousers", "dress", "loafers"),
    "여자친구": ("date outfit", "smart casual", "clean", "shirt", "knit", "cardigan", "trousers", "loafers"),
    "여친": ("date outfit", "smart casual", "clean", "shirt", "knit", "cardigan", "trousers", "loafers"),
    "남자친구": ("date outfit", "smart casual", "clean", "shirt", "knit", "cardigan", "trousers", "loafers"),
    "남친": ("date outfit", "smart casual", "clean", "shirt", "knit", "cardigan", "trousers", "loafers"),
    "도서관": ("library", "study", "clean", "minimal", "cardigan", "knit", "shirt", "trousers", "jeans", "sneakers"),
    "스터디": ("study", "clean", "minimal", "cardigan", "knit", "shirt", "trousers", "jeans", "sneakers"),
    "공부": ("study", "comfortable casual", "clean", "minimal", "cardigan", "knit", "shirt", "trousers"),
    "팀플": ("study", "clean", "minimal", "shirt", "cardigan", "trousers", "sneakers"),
    "카페": ("cafe", "brunch", "clean casual", "cardigan", "shirt", "blouse", "knit", "jeans", "trousers"),
    "브런치": ("brunch", "clean casual", "cardigan", "shirt", "blouse", "knit", "jeans", "trousers"),
    "친구만날": ("casual meetup", "clean casual", "cardigan", "shirt", "knit", "jeans", "sneakers"),
    "출근": ("office", "workwear", "shirt", "blouse", "blazer", "jacket", "trousers", "slacks", "clean", "minimal"),
    "회사": ("office", "workwear", "shirt", "blouse", "blazer", "jacket", "trousers", "slacks", "clean", "minimal"),
    "오피스": ("office", "workwear", "shirt", "blouse", "blazer", "jacket", "trousers", "slacks", "clean", "minimal"),
    "면접": ("interview", "formal", "shirt", "blouse", "blazer", "jacket", "trousers", "slacks", "clean"),
    "비오는": ("rainy", "jacket", "coat", "parka", "hooded", "dark", "trousers", "boots", "sneakers", "practical"),
    "비 오는": ("rainy", "jacket", "coat", "parka", "hooded", "dark", "trousers", "boots", "sneakers", "practical"),
    "장마": ("rainy", "jacket", "coat", "parka", "hooded", "dark", "trousers", "boots", "sneakers", "practical"),
    "여행": ("travel", "comfortable casual", "relaxed fit", "hoodie", "sweatshirt", "joggers", "trousers", "t-shirt", "sneakers", "jacket"),
    "공항": ("airport", "travel", "comfortable casual", "relaxed fit", "hoodie", "sweatshirt", "joggers", "trousers", "sneakers"),
    "운동": ("sportswear", "training", "leggings", "t-shirt", "tank top", "sneakers", "joggers", "sweatshirt"),
    "헬스": ("gym", "sportswear", "training", "leggings", "t-shirt", "tank top", "sneakers", "joggers"),
    "러닝": ("running", "sportswear", "training", "leggings", "t-shirt", "sneakers", "joggers"),
    "산책": ("walking", "comfortable casual", "sneakers", "t-shirt", "sweatshirt", "trousers", "jacket"),
    "실루엣": ("silhouette", "shape"),
    "포인트": ("accent", "detail"),
    "룩": ("look", "outfit", "style"),
}

ENGLISH_QUERY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "jacket": ("outerwear", "coat", "blouson"),
    "coat": ("outerwear",),
    "dress": ("one-piece",),
    "skirt": ("midi", "mini", "pleated"),
    "shirt": ("top", "blouse"),
    "tee": ("t-shirt", "top"),
    "t-shirt": ("tee", "top"),
    "sweater": ("knit",),
    "knit": ("sweater",),
    "hoodie": ("hooded sweatshirt", "sweatshirt", "casual"),
    "sweatshirt": ("hoodie", "casual"),
    "sweatpants": ("joggers", "track pants", "training pants"),
    "joggers": ("sweatpants", "relaxed trousers"),
    "pants": ("trousers",),
    "trousers": ("pants",),
    "jeans": ("denim",),
    "leather": ("faux leather",),
    "glossy": ("shiny", "lustrous"),
    "minimal": ("minimalist", "clean"),
    "silver": ("metallic",),
}


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in text)


def expand_fashion_query(text: str) -> str:
    query = " ".join((text or "").strip().split())
    if not query:
        return ""

    additions: list[str] = []
    lowered = query.lower()

    for phrase, mapped_terms in sorted(KOREAN_FASHION_LEXICON.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase in query:
            additions.extend(mapped_terms)

    for token in re.findall(r"[a-z0-9-]+", lowered):
        additions.extend(ENGLISH_QUERY_SYNONYMS.get(token, ()))

    if contains_hangul(query):
        additions.extend(["fashion", "apparel", "product"])

    unique_terms: list[str] = []
    seen: set[str] = set()
    for token in additions:
        normalized = token.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_terms.append(token)

    if not unique_terms:
        return query
    return f"{query}. Keywords: {' '.join(unique_terms)}"
