export type RecommendationItem = {
  id: number;
  title: string;
  brand: string;
  price: string;
  reason: string;
  rank: number;
  score: number;
  accent: string;
  imageUrl?: string;
};

export type RecommendationStage = {
  label: string;
  value: string;
};

export type RecommendationBundle = {
  items: RecommendationItem[];
  totalLatency: string;
  stages: RecommendationStage[];
  persona: string;
};

export type SearchItem = {
  id: number;
  title: string;
  brand: string;
  price: string;
  similarity: number;
  searchType: string;
  responseTime: string;
  summary: string;
  accent: string;
  imageUrl?: string;
};

const recommendationFallbackPalette = [
  "linear-gradient(135deg, #3f284f 0%, #121520 100%)",
  "linear-gradient(135deg, #65513a 0%, #171a23 100%)",
  "linear-gradient(135deg, #0a5960 0%, #13161d 100%)",
  "linear-gradient(135deg, #28465d 0%, #141720 100%)",
  "linear-gradient(135deg, #5a3f3f 0%, #15161f 100%)",
];

const searchFallbackPalette = [
  "linear-gradient(135deg, #35244d 0%, #161822 100%)",
  "linear-gradient(135deg, #84553a 0%, #1a1d26 100%)",
  "linear-gradient(135deg, #04545f 0%, #131620 100%)",
  "linear-gradient(135deg, #26314c 0%, #11151d 100%)",
  "linear-gradient(135deg, #5b402f 0%, #181720 100%)",
];

type ApiRecommendationItem = {
  id?: number;
  item_id?: number | string;
  product_id?: number | string;
  title?: string;
  name?: string;
  brand?: string;
  price?: string | number;
  reason?: string;
  rank?: number;
  score?: number;
  accent?: string;
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
};

type ApiRecommendationResponse = {
  items?: ApiRecommendationItem[];
  recommendations?: ApiRecommendationItem[];
  totalLatency?: string;
  total_ms?: number;
  latency_ms?: number;
  candidate_ms?: number;
  ranking_ms?: number;
  reranking_ms?: number;
  pipeline_latency?: {
    candidate_ms?: number;
    ranking_ms?: number;
    reranking_ms?: number;
    total_ms?: number;
  };
  persona?: string;
};

type ApiSearchItem = {
  id?: number;
  item_id?: number | string;
  product_id?: number | string;
  title?: string;
  name?: string;
  brand?: string;
  price?: string | number;
  summary?: string;
  description?: string;
  score?: number;
  similarity?: number;
  accent?: string;
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
};

type ApiSearchResponse =
  | {
      items?: ApiSearchItem[];
      results?: ApiSearchItem[];
      latency_ms?: number;
      total_ms?: number;
      mode?: string;
    }
  | ApiSearchItem[];

function getApiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL ?? "").trim();
}

function buildApiUrl(path: string): string {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    return path;
  }

  return `${baseUrl.replace(/\/$/, "")}${path}`;
}

function toCurrencyLabel(value: string | number | undefined): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${value.toLocaleString("ko-KR")}원`;
  }

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  return "가격 정보 없음";
}

function toLatencyLabel(value: number | string | undefined, fallback: string): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${Math.round(value)}ms`;
  }

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  return fallback;
}

function toNumericId(...values: Array<number | string | undefined>): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return Date.now() + Math.floor(Math.random() * 1000);
}

function toImageUrl(item: {
  image_url?: string;
  imageUrl?: string;
  img_url?: string;
  thumbnail_url?: string;
  thumbnailUrl?: string;
}): string | undefined {
  return item.image_url ?? item.imageUrl ?? item.img_url ?? item.thumbnail_url ?? item.thumbnailUrl;
}

function normalizeRecommendationBundle(
  payload: ApiRecommendationResponse,
  topN: number,
): RecommendationBundle {
  const rawItems = payload.items ?? payload.recommendations ?? [];
  const pipelineLatency = payload.pipeline_latency;
  const totalLatencyMs = payload.total_ms ?? payload.latency_ms ?? pipelineLatency?.total_ms;
  const candidateMs = payload.candidate_ms ?? pipelineLatency?.candidate_ms;
  const rankingMs = payload.ranking_ms ?? pipelineLatency?.ranking_ms;
  const rerankingMs = payload.reranking_ms ?? pipelineLatency?.reranking_ms;
  const items = rawItems.slice(0, topN).map((item, index) => ({
    id: toNumericId(item.id, item.item_id, item.product_id),
    title: item.title ?? item.name ?? `Recommendation ${index + 1}`,
    brand: item.brand ?? "Unknown Brand",
    price: toCurrencyLabel(item.price),
    reason: item.reason ?? "추천 이유 정보가 아직 제공되지 않았습니다.",
    rank: item.rank ?? index + 1,
    score: typeof item.score === "number" ? item.score : Math.max(0.5, 0.95 - index * 0.04),
    accent: item.accent ?? recommendationFallbackPalette[index % recommendationFallbackPalette.length],
    imageUrl: toImageUrl(item),
  }));

  return {
    items,
    totalLatency: toLatencyLabel(totalLatencyMs ?? payload.totalLatency, "0ms"),
    persona: payload.persona ?? "개인화 추천",
    stages: [
      candidateMs !== undefined ? { label: "Candidate", value: toLatencyLabel(candidateMs, "0ms") } : null,
      rankingMs !== undefined ? { label: "Ranking", value: toLatencyLabel(rankingMs, "0ms") } : null,
      rerankingMs !== undefined ? { label: "Reranking", value: toLatencyLabel(rerankingMs, "0ms") } : null,
    ].filter((stage): stage is RecommendationStage => stage !== null),
  };
}

function normalizeSearchItems(
  payload: ApiSearchResponse,
  fallbackMode: string,
): { items: SearchItem[]; responseTime: string } {
  const rawItems = Array.isArray(payload) ? payload : payload.items ?? payload.results ?? [];
  const responseTime = Array.isArray(payload)
    ? "0ms"
    : toLatencyLabel(payload.latency_ms ?? payload.total_ms, "0ms");
  const searchType =
    fallbackMode === "multimodal" ? "텍스트 + 이미지" : fallbackMode === "image" ? "이미지" : "텍스트";

  return {
    responseTime,
    items: rawItems.map((item, index) => ({
      id: toNumericId(item.id, item.item_id, item.product_id),
      title: item.title ?? item.name ?? `Search Result ${index + 1}`,
      brand: item.brand ?? "Unknown Brand",
      price: toCurrencyLabel(item.price),
      similarity:
        typeof item.similarity === "number"
          ? item.similarity
          : typeof item.score === "number"
            ? item.score
            : Math.max(0.5, 0.95 - index * 0.05),
      searchType,
      responseTime,
      summary: item.summary ?? item.description ?? "검색 결과 설명이 제공되지 않았습니다.",
      accent: item.accent ?? searchFallbackPalette[index % searchFallbackPalette.length],
      imageUrl: toImageUrl(item),
    })),
  };
}

export async function fetchRecommendations(
  userId: string,
  topN: number,
  _seed: number,
  _personaHint?: string,
): Promise<RecommendationBundle> {
  const url = new URL(buildApiUrl("/api/recommend"), window.location.origin);
  url.searchParams.set("user_id", userId);
  url.searchParams.set("top_n", String(topN));

  const response = await fetch(url.toString(), {
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Recommendation API failed with ${response.status}`);
  }

  const payload = (await response.json()) as ApiRecommendationResponse;
  return normalizeRecommendationBundle(payload, topN);
}

export async function fetchSearchResults(params: {
  query: string;
  imageBase64?: string | null;
  topK: number;
  mode: "text" | "image" | "multimodal";
}): Promise<{ items: SearchItem[]; responseTime: string }> {
  const response = await fetch(buildApiUrl("/api/search"), {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: params.query,
      image_base64: params.imageBase64 ?? null,
      top_k: params.topK,
    }),
  });

  if (!response.ok) {
    throw new Error(`Search API failed with ${response.status}`);
  }

  const payload = (await response.json()) as ApiSearchResponse;
  return normalizeSearchItems(payload, params.mode);
}

export async function sendInteractionEvent(input: {
  userId: string;
  itemId: number | string;
  eventType?: "click" | "purchase";
  category?: string;
}): Promise<void> {
  await fetch(buildApiUrl("/api/events"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: input.userId,
      item_id: String(input.itemId),
      event_type: input.eventType ?? "click",
      category: input.category ?? null,
    }),
  });
}
