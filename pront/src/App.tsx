import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  BudgetSetBundle,
  OnboardingPersonaScores,
  RecommendationBundle,
  fetchBudgetSets,
  fetchOnboardingPersonaScores,
  fetchRecommendations,
  fetchSearchResults,
  selectOnboardingPersona,
  sendInteractionEvent,
} from "./api";

type SearchMode = "text" | "image" | "multimodal";

type SearchResult = {
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

type UploadedImage = {
  name: string;
  sizeLabel: string;
  base64: string;
};

type PersonaOption = {
  key: string;
  name: string;
  title: string;
  summary: string;
  traits: string[];
};

const baseResults: Record<SearchMode, SearchResult[]> = {
    text: [
      {
        id: 1,
      title: "Urban Edge Rider Jacket",
      brand: "Mode Atelier",
      price: "89,000원",
      similarity: 0.94,
      searchType: "텍스트 검색",
      responseTime: "128ms",
      summary: "질감과 아우터 무드를 반영한 텍스트 기반 탐색 결과입니다.",
        accent: "linear-gradient(135deg, #35244d 0%, #161822 100%)",
      },
      {
        id: 2,
      title: "Minimal Zip Blouson",
      brand: "Noir Form",
      price: "42,000원",
      similarity: 0.9,
      searchType: "텍스트 검색",
      responseTime: "128ms",
      summary: "미니멀한 블랙 계열 취향과 잘 맞는 후보입니다.",
        accent: "linear-gradient(135deg, #84553a 0%, #1a1d26 100%)",
      },
    ],
    image: [
    {
      id: 3,
      title: "Silver Trim Moto Crop",
      brand: "Avenue N",
      price: "76,000원",
      similarity: 0.96,
      searchType: "이미지 검색",
      responseTime: "173ms",
      summary: "업로드 이미지의 실루엣과 광택감을 반영한 결과입니다.",
        accent: "linear-gradient(135deg, #26314c 0%, #11151d 100%)",
      },
      {
        id: 4,
      title: "Gloss Rider Short",
      brand: "Studio Hex",
      price: "58,000원",
      similarity: 0.91,
      searchType: "이미지 검색",
      responseTime: "173ms",
      summary: "유사한 재질과 길이감을 우선 반영한 후보입니다.",
        accent: "linear-gradient(135deg, #5b402f 0%, #181720 100%)",
      },
    ],
    multimodal: [
    {
      id: 5,
      title: "Chrome Detail Urban Rider",
      brand: "Modu Lab",
      price: "98,000원",
      similarity: 0.98,
      searchType: "텍스트 + 이미지",
      responseTime: "214ms",
      summary: "텍스트 설명과 이미지 특징이 함께 반영된 상위 결과입니다.",
        accent: "linear-gradient(135deg, #42294f 0%, #11131c 100%)",
      },
      {
        id: 6,
      title: "Blackline Cropped Moto",
      brand: "Noir Craft",
      price: "71,000원",
      similarity: 0.94,
      searchType: "텍스트 + 이미지",
      responseTime: "214ms",
      summary: "질감과 스타일 키워드가 함께 맞아 높은 점수를 받은 결과입니다.",
        accent: "linear-gradient(135deg, #72412f 0%, #171923 100%)",
      },
    ],
  };

const suggestions = [
  "미니멀한 블랙 아우터",
  "실버 디테일이 있는 스트리트 룩",
  "출근용으로 입을 수 있는 자켓",
];

const personaOptions: PersonaOption[] = [
  {
    key: "trendsetter",
    name: "트렌드세터형",
    title: "새로운 스타일을 빠르게 시도해요",
    summary: "유행과 변화에 민감하고 다양한 룩을 탐색하는 성향입니다.",
    traits: ["유행 민감", "실험적", "빠른 반응"],
  },
  {
    key: "practical",
    name: "실용주의형",
    title: "착용감과 활용도를 중요하게 봐요",
    summary: "오래 입기 좋고 다양한 상황에 맞는 아이템을 선호합니다.",
    traits: ["실용성", "기본 아이템", "활용도"],
  },
  {
    key: "value",
    name: "가성비추구형",
    title: "가격 대비 만족도를 중요하게 봐요",
    summary: "할인과 가격 메리트를 함께 고려하는 성향입니다.",
    traits: ["가격 민감", "할인 선호", "비교 구매"],
  },
  {
    key: "brand_loyal",
    name: "브랜드충성형",
    title: "익숙한 브랜드를 꾸준히 선택해요",
    summary: "기존 만족 경험이 있는 브랜드와 카테고리를 반복 탐색합니다.",
    traits: ["브랜드 선호", "재구매", "안정적 취향"],
  },
  {
    key: "impulse",
    name: "충동구매형",
    title: "마음에 들면 빠르게 결정해요",
    summary: "즉각적인 매력과 인상적인 디테일에 민감하게 반응합니다.",
    traits: ["빠른 결정", "즉흥성", "시각 반응"],
  },
  {
    key: "careful",
    name: "신중탐색형",
    title: "여러 옵션을 오래 비교해요",
    summary: "리뷰, 소재, 가격을 충분히 비교한 뒤 결정하는 성향입니다.",
    traits: ["비교 탐색", "정보 수집", "신중한 결정"],
  },
  {
    key: "repeat_stable",
    name: "반복구매형",
    title: "비슷한 상품을 꾸준히 다시 찾아요",
    summary: "익숙한 카테고리와 검증된 아이템을 반복 구매하는 성향입니다.",
    traits: ["재구매", "안정성", "반복 선택"],
  },
  {
    key: "color_focus",
    name: "색상집중형",
    title: "선호하는 색감을 중심으로 봐요",
    summary: "특정 컬러 계열을 우선해서 탐색하는 경향이 강합니다.",
    traits: ["컬러 우선", "톤 선호", "시각 취향"],
  },
  {
    key: "category_focus",
    name: "카테고리집중형",
    title: "원하는 카테고리를 깊게 파고들어요",
    summary: "특정 카테고리 안에서 다양한 옵션을 오래 비교합니다.",
    traits: ["카테고리 몰입", "깊은 비교", "명확한 관심사"],
  },
];

const onboardingStyleOptions = ["casual", "minimal", "street", "sporty", "feminine", "classic"];

const emptyBundle: RecommendationBundle = {
  items: [],
  totalLatency: "0ms",
  stages: [],
  persona: "미분류",
};

const emptyBudgetSetBundle: BudgetSetBundle = {
  budget: 0,
  setCount: 0,
  sets: [],
};

function ResultVisual({
  imageUrl,
  title,
  accent,
}: {
  imageUrl?: string;
  title: string;
  accent: string;
}) {
  return (
    <div className="result-visual" style={{ background: accent }}>
      {imageUrl ? <img className="result-image" src={imageUrl} alt={title} loading="lazy" /> : null}
    </div>
  );
}

function App() {
  const [isRegistered, setIsRegistered] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [selectedOnboardingPersona, setSelectedOnboardingPersona] = useState("trendsetter");
  const [query, setQuery] = useState("광택감 있는 블랙 아우터와 실버 포인트 자켓");
  const [userId, setUserId] = useState("user_1024");
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [searchMode, setSearchMode] = useState<SearchMode>("multimodal");
  const [results, setResults] = useState<SearchResult[]>(baseResults.multimodal);
  const [activeLatency, setActiveLatency] = useState("214ms");
  const [isSearching, setIsSearching] = useState(false);
  const [lastSearchedAt, setLastSearchedAt] = useState("방금 전");

  const [topN, setTopN] = useState(5);
  const [recommendationWeight, setRecommendationWeight] = useState(70);
  const [budget, setBudget] = useState("200000");
  const [activeBundle, setActiveBundle] = useState<RecommendationBundle>(emptyBundle);
  const [recommendationSeed, setRecommendationSeed] = useState(0);
  const [isRefreshingRecommendations, setIsRefreshingRecommendations] = useState(false);
  const [recommendationError, setRecommendationError] = useState<string | null>(null);
  const [budgetSets, setBudgetSets] = useState<BudgetSetBundle>(emptyBudgetSetBundle);
  const [isLoadingBudgetSets, setIsLoadingBudgetSets] = useState(false);
  const [budgetSetError, setBudgetSetError] = useState<string | null>(null);

  const [onboardingDescription, setOnboardingDescription] = useState("");
  const [selectedStyles, setSelectedStyles] = useState<string[]>(["minimal"]);
  const [personaScores, setPersonaScores] = useState<OnboardingPersonaScores>({});
  const [isAnalyzingOnboarding, setIsAnalyzingOnboarding] = useState(false);
  const [isSubmittingPersona, setIsSubmittingPersona] = useState(false);
  const [onboardingError, setOnboardingError] = useState<string | null>(null);
  const isManagingHistoryRef = useRef(false);

  const popularityWeight = 100 - recommendationWeight;
  const pricePriorityWeight = recommendationWeight / 50;
  const popularityPriorityWeight = popularityWeight / 50;
  const budgetLabel = `${Number(budget || 0).toLocaleString("ko-KR")}원`;

  const helperMessage = useMemo(() => {
    if (searchMode === "text") {
      return "텍스트 질의만으로 유사 상품을 찾습니다.";
    }
    if (searchMode === "image") {
      return "업로드 이미지 특징을 기반으로 시각적으로 비슷한 상품을 찾습니다.";
    }
    return "텍스트와 이미지 신호를 함께 반영해 더 강한 후보를 우선 정렬합니다.";
  }, [searchMode]);

  useEffect(() => {
    setPersonaScores({});
    setOnboardingError(null);
  }, [onboardingDescription, selectedStyles]);

  useEffect(() => {
    if (!isRegistered || showOnboarding) {
      return;
    }

    let cancelled = false;

    const loadRecommendations = async () => {
      setIsRefreshingRecommendations(true);
      setRecommendationError(null);

      try {
        const bundle = await fetchRecommendations(userId.trim() || "anonymous", topN, recommendationSeed, {
          personaHint: selectedOnboardingPersona,
          priceWeight: pricePriorityWeight,
          popularityWeight: popularityPriorityWeight,
          includeReasons: true,
        });

        if (!cancelled) {
          setActiveBundle(bundle);
        }
      } catch {
        if (!cancelled) {
          setRecommendationError("추천 결과를 불러오지 못했습니다.");
          setActiveBundle(emptyBundle);
        }
      } finally {
        if (!cancelled) {
          setIsRefreshingRecommendations(false);
        }
      }
    };

    void loadRecommendations();

    return () => {
      cancelled = true;
    };
  }, [
    isRegistered,
    showOnboarding,
    userId,
    topN,
    recommendationSeed,
    selectedOnboardingPersona,
    recommendationWeight,
    popularityWeight,
    pricePriorityWeight,
    popularityPriorityWeight,
  ]);

  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      const nextView = event.state?.view;

      if (showOnboarding && nextView !== "onboarding") {
        isManagingHistoryRef.current = true;
        setShowOnboarding(false);
        isManagingHistoryRef.current = false;
      }
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [showOnboarding]);

  useEffect(() => {
    if (!isRegistered || isManagingHistoryRef.current) {
      return;
    }

    const currentView = window.history.state?.view;

    if (showOnboarding && currentView !== "onboarding") {
      window.history.pushState({ view: "onboarding" }, "");
      return;
    }

    if (!showOnboarding && currentView === "onboarding") {
      window.history.replaceState({ view: "main" }, "");
    }
  }, [isRegistered, showOnboarding]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setUploadedImage(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const [, base64 = ""] = result.split(",");
      const sizeInMb = file.size / (1024 * 1024);

      setUploadedImage({
        name: file.name,
        sizeLabel: `${sizeInMb.toFixed(2)}MB`,
        base64,
      });

      setSearchMode((currentMode) => (currentMode === "text" ? "multimodal" : currentMode));
    };

    reader.readAsDataURL(file);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmedQuery = query.trim();
    const nextMode: SearchMode =
      trimmedQuery && uploadedImage ? "multimodal" : uploadedImage ? "image" : "text";

    setSearchMode(nextMode);
    setIsSearching(true);

    try {
      const response = await fetchSearchResults({
        query: trimmedQuery,
        imageBase64: uploadedImage?.base64 ?? null,
        topK: 10,
        mode: nextMode,
      });

      if (response.items.length > 0) {
        setResults(response.items);
        setActiveLatency(response.responseTime);
      } else {
        setResults(baseResults[nextMode]);
        setActiveLatency(baseResults[nextMode][0]?.responseTime ?? "128ms");
      }

      if (isRegistered && trimmedQuery) {
        try {
          await sendInteractionEvent({
            userId: userId.trim() || "anonymous",
            eventType: "search",
            queryText: trimmedQuery,
          });
          setRecommendationSeed((current) => current + 1);
        } catch {
          // Search results should remain visible even if personalization logging fails.
        }
      }
    } catch {
      setResults(baseResults[nextMode]);
      setActiveLatency(baseResults[nextMode][0]?.responseTime ?? "128ms");
    } finally {
      setLastSearchedAt(new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }));
      setIsSearching(false);
    }
  };

  const handleRecommendationClick = (itemId: number) => {
    void sendInteractionEvent({
      userId: userId.trim() || "anonymous",
      itemId,
      eventType: "click",
    });
  };

  const applySuggestion = (value: string) => {
    setQuery(value);
  };

  const toggleStyleChoice = (style: string) => {
    setSelectedStyles((current) =>
      current.includes(style) ? current.filter((value) => value !== style) : [...current, style],
    );
  };

  const handleSignUp = () => {
    if (!userId.trim()) {
      return;
    }

    setIsRegistered(true);
    setShowOnboarding(true);
  };

  const runOnboardingAnalysis = async () => {
    if (!userId.trim() || !onboardingDescription.trim()) {
      setOnboardingError("사용자 ID와 취향 설명을 입력해 주세요.");
      return;
    }

    setIsAnalyzingOnboarding(true);
    setOnboardingError(null);

    try {
      const scores = await fetchOnboardingPersonaScores({
        userId: userId.trim(),
        description: onboardingDescription.trim(),
        styleChoices: selectedStyles,
        budgetRange: null,
      });

      setPersonaScores(scores);
      const topPersona = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0];
      if (topPersona) {
        setSelectedOnboardingPersona(topPersona);
      }
    } catch {
      setOnboardingError("페르소나 분석에 실패했습니다. 백엔드 설정을 확인해 주세요.");
    } finally {
      setIsAnalyzingOnboarding(false);
    }
  };

  const refreshRecommendations = () => {
    setRecommendationSeed((current) => current + 1);
  };

  const loadBudgetSets = async () => {
    const parsedBudget = Number(budget);
    if (!userId.trim() || !Number.isFinite(parsedBudget) || parsedBudget <= 0) {
      setBudgetSetError("유효한 사용자 ID와 예산을 입력해 주세요.");
      return;
    }

    setIsLoadingBudgetSets(true);
    setBudgetSetError(null);

    try {
      const bundle = await fetchBudgetSets({
        userId: userId.trim(),
        budget: parsedBudget,
        setCount: 3,
      });
      setBudgetSets(bundle);
    } catch {
      setBudgetSetError("예산 세트 추천 결과를 불러오지 못했습니다.");
      setBudgetSets(emptyBudgetSetBundle);
    } finally {
      setIsLoadingBudgetSets(false);
    }
  };

  const startWithPersona = async () => {
    setIsSubmittingPersona(true);
    setOnboardingError(null);

    try {
      await selectOnboardingPersona({
        userId: userId.trim() || "anonymous",
        persona: selectedOnboardingPersona,
      });
      setShowOnboarding(false);
      setBudgetSets(emptyBudgetSetBundle);
      setRecommendationSeed((current) => current + 1);
    } catch {
      setOnboardingError("선택한 페르소나를 저장하지 못했습니다.");
    } finally {
      setIsSubmittingPersona(false);
    }
  };

  const modeLabel =
    searchMode === "multimodal" ? "멀티모달" : searchMode === "image" ? "이미지" : "텍스트";
  const selectedPersona =
    personaOptions.find((persona) => persona.key === selectedOnboardingPersona) ?? null;
  const selectedPersonaLabel = selectedPersona?.name ?? selectedOnboardingPersona;

  if (showOnboarding) {
    return (
      <div className="app-shell onboarding-shell">
        <section className="onboarding-panel">
          <div className="onboarding-copy">
            <p className="eyebrow">Personalization Setup</p>
            <h1>먼저 취향을 알려주시면 추천을 바로 맞춰드립니다.</h1>
            <p>
              취향 설명과 스타일 선택을 바탕으로 페르소나를 분석하고, 결과를 확정하면
              개인화 추천에 즉시 반영됩니다.
            </p>
          </div>

          <div className="search-composer">
            <label className="search-box">
              <span>취향 설명</span>
              <input
                value={onboardingDescription}
                onChange={(event) => setOnboardingDescription(event.target.value)}
                placeholder="예: 미니멀한 블랙 아우터와 실용적인 출근룩을 자주 봅니다"
                aria-label="온보딩 취향 설명"
              />
            </label>

            <div className="signal-list">
              {onboardingStyleOptions.map((style) => (
                <button
                  key={style}
                  type="button"
                  className={selectedStyles.includes(style) ? "mini-button active" : "mini-button"}
                  onClick={() => toggleStyleChoice(style)}
                >
                  {style}
                </button>
              ))}
            </div>

            <div className="recommendation-toolbar">
              <button
                type="button"
                className="primary-button"
                onClick={runOnboardingAnalysis}
                disabled={isAnalyzingOnboarding}
              >
                {isAnalyzingOnboarding ? "분석 중..." : "취향 분석하기"}
              </button>
            </div>
          </div>

          {selectedPersona && Object.keys(personaScores).length > 0 ? (
            <div className="persona-grid">
              <article className="persona-option active">
                <p className="persona-name">{selectedPersona.name}</p>
                <h2>{selectedPersona.title}</h2>
                <p className="persona-summary">{selectedPersona.summary}</p>
                <strong>{personaScores[selectedPersona.key] ?? 0}%</strong>
                <div className="persona-traits">
                  {selectedPersona.traits.map((trait) => (
                    <span key={trait} className="badge">
                      {trait}
                    </span>
                  ))}
                </div>
              </article>
            </div>
          ) : null}

          <div className="onboarding-footer">
            <div className="persona-card">
              <span>현재 선택된 페르소나</span>
              <strong>{selectedPersonaLabel}</strong>
            </div>
            <button
              type="button"
              className="primary-button"
              onClick={startWithPersona}
              disabled={isSubmittingPersona || Object.keys(personaScores).length === 0}
            >
              {isSubmittingPersona ? "저장 중..." : "이 취향으로 추천 받기"}
            </button>
          </div>
          {onboardingError ? <p className="status-text">{onboardingError}</p> : null}
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ModeMosaic</p>
          <h1>Multimodal Search</h1>
        </div>
        <div className="topbar-meta">
          <span>검색 모드: {modeLabel}</span>
          <span>최근 검색: {lastSearchedAt}</span>
          <span>사용자 ID: {userId}</span>
          <span>개인화 상태: {isRegistered ? "설정 완료" : "설정 전"}</span>
          <span>선택한 취향: {selectedPersonaLabel}</span>
          <span>현재 추천 성향: {activeBundle.persona}</span>
          <span>추천 개수: Top {topN}</span>
          <span>
            개인화 {recommendationWeight} / 대중성 {popularityWeight}
          </span>
          <span>예산: {budgetLabel}</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel signup-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">User Setup</p>
              <h3>사용자 정보 설정</h3>
            </div>
          </div>
          <div className="signup-row">
            <label className="user-id-field">
              <span>User ID</span>
              <input
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                placeholder="예: user_1024"
                aria-label="회원가입 사용자 ID"
              />
            </label>
            <button
              type="button"
              className="primary-button"
              onClick={handleSignUp}
              disabled={!userId.trim() || isRegistered}
            >
              {isRegistered ? "설정 완료" : "설정 시작"}
            </button>
          </div>
          <p className="status-text signup-text">
            사용자 정보를 설정한 뒤 취향 분석을 진행하면 초기 추천에 바로 반영됩니다.
          </p>
        </section>

        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Search Experience</p>
            <h2>검색 화면</h2>
            <p className="hero-description">
              검색어와 이미지를 함께 입력하면 상황에 맞는 검색 방식을 자동으로 적용하고,
              결과 카드에는 유사도와 응답 시간을 함께 보여줍니다.
            </p>

            <div className="suggestion-row">
              {suggestions.map((item) => (
                <button
                  key={item}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => applySuggestion(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <form className="search-composer" onSubmit={handleSubmit}>
            <div className="search-tabs" aria-label="검색 모드">
              <button
                type="button"
                className={searchMode === "text" ? "active" : ""}
                onClick={() => setSearchMode("text")}
              >
                텍스트
              </button>
              <button
                type="button"
                className={searchMode === "image" ? "active" : ""}
                onClick={() => setSearchMode("image")}
              >
                이미지
              </button>
              <button
                type="button"
                className={searchMode === "multimodal" ? "active" : ""}
                onClick={() => setSearchMode("multimodal")}
              >
                텍스트 + 이미지
              </button>
            </div>

            <label className="search-box">
              <span>텍스트 검색어</span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="예: 광택감 있는 블랙 아우터와 실버 포인트 자켓"
                aria-label="텍스트 검색어"
              />
            </label>

            <div className="composer-grid">
              <label className="upload-tile upload-label">
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <p>이미지 업로드</p>
                <span>
                  {uploadedImage
                    ? `${uploadedImage.name} · ${uploadedImage.sizeLabel}`
                    : "착장 사진, 스크린샷, 무드보드 이미지를 올려 보세요"}
                </span>
              </label>

              <div className="context-tile">
                <p>현재 검색 상태</p>
                <span>{helperMessage}</span>
              </div>
            </div>

            <div className="signal-list">
              <div className="signal-chip">
                <strong>입력 텍스트</strong>
                <span>{query.trim() || "텍스트 없이 이미지 기반 검색 대기 중"}</span>
              </div>
              <div className="signal-chip">
                <strong>업로드 이미지</strong>
                <span>{uploadedImage ? uploadedImage.name : "아직 업로드된 이미지가 없습니다."}</span>
              </div>
              <div className="signal-chip">
                <strong>실행 모드</strong>
                <span>{modeLabel}</span>
              </div>
            </div>

            <div className="search-actions">
              <button type="submit" className="primary-button" disabled={isSearching}>
                {isSearching ? "검색 중..." : "검색 실행"}
              </button>
              <span className="search-hint">텍스트만, 이미지만, 또는 둘 다 함께 검색할 수 있습니다.</span>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Search Results</p>
              <h3>검색 결과</h3>
            </div>
            <div className="heading-metrics">
              <span className="metric">응답 시간 {activeLatency}</span>
              <span className="metric">결과 수 {results.length}</span>
            </div>
          </div>

          <div className="result-list">
            {results.map((item) => (
              <article key={item.id} className="result-card">
                <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                <div className="result-meta">
                  <div className="result-topline">
                    <p>{item.brand}</p>
                    <strong>{item.price}</strong>
                  </div>
                  <h4>{item.title}</h4>
                  <p>{item.summary}</p>
                  <div className="result-stats">
                    <span className="badge">유사도 {(item.similarity * 100).toFixed(1)}%</span>
                    <span className="badge">{item.searchType}</span>
                    <span className="badge">응답 {item.responseTime}</span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Personalized Recommendations</p>
              <h3>
                {userId} 사용자에게 맞춘 추천 결과
              </h3>
            </div>
            <div className="heading-metrics">
              <span className="metric">Top-N {topN}</span>
              <span className="metric">총 추천 시간 {activeBundle.totalLatency}</span>
            </div>
          </div>

          <div className="recommendation-toolbar">
            <div className="recommendation-controls">
              <label className="user-id-field">
                <span>User ID</span>
                <input
                  value={userId}
                  onChange={(event) => setUserId(event.target.value)}
                  placeholder="예: user_1024"
                  aria-label="추천 대상 사용자 ID"
                  disabled={!isRegistered}
                />
              </label>
              <div className="persona-card">
                <span>선택한 페르소나</span>
                <strong>{selectedPersonaLabel}</strong>
              </div>
              <div className="persona-card">
                <span>추천에 반영된 성향</span>
                <strong>{activeBundle.persona}</strong>
              </div>
              <label className="user-id-field budget-field">
                <span>예산</span>
                <input
                  type="number"
                  min="0"
                  step="1000"
                  value={budget}
                  onChange={(event) => setBudget(event.target.value)}
                  placeholder="예: 200000"
                  aria-label="추천 예산"
                />
              </label>
            </div>
            <div className="recommendation-actions">
              <div className="topn-group" role="group" aria-label="Top N 추천 개수">
                {[3, 5].map((count) => (
                  <button
                    key={count}
                    type="button"
                    className={topN === count ? "mini-button active" : "mini-button"}
                    onClick={() => setTopN(count)}
                  >
                    Top {count}
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="primary-button"
                onClick={refreshRecommendations}
                disabled={isRefreshingRecommendations || !isRegistered}
              >
                {isRefreshingRecommendations ? "불러오는 중..." : "추천 다시 보기"}
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={loadBudgetSets}
                disabled={isLoadingBudgetSets || !isRegistered}
              >
                {isLoadingBudgetSets ? "세트 구성 중..." : "예산 안에서 세트 보기"}
              </button>
            </div>
          </div>

          <div className="weight-panel">
            <div className="weight-copy">
              <p className="eyebrow">Recommendation Control</p>
              <h4>개인화와 대중성의 비중을 조절합니다</h4>
              <p>슬라이더를 움직이면 추천 결과에서 취향 반영 강도와 대중성을 함께 조절할 수 있습니다.</p>
            </div>
            <div className="weight-control">
              <div className="weight-labels">
                <span>개인화 {recommendationWeight}%</span>
                <span>대중성 {popularityWeight}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={recommendationWeight}
                onChange={(event) => setRecommendationWeight(Number(event.target.value))}
                aria-label="개인화와 대중성 비중 조절"
              />
            </div>
          </div>

          {!isRegistered ? (
            <p className="status-text">사용자 설정과 취향 분석을 마치면 추천 결과가 여기에 표시됩니다.</p>
          ) : null}
          {recommendationError ? <p className="status-text">{recommendationError}</p> : null}
          {isRefreshingRecommendations ? (
            <p className="status-text">최신 추천 결과를 불러오는 중입니다.</p>
          ) : null}
          {budgetSetError ? <p className="status-text">{budgetSetError}</p> : null}

          <div className="stage-list">
            {activeBundle.stages.map((stage) => (
              <div key={stage.label} className="stage-chip">
                <strong>{stage.label}</strong>
                <span>{stage.value}</span>
              </div>
            ))}
          </div>

          <div className="recommendation-list">
            {activeBundle.items.map((item) => (
              <article
                key={item.id}
                className="result-card"
                onClick={() => handleRecommendationClick(item.id)}
              >
                <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                <div className="result-meta">
                  <div className="result-topline">
                    <p>
                      #{item.rank} · {item.brand}
                    </p>
                    <strong>{item.price}</strong>
                  </div>
                  <h4>{item.title}</h4>
                  <div className="reason-callout">
                    <span className="reason-label">추천 이유</span>
                    <p>{item.reason}</p>
                  </div>
                  <div className="result-stats">
                    <span className="badge">추천 점수 {(item.score * 100).toFixed(1)}%</span>
                    <span className="badge">{userId}</span>
                    <span className="badge">{activeBundle.persona}</span>
                    <span className="badge">예산 {budgetLabel}</span>
                  </div>
                </div>
              </article>
            ))}
          </div>

          <div className="section-heading" style={{ marginTop: 24 }}>
            <div>
              <p className="eyebrow">Budget Set</p>
              <h3>예산 안에서 구성한 추천 세트</h3>
            </div>
            <div className="heading-metrics">
              <span className="metric">예산 {budgetLabel}</span>
              <span className="metric">세트 수 {budgetSets.setCount}</span>
            </div>
          </div>

          {budgetSets.sets.length === 0 ? (
            <p className="status-text">예산 안에서 세트 보기를 누르면 추천 조합 결과가 여기에 표시됩니다.</p>
          ) : null}

          <div className="recommendation-list">
            {budgetSets.sets.map((setItems, setIndex) => (
              <article key={`set-${setIndex}`} className="panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Outfit Set</p>
                    <h3>세트 {setIndex + 1}</h3>
                  </div>
                  <div className="heading-metrics">
                    <span className="metric">
                      총액{" "}
                      {setItems
                        .reduce((sum, item) => sum + Number(item.price.replace(/[^0-9]/g, "") || 0), 0)
                        .toLocaleString("ko-KR")}
                      원
                    </span>
                  </div>
                </div>
                <div className="result-list">
                  {setItems.map((item) => (
                    <div key={`${setIndex}-${item.id}`} className="result-card">
                      <ResultVisual imageUrl={item.imageUrl} title={item.title} accent={item.accent} />
                      <div className="result-meta">
                        <div className="result-topline">
                          <p>{item.brand}</p>
                          <strong>{item.price}</strong>
                        </div>
                        <h4>{item.title}</h4>
                        <p>{item.category}</p>
                        <div className="result-stats">
                          <span className="badge">세트 점수 {(item.score * 100).toFixed(1)}%</span>
                          <span className="badge">{item.category}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
