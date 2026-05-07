import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import {
  RecommendationBundle,
  fetchRecommendations,
  fetchSearchResults,
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
      summary: "광택 있는 블랙 아우터와 실버 하드웨어 조건에 강하게 매칭된 결과입니다.",
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
      summary: "미니멀, 블랙, 캐주얼 무드 선호가 많이 겹치는 대체 후보입니다.",
      accent: "linear-gradient(135deg, #84553a 0%, #1a1d26 100%)",
    },
    {
      id: 3,
      title: "Night Shift Vegan Leather",
      brand: "Archive Move",
      price: "64,000원",
      similarity: 0.87,
      searchType: "텍스트 검색",
      responseTime: "128ms",
      summary: "소재 질감과 착용 상황 조건을 함께 만족하는 상품입니다.",
      accent: "linear-gradient(135deg, #04545f 0%, #131620 100%)",
    },
  ],
  image: [
    {
      id: 4,
      title: "Silver Trim Moto Crop",
      brand: "Avenue N",
      price: "76,000원",
      similarity: 0.96,
      searchType: "이미지 검색",
      responseTime: "173ms",
      summary: "업로드 이미지의 실루엣과 메탈 포인트를 가장 가깝게 반영한 결과입니다.",
      accent: "linear-gradient(135deg, #26314c 0%, #11151d 100%)",
    },
    {
      id: 5,
      title: "Gloss Rider Short",
      brand: "Studio Hex",
      price: "58,000원",
      similarity: 0.91,
      searchType: "이미지 검색",
      responseTime: "173ms",
      summary: "질감과 길이감이 비슷한 이미지 후보를 매칭했습니다.",
      accent: "linear-gradient(135deg, #5b402f 0%, #181720 100%)",
    },
    {
      id: 6,
      title: "Metro Faux Leather Zip-up",
      brand: "Common Surface",
      price: "39,000원",
      similarity: 0.88,
      searchType: "이미지 검색",
      responseTime: "173ms",
      summary: "비슷한 착장 비율과 어두운 색 분포를 가진 후보입니다.",
      accent: "linear-gradient(135deg, #0d5c5c 0%, #141821 100%)",
    },
  ],
  multimodal: [
    {
      id: 7,
      title: "Chrome Detail Urban Rider",
      brand: "Modu Lab",
      price: "98,000원",
      similarity: 0.98,
      searchType: "텍스트 + 이미지",
      responseTime: "214ms",
      summary: "텍스트 의도와 이미지 특징이 동시에 일치해 가장 높은 점수를 받은 결과입니다.",
      accent: "linear-gradient(135deg, #42294f 0%, #11131c 100%)",
    },
    {
      id: 8,
      title: "Blackline Cropped Moto",
      brand: "Noir Craft",
      price: "71,000원",
      similarity: 0.94,
      searchType: "텍스트 + 이미지",
      responseTime: "214ms",
      summary: "질감의 분위기와 업로드 이미지의 디테일을 함께 반영한 검색 결과입니다.",
      accent: "linear-gradient(135deg, #72412f 0%, #171923 100%)",
    },
    {
      id: 9,
      title: "Late Evening Leather Bloom",
      brand: "Volume Edit",
      price: "83,000원",
      similarity: 0.9,
      searchType: "텍스트 + 이미지",
      responseTime: "214ms",
      summary: "룩의 사용 상황과 이미지 기반 스타일 선호를 함께 반영한 후보입니다.",
      accent: "linear-gradient(135deg, #00545c 0%, #10151d 100%)",
    },
  ],
};

const suggestions = [
  "미니멀 블랙 가죽 재킷",
  "실버 포인트가 있는 스트리트 룩",
  "봄 데일리용 체크 아우터",
];

const personaOptions: PersonaOption[] = [
  {
    name: "트렌드세터",
    title: "새로운 스타일을 먼저 시도해요",
    summary: "유행, 바이럴 아이템, 시즌 무드에 빠르게 반응하는 유형입니다.",
    traits: ["유행 민감", "스타일 실험", "빠른 반응"],
  },
  {
    name: "실용주의자",
    title: "착용감과 활용도를 중요하게 봐요",
    summary: "오래 입을 수 있고 여러 상황에 맞는 아이템을 선호하는 유형입니다.",
    traits: ["활용도 우선", "편안한 착용감", "기본 아이템 선호"],
  },
  {
    name: "가성비추구",
    title: "가격 대비 만족도가 중요해요",
    summary: "가격, 할인, 품질 균형을 꼼꼼히 보는 유형입니다.",
    traits: ["가격 민감", "할인 선호", "비교 구매"],
  },
  {
    name: "브랜드충성형",
    title: "좋아하는 브랜드를 꾸준히 선택해요",
    summary: "브랜드 정체성과 구매 경험을 중요하게 여기는 유형입니다.",
    traits: ["브랜드 선호", "재구매 경향", "일관된 취향"],
  },
  {
    name: "충동구매형",
    title: "마음에 들면 빠르게 결정해요",
    summary: "강한 시각적 매력이나 즉각적인 만족에 반응하는 유형입니다.",
    traits: ["빠른 결정", "비주얼 반응", "즉시 구매"],
  },
  {
    name: "신중탐색형",
    title: "여러 옵션을 오래 비교해요",
    summary: "리뷰, 소재, 가격을 충분히 검토한 뒤 결정하는 유형입니다.",
    traits: ["오래 비교", "정보 탐색", "신중한 결정"],
  },
  {
    name: "재구매반복형",
    title: "익숙한 상품을 다시 찾는 편이에요",
    summary: "만족했던 상품이나 비슷한 스타일을 반복 구매하는 유형입니다.",
    traits: ["재구매 선호", "검증된 선택", "안정적 취향"],
  },
  {
    name: "색상집중형",
    title: "선호하는 색감이 뚜렷해요",
    summary: "특정 컬러 팔레트 안에서 아이템을 고르는 경향이 강한 유형입니다.",
    traits: ["색감 우선", "톤 일관성", "컬러 필터"],
  },
  {
    name: "카테고리집중형",
    title: "관심 카테고리를 깊게 탐색해요",
    summary: "특정 카테고리 안에서 다양한 옵션을 집중적으로 살펴보는 유형입니다.",
    traits: ["카테고리 몰입", "내부 비교", "명확한 관심사"],
  },
];

const emptyBundle: RecommendationBundle = {
  items: [],
  totalLatency: "0ms",
  stages: [],
  persona: "미분류",
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
  const [selectedOnboardingPersona, setSelectedOnboardingPersona] = useState("트렌드세터");
  const [query, setQuery] = useState("광택감 있는 블랙 아우터에 실버 포인트가 있는 룩");
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

  const helperMessage = useMemo(() => {
    if (searchMode === "text") {
      return "텍스트 질의만으로 유사 상품을 찾습니다.";
    }

    if (searchMode === "image") {
      return "업로드한 이미지 특징을 기반으로 시각적으로 유사한 상품을 찾습니다.";
    }

    return "텍스트 의도와 이미지 특징을 함께 반영해 가장 강한 후보를 우선 정렬합니다.";
  }, [searchMode]);

  useEffect(() => {
    if (!isRegistered || showOnboarding) {
      return;
    }

    let cancelled = false;

    const loadRecommendations = async () => {
      setIsRefreshingRecommendations(true);
      setRecommendationError(null);

      try {
        const bundle = await fetchRecommendations(
          userId.trim() || "anonymous",
          topN,
          recommendationSeed,
          selectedOnboardingPersona,
        );

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
  }, [isRegistered, showOnboarding, userId, topN, recommendationSeed, selectedOnboardingPersona]);

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
    } catch {
      setResults(baseResults[nextMode]);
      setActiveLatency(baseResults[nextMode][0]?.responseTime ?? "128ms");
    } finally {
      setLastSearchedAt(
        new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }),
      );
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

  const handleSignUp = () => {
    if (!userId.trim()) {
      return;
    }

    setIsRegistered(true);
    setShowOnboarding(true);
  };

  const refreshRecommendations = () => {
    setRecommendationSeed((current) => current + 1);
  };

  const startWithPersona = () => {
    setShowOnboarding(false);
    setRecommendationSeed(0);
  };

  const popularityWeight = 100 - recommendationWeight;
  const modeLabel =
    searchMode === "multimodal" ? "멀티모달" : searchMode === "image" ? "이미지" : "텍스트";
  const budgetLabel = `${Number(budget || 0).toLocaleString("ko-KR")}원`;

  if (showOnboarding) {
    return (
      <div className="app-shell onboarding-shell">
        <section className="onboarding-panel">
          <div className="onboarding-copy">
            <p className="eyebrow">Cold Start Onboarding</p>
            <h1>처음 방문하셨군요. 먼저 쇼핑 성향을 알려주세요.</h1>
            <p>
              추천 정확도를 높이기 위해 가장 가까운 페르소나를 하나 선택해 주세요. 이 선택은
              초기 추천에만 사용되고 이후 행동 데이터로 계속 업데이트됩니다.
            </p>
          </div>

          <div className="persona-grid">
            {personaOptions.map((persona) => (
              <button
                key={persona.name}
                type="button"
                className={
                  selectedOnboardingPersona === persona.name
                    ? "persona-option active"
                    : "persona-option"
                }
                onClick={() => setSelectedOnboardingPersona(persona.name)}
              >
                <p className="persona-name">{persona.name}</p>
                <h2>{persona.title}</h2>
                <p className="persona-summary">{persona.summary}</p>
                <div className="persona-traits">
                  {persona.traits.map((trait) => (
                    <span key={trait} className="badge">
                      {trait}
                    </span>
                  ))}
                </div>
              </button>
            ))}
          </div>

          <div className="onboarding-footer">
            <div className="persona-card">
              <span>선택한 페르소나</span>
              <strong>{selectedOnboardingPersona}</strong>
            </div>
            <button type="button" className="primary-button" onClick={startWithPersona}>
              이 성향으로 시작하기
            </button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ModeMosaic</p>
          <h1>텍스트와 이미지 검색, 개인화 추천까지 이어지는 패션 탐색 화면</h1>
        </div>
        <div className="topbar-meta">
          <span>검색 모드: {modeLabel}</span>
          <span>최근 검색: {lastSearchedAt}</span>
          <span>추천 대상: {userId}</span>
          <span>회원 상태: {isRegistered ? "가입 완료" : "미가입"}</span>
          <span>초기 페르소나: {selectedOnboardingPersona}</span>
          <span>추론 페르소나: {activeBundle.persona}</span>
          <span>추천 Top-N: {topN}</span>
          <span>
            추천형 {recommendationWeight} / 인기형 {popularityWeight}
          </span>
          <span>예산: {budgetLabel}</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel signup-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Registration</p>
              <h3>먼저 user_id로 회원가입을 진행하세요</h3>
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
              {isRegistered ? "회원가입 완료" : "회원가입"}
            </button>
          </div>
          <p className="status-text signup-text">
            회원가입 버튼을 누르면 다음 단계로 페르소나 후보가 열리고, 그 결과가 초기 추천에
            반영됩니다.
          </p>
        </section>

        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Search Console</p>
            <h2>검색 시작점에서 바로 멀티모달 탐색이 가능하도록 구성한 화면</h2>
            <p className="hero-description">
              텍스트 질의와 이미지 업로드를 함께 받아 검색 타입을 자동으로 판단하고, 결과 카드에는
              유사도 점수와 응답 시간을 함께 보여줍니다.
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
                placeholder="예: 광택감 있는 블랙 아우터에 실버 포인트가 있는 룩"
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
                    : "착장 사진, 스크린샷, 무드보드 이미지를 올려보세요."}
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
              <span className="search-hint">
                텍스트만, 이미지만, 또는 둘을 함께 검색할 수 있습니다.
              </span>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Search Results</p>
              <h3>검색 결과 리스트</h3>
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
              <p className="eyebrow">Recommendations</p>
              <h3>
                {userId} · {activeBundle.persona} 기반 추천 결과
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
                <span>Onboarding Persona</span>
                <strong>{selectedOnboardingPersona}</strong>
              </div>
              <div className="persona-card">
                <span>Detected Persona</span>
                <strong>{activeBundle.persona}</strong>
              </div>
              <label className="user-id-field budget-field">
                <span>Budget</span>
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
                {isRefreshingRecommendations ? "재추천 중..." : "재추천"}
              </button>
            </div>
          </div>

          <div className="weight-panel">
            <div className="weight-copy">
              <p className="eyebrow">Diversity Control</p>
              <h4>추천형과 인기형 가중치 조절</h4>
              <p>
                추천형을 높이면 개인 성향이 강해지고, 인기형을 높이면 더 대중적인 상품이 상위에
                노출됩니다.
              </p>
            </div>
            <div className="weight-control">
              <div className="weight-labels">
                <span>추천형 {recommendationWeight}%</span>
                <span>인기형 {popularityWeight}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={recommendationWeight}
                onChange={(event) => setRecommendationWeight(Number(event.target.value))}
                aria-label="추천형과 인기형 가중치 조절"
              />
            </div>
          </div>

          {!isRegistered ? (
            <p className="status-text">
              회원가입과 페르소나 선택을 완료하면 추천 결과를 불러옵니다.
            </p>
          ) : null}
          {recommendationError ? <p className="status-text">{recommendationError}</p> : null}
          {isRefreshingRecommendations ? (
            <p className="status-text">
              추천 API에서 {userId} 기반 결과를 불러오는 중입니다.
            </p>
          ) : null}

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
                  <p>{item.reason}</p>
                  <div className="result-stats">
                    <span className="badge">추천 점수 {(item.score * 100).toFixed(1)}%</span>
                    <span className="badge">개인화 추천</span>
                    <span className="badge">{userId}</span>
                    <span className="badge">{activeBundle.persona}</span>
                    <span className="badge">예산 {budgetLabel}</span>
                    <span className="badge">
                      추천형 {recommendationWeight} / 인기형 {popularityWeight}
                    </span>
                    <span className="badge">추천 이유 표시</span>
                  </div>
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
