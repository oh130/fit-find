from __future__ import annotations

import ast
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ab_test import compare_group_means
from metrics import mean_hit_rate_at_k, mean_ndcg_at_k, mean_reciprocal_rank


DEFAULT_RANKED_LISTS = """[
    ["item_7", "item_3", "item_9", "item_1"],
    ["item_2", "item_6", "item_8", "item_4"],
    ["item_5", "item_1", "item_2", "item_3"]
]"""

DEFAULT_RELEVANT_LISTS = """[
    ["item_3"],
    ["item_4", "item_8"],
    ["item_10"]
]"""

DEFAULT_CONTROL = "0,1,0,0,1,0,1,0,0,1"
DEFAULT_TREATMENT = "1,1,0,1,1,0,1,1,0,1"
SEARCH_REPORT_PATH = Path(__file__).resolve().with_name("search_metrics_report.json")
RECOMMENDATION_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "rec_models" / "reports" / "baseline" / "baseline_metrics.json"
)

PERSONA_LABELS = {
    "trendsetter": "트렌드세터형",
    "practical": "실용주의형",
    "value": "가성비추구형",
    "brand_loyal": "브랜드충성형",
    "impulse": "충동구매형",
    "careful": "신중탐색형",
    "repeat_stable": "반복구매형",
    "color_focus": "색상집중형",
    "category_focus": "카테고리집중형",
}

DEFAULT_PERSONA_SCORES = {
    "trendsetter": 28,
    "practical": 16,
    "value": 14,
    "brand_loyal": 7,
    "impulse": 6,
    "careful": 12,
    "repeat_stable": 5,
    "color_focus": 8,
    "category_focus": 4,
}

DEFAULT_ONBOARDING_RESPONSE = {
    "persona_scores": DEFAULT_PERSONA_SCORES,
}

DEFAULT_RECOMMENDATION_RESPONSE = {
    "user_id": "user_1024",
    "persona": "trendsetter",
    "recommendations": [
        {
            "product_id": "0825137001",
            "name": "Urban Edge Rider Jacket",
            "brand": "Mode Atelier",
            "category": "Outer",
            "price": 89000,
            "score": 0.94,
            "rank": 1,
            "reason": "ranking_score",
            "reason_text": "최근 탐색한 블랙 아우터 취향과 가장 가깝고, 실버 포인트 디테일이 잘 맞습니다.",
        },
        {
            "product_id": "0921184002",
            "name": "Minimal Zip Blouson",
            "brand": "Noir Form",
            "category": "Top",
            "price": 42000,
            "score": 0.9,
            "rank": 2,
            "reason": "session_interest_match",
            "reason_text": "미니멀한 출근룩 수요와 예산 범위를 함께 만족하는 안정적인 후보입니다.",
        },
        {
            "product_id": "0754401005",
            "name": "Chrome Detail Urban Rider",
            "brand": "Modu Lab",
            "category": "Accessory",
            "price": 58000,
            "score": 0.87,
            "rank": 3,
            "reason": "mab_exploration",
            "reason_text": "현재 취향과 유사하면서도 새로운 조합을 탐색하기 위한 실험 슬롯 상품입니다.",
        },
    ],
    "pipeline_latency": {
        "candidate_ms": 48,
        "ranking_ms": 61,
        "reranking_ms": 18,
        "total_ms": 127,
    },
}

DEFAULT_BUDGET_SET_RESPONSE = {
    "budget": 200000,
    "set_count": 2,
    "sets": [
        [
            {
                "article_id": "0825137001",
                "name": "Urban Edge Rider Jacket",
                "brand": "Mode Atelier",
                "category": "Outer",
                "price_int": 89000,
                "score": 0.94,
            },
            {
                "article_id": "0921184002",
                "name": "Minimal Zip Blouson",
                "brand": "Noir Form",
                "category": "Top",
                "price_int": 42000,
                "score": 0.9,
            },
            {
                "article_id": "0754401005",
                "name": "Chrome Detail Urban Rider",
                "brand": "Modu Lab",
                "category": "Accessory",
                "price_int": 58000,
                "score": 0.87,
            },
        ],
        [
            {
                "article_id": "0861123007",
                "name": "Blackline Cropped Moto",
                "brand": "Noir Craft",
                "category": "Outer",
                "price_int": 71000,
                "score": 0.91,
            },
            {
                "article_id": "0738829004",
                "name": "Gloss Rider Short",
                "brand": "Studio Hex",
                "category": "Bottom",
                "price_int": 58000,
                "score": 0.84,
            },
            {
                "article_id": "0910022008",
                "name": "Silver Trim Moto Crop",
                "brand": "Avenue N",
                "category": "Top",
                "price_int": 76000,
                "score": 0.88,
            },
        ],
    ],
}


def parse_nested_list(raw_text: str) -> list[list[str]]:
    parsed = ast.literal_eval(raw_text)
    if not isinstance(parsed, list):
        raise ValueError("Input must be a list.")

    result: list[list[str]] = []
    for row in parsed:
        if not isinstance(row, (list, tuple, set)):
            raise ValueError("Each row must be a list, tuple, or set.")
        result.append([str(item) for item in row])
    return result


def parse_numeric_series(raw_text: str) -> list[float]:
    values = [value.strip() for value in raw_text.split(",") if value.strip()]
    if not values:
        raise ValueError("Input must contain at least one numeric value.")
    return [float(value) for value in values]


def parse_item_cell(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    raw_text = str(value).strip()
    if not raw_text:
        return []

    try:
        parsed = ast.literal_eval(raw_text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    delimiter = "|" if "|" in raw_text else ","
    return [item.strip() for item in raw_text.split(delimiter) if item.strip()]


def load_ranking_lists_from_csv(uploaded_file) -> tuple[list[list[str]], list[list[str]], pd.DataFrame]:
    df = pd.read_csv(uploaded_file)
    required_columns = {"ranked_items", "relevant_items"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Ranking CSV must contain 'ranked_items' and 'relevant_items' columns.")

    ranked_lists = [parse_item_cell(value) for value in df["ranked_items"]]
    relevant_lists = [parse_item_cell(value) for value in df["relevant_items"]]
    preview_df = df.copy()
    if "query_id" not in preview_df.columns:
        preview_df.insert(0, "query_id", range(1, len(preview_df) + 1))
    return ranked_lists, relevant_lists, preview_df


def load_ab_data_from_csv(uploaded_file) -> tuple[list[float], list[float], pd.DataFrame]:
    df = pd.read_csv(uploaded_file)

    if {"group", "value"}.issubset(df.columns):
        normalized_group = df["group"].astype(str).str.strip().str.lower()
        control = df.loc[normalized_group == "control", "value"].astype(float).tolist()
        treatment = df.loc[normalized_group == "treatment", "value"].astype(float).tolist()
        if not control or not treatment:
            raise ValueError("A/B CSV with 'group' and 'value' must include both control and treatment rows.")
        return control, treatment, df

    if {"control", "treatment"}.issubset(df.columns):
        control = df["control"].dropna().astype(float).tolist()
        treatment = df["treatment"].dropna().astype(float).tolist()
        if not control or not treatment:
            raise ValueError("A/B CSV columns 'control' and 'treatment' must both contain values.")
        return control, treatment, df

    raise ValueError("A/B CSV must contain either 'group'+'value' or 'control'+'treatment' columns.")


st.set_page_config(page_title="Ranking Metrics Dashboard", layout="wide")
st.title("Search Engine, Ranking Metrics and A/B Test Dashboard")
st.caption("HitRate, MRR, nDCG, p-value, confidence interval")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K", min_value=1, max_value=20, value=3)
    confidence_level = st.slider("Confidence level", min_value=0.8, max_value=0.99, value=0.95, step=0.01)
    num_bootstrap = st.slider("Bootstrap samples", min_value=500, max_value=10000, value=3000, step=500)
    num_permutations = st.slider("Permutation samples", min_value=500, max_value=10000, value=3000, step=500)

st.subheader("Search Engine Metrics")
if SEARCH_REPORT_PATH.exists():
    try:
        report = json.loads(SEARCH_REPORT_PATH.read_text(encoding="utf-8"))
        search_metrics = report.get("search", {})
        checks = report.get("checks", {})
        thresholds = report.get("thresholds", {})
        metadata = report.get("metadata", {})
        metric_k_value = int(metadata.get("metric_k", 10))
        ndcg_metric_name = f"NDCG@{metric_k_value}"

        search_cards = st.columns(5)
        search_cards[0].metric("Samples", f"{metadata.get('samples_evaluated', 0)}")
        search_cards[1].metric("MRR", f"{search_metrics.get('MRR', 0.0):.4f}")
        search_cards[2].metric(
            f"nDCG@{metric_k_value}",
            f"{search_metrics.get(ndcg_metric_name, 0.0):.4f}",
        )
        search_cards[3].metric("Avg latency", f"{search_metrics.get('avg_wall_latency_ms', 0.0):.2f} ms")
        search_cards[4].metric("P95 latency", f"{search_metrics.get('p95_wall_latency_ms', 0.0):.2f} ms")

        status_df = pd.DataFrame(
            [
                {"metric": "MRR", "value": search_metrics.get("MRR", 0.0), "target": thresholds.get("mrr_min", 0.55)},
                {
                    "metric": f"nDCG@{metric_k_value}",
                    "value": search_metrics.get(ndcg_metric_name, 0.0),
                    "target": thresholds.get("ndcg_at_10_min", 0.50),
                },
                {
                    "metric": "Latency(ms)",
                    "value": search_metrics.get("avg_wall_latency_ms", 0.0),
                    "target": thresholds.get("latency_ms_max", 200.0),
                },
            ]
        )
        status_df["passed"] = [
            checks.get("mrr_meets_target", False),
            checks.get("ndcg_meets_target", False),
            checks.get("latency_within_200ms", False),
        ]
        status_df["passed"] = status_df["passed"].map({True: "PASS", False: "FAIL"})

        quality_df = status_df[status_df["metric"] != "Latency(ms)"].copy()
        latency_df = status_df[status_df["metric"] == "Latency(ms)"].copy()

        quality_chart = (
            alt.Chart(quality_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "passed:N",
                    legend=None,
                    scale=alt.Scale(domain=["PASS", "FAIL"], range=["#2E8B57", "#C0392B"]),
                ),
                tooltip=[
                    "metric",
                    alt.Tooltip("value:Q", format=".4f"),
                    alt.Tooltip("target:Q", format=".4f"),
                    alt.Tooltip("passed:N", title="status"),
                ],
            )
            .properties(height=240)
        )

        quality_target_rule = (
            alt.Chart(quality_df)
            .mark_rule(color="#2C3E50", strokeDash=[4, 4], strokeWidth=2)
            .encode(y="target:Q", x="metric:N")
        )

        latency_chart = (
            alt.Chart(latency_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#1F77B4")
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", title="ms"),
                tooltip=[
                    "metric",
                    alt.Tooltip("value:Q", format=".2f"),
                    alt.Tooltip("target:Q", format=".2f"),
                    alt.Tooltip("passed:N", title="status"),
                ],
            )
            .properties(height=240)
        )

        latency_target_rule = (
            alt.Chart(latency_df)
            .mark_rule(color="#C0392B", strokeDash=[4, 4], strokeWidth=2)
            .encode(y="target:Q")
        )

        search_chart_left, search_chart_right = st.columns(2)
        with search_chart_left:
            st.altair_chart(quality_chart + quality_target_rule, use_container_width=True)
        with search_chart_right:
            st.altair_chart(latency_chart + latency_target_rule, use_container_width=True)

        st.dataframe(status_df, use_container_width=True, hide_index=True)
        st.caption(f"Report source: {SEARCH_REPORT_PATH.name}")
    except Exception as error:
        st.error(f"Search report error: {error}")
else:
    st.info(
        "search_metrics_report.json not found. "
        "Run `python .\\search_engine\\generate_search_metrics_report.py --endpoint http://localhost:8002/search` "
        "and refresh this page after the JSON file is generated."
    )

st.divider()
st.subheader("Recommendation Metrics")
if RECOMMENDATION_REPORT_PATH.exists():
    try:
        report = json.loads(RECOMMENDATION_REPORT_PATH.read_text(encoding="utf-8"))
        metadata = report.get("metadata", {})
        candidate = report.get("candidate", {})
        ranking = report.get("ranking", {})
        recommendation = report.get("recommendation", {}).get("current_model", {})
        cold_start = recommendation.get("cold_start_subset", {})

        rec_cards = st.columns(6)
        rec_cards[0].metric("Recall@300", f"{candidate.get('Recall@300', 0.0):.4f}")
        rec_cards[1].metric("Ranking AUC", f"{ranking.get('auc', 0.0):.4f}")
        rec_cards[2].metric("HitRate@50", f"{recommendation.get('HitRate@50', 0.0):.4f}")
        rec_cards[3].metric("NDCG@50", f"{recommendation.get('NDCG@50', 0.0):.4f}")
        rec_cards[4].metric("Coverage@50", f"{recommendation.get('Coverage@50', 0.0):.4f}")
        rec_cards[5].metric("Users", f"{recommendation.get('users_evaluated', 0)}")

        recommendation_df = pd.DataFrame(
            [
                {"metric": "Recall@300", "value": candidate.get("Recall@300", 0.0), "group": "Candidate"},
                {"metric": "Ranking AUC", "value": ranking.get("auc", 0.0), "group": "Ranking"},
                {"metric": "HitRate@50", "value": recommendation.get("HitRate@50", 0.0), "group": "Recommendation"},
                {"metric": "NDCG@50", "value": recommendation.get("NDCG@50", 0.0), "group": "Recommendation"},
                {"metric": "Coverage@50", "value": recommendation.get("Coverage@50", 0.0), "group": "Recommendation"},
            ]
        )
        recommendation_chart = (
            alt.Chart(recommendation_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("group:N", legend=alt.Legend(title="pipeline")),
                tooltip=["metric", alt.Tooltip("value:Q", format=".4f"), "group"],
            )
            .properties(height=280)
        )
        st.altair_chart(recommendation_chart, use_container_width=True)

        rec_meta_col, cold_start_col = st.columns(2)
        with rec_meta_col:
            st.markdown("**Recommendation Report Summary**")
            st.write(
                {
                    "source_rows": metadata.get("rows", 0),
                    "users": metadata.get("users", 0),
                    "items": metadata.get("items", 0),
                    "top_k": metadata.get("top_k", 0),
                    "candidate_k": metadata.get("candidate_k", 0),
                }
            )
        with cold_start_col:
            st.markdown("**Cold Start Subset**")
            st.write(
                {
                    "users_evaluated": cold_start.get("users_evaluated", 0),
                    "HitRate@50": round(float(cold_start.get("HitRate@50", 0.0)), 4),
                    "NDCG@50": round(float(cold_start.get("NDCG@50", 0.0)), 4),
                    "Coverage@50": round(float(cold_start.get("Coverage@50", 0.0)), 4),
                }
            )
        st.caption(f"Report source: {RECOMMENDATION_REPORT_PATH.name}")
    except Exception as error:
        st.error(f"Recommendation report error: {error}")
else:
    st.info("Recommendation metrics report not found.")

st.divider()
demo_col_left, demo_col_right = st.columns(2)

with demo_col_left:
    st.subheader("Onboarding Persona Preview")
    st.caption("LLM 온보딩 결과를 발표용으로 빠르게 시각화하는 섹션")
    persona_source = st.radio(
        "Persona source",
        options=["Default sample", "Manual JSON"],
        horizontal=True,
        key="persona_source",
    )

    persona_scores = DEFAULT_ONBOARDING_RESPONSE["persona_scores"]
    if persona_source == "Manual JSON":
        persona_json = st.text_area(
            "Persona scores JSON",
            value=json.dumps(DEFAULT_ONBOARDING_RESPONSE, ensure_ascii=False, indent=2),
            height=220,
        )
        try:
            parsed_payload = json.loads(persona_json)
            parsed_scores = parsed_payload.get("persona_scores", {}) if isinstance(parsed_payload, dict) else {}
            if isinstance(parsed_scores, dict):
                persona_scores = {str(key): float(value) for key, value in parsed_scores.items()}
        except Exception as error:
            st.error(f"Persona JSON error: {error}")

    persona_df = pd.DataFrame(
        [
            {"persona": PERSONA_LABELS.get(key, key), "score": value}
            for key, value in persona_scores.items()
        ]
    ).sort_values("score", ascending=False)

    top_persona = persona_df.iloc[0]["persona"] if not persona_df.empty else "-"
    st.metric("Top Persona", top_persona)

    persona_chart = (
        alt.Chart(persona_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("score:Q", title="score (%)"),
            y=alt.Y("persona:N", sort="-x", title=None),
            color=alt.value("#1F77B4"),
            tooltip=["persona", alt.Tooltip("score:Q", format=".1f")],
        )
        .properties(height=320)
    )
    st.altair_chart(persona_chart, use_container_width=True)
    st.dataframe(persona_df, use_container_width=True, hide_index=True)
    with st.expander("Onboarding API payload preview", expanded=False):
        st.json({"persona_scores": persona_scores})

with demo_col_right:
    st.subheader("Budget Set Preview")
    st.caption("예산 기반 세트 추천 결과를 발표용으로 미리 보여주는 섹션")
    budget_limit = st.number_input("Budget", min_value=10000, value=DEFAULT_BUDGET_SET_RESPONSE["budget"], step=10000)
    budget_rows = []
    for set_index, set_items in enumerate(DEFAULT_BUDGET_SET_RESPONSE["sets"], start=1):
        for item in set_items:
            budget_rows.append(
                {
                    "set_name": f"세트 {set_index}",
                    "article_id": item["article_id"],
                    "item_name": item["name"],
                    "brand": item["brand"],
                    "category": item["category"],
                    "price": item["price_int"],
                    "score": item["score"],
                }
            )
    budget_set_df = pd.DataFrame(budget_rows)
    total_by_set = budget_set_df.groupby("set_name", as_index=False)["price"].sum()
    total_by_set["within_budget"] = total_by_set["price"] <= budget_limit

    budget_cards = st.columns(len(total_by_set) if len(total_by_set) > 0 else 1)
    for card, row in zip(budget_cards, total_by_set.itertuples(index=False)):
        card.metric(
            row.set_name,
            f"{int(row.price):,}원",
            "PASS" if row.within_budget else "OVER",
        )

    budget_chart = (
        alt.Chart(total_by_set)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("set_name:N", title=None),
            y=alt.Y("price:Q", title="total price"),
            color=alt.Color(
                "within_budget:N",
                scale=alt.Scale(domain=[True, False], range=["#2E8B57", "#C0392B"]),
                legend=alt.Legend(title="within budget"),
            ),
            tooltip=["set_name", alt.Tooltip("price:Q", format=",.0f"), "within_budget"],
        )
        .properties(height=260)
    )
    budget_rule = alt.Chart(pd.DataFrame([{"budget": budget_limit}])).mark_rule(strokeDash=[4, 4]).encode(
        y="budget:Q"
    )
    st.altair_chart(budget_chart + budget_rule, use_container_width=True)
    st.dataframe(budget_set_df, use_container_width=True, hide_index=True)
    with st.expander("Budget Set API payload preview", expanded=False):
        st.json(DEFAULT_BUDGET_SET_RESPONSE)

st.divider()
st.subheader("Recommendation Reason Preview")
st.caption("include_reasons=true 응답의 reason_text를 발표용으로 확인하는 섹션")

reason_df = pd.DataFrame(
    [
        {
            "rank": item["rank"],
            "product_id": item["product_id"],
            "name": item["name"],
            "reason": item["reason"],
            "reason_text": item["reason_text"],
            "score": item["score"],
            "price": item["price"],
        }
        for item in DEFAULT_RECOMMENDATION_RESPONSE["recommendations"]
    ]
)

reason_cards = st.columns(len(reason_df) if len(reason_df) > 0 else 1)
for card, row in zip(reason_cards, reason_df.itertuples(index=False)):
    card.metric(f"Rank {row.rank}", row.name, f"{row.score:.2f}")

for row in reason_df.itertuples(index=False):
    with st.container():
        st.markdown(f"**#{row.rank} {row.name}**")
        st.write(
            {
                "product_id": row.product_id,
                "reason": row.reason,
                "reason_text": row.reason_text,
                "price": row.price,
                "score": round(float(row.score), 4),
            }
        )

latency_payload = DEFAULT_RECOMMENDATION_RESPONSE["pipeline_latency"]
latency_df = pd.DataFrame(
    [
        {"stage": "Candidate", "latency_ms": latency_payload["candidate_ms"]},
        {"stage": "Ranking", "latency_ms": latency_payload["ranking_ms"]},
        {"stage": "Reranking", "latency_ms": latency_payload["reranking_ms"]},
        {"stage": "Total", "latency_ms": latency_payload["total_ms"]},
    ]
)
reason_latency_chart = (
    alt.Chart(latency_df)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#6C5CE7")
    .encode(
        x=alt.X("stage:N", title=None),
        y=alt.Y("latency_ms:Q", title="ms"),
        tooltip=["stage", "latency_ms"],
    )
    .properties(height=220)
)
st.altair_chart(reason_latency_chart, use_container_width=True)
with st.expander("Recommendation API payload preview", expanded=False):
    st.json(DEFAULT_RECOMMENDATION_RESPONSE)

st.divider()
ranking_col, ab_col = st.columns(2)

with ranking_col:
    st.subheader("Ranking Metrics")
    ranking_file = st.file_uploader("Upload ranking CSV", type="csv", key="ranking_csv")
    st.caption("Ranking CSV columns: 'ranked_items', 'relevant_items', optional 'query_id'. List format can be ['a','b'] or a|b|c.")

    ranked_input = st.text_area("Ranked lists", value=DEFAULT_RANKED_LISTS, height=180, disabled=ranking_file is not None)
    relevant_input = st.text_area("Relevant lists", value=DEFAULT_RELEVANT_LISTS, height=180, disabled=ranking_file is not None)

    try:
        if ranking_file is not None:
            ranked_lists, relevant_lists, ranking_preview_df = load_ranking_lists_from_csv(ranking_file)
        else:
            ranked_lists = parse_nested_list(ranked_input)
            relevant_lists = parse_nested_list(relevant_input)
            ranking_preview_df = pd.DataFrame(
                {
                    "query_id": range(1, len(ranked_lists) + 1),
                    "ranked_items": ranked_lists,
                    "relevant_items": relevant_lists,
                }
            )

        metrics_df = pd.DataFrame(
            [
                {"metric": f"HitRate@{k}", "value": mean_hit_rate_at_k(ranked_lists, relevant_lists, k)},
                {"metric": "MRR", "value": mean_reciprocal_rank(ranked_lists, relevant_lists)},
                {"metric": f"nDCG@{k}", "value": mean_ndcg_at_k(ranked_lists, relevant_lists, k)},
            ]
        )

        metric_cards = st.columns(3)
        for card, row in zip(metric_cards, metrics_df.itertuples(index=False)):
            card.metric(row.metric, f"{row.value:.4f}")

        ranking_chart = (
            alt.Chart(metrics_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", legend=None),
                tooltip=["metric", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=320)
        )
        st.altair_chart(ranking_chart, use_container_width=True)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        with st.expander("Ranking data preview", expanded=False):
            st.dataframe(ranking_preview_df, use_container_width=True, hide_index=True)
    except Exception as error:
        st.error(f"Ranking input error: {error}")

with ab_col:
    st.subheader("A/B Test")
    ab_file = st.file_uploader("Upload A/B CSV", type="csv", key="ab_csv")
    st.caption("A/B CSV columns: 'group','value' or 'control','treatment'.")

    control_input = st.text_area("Control values", value=DEFAULT_CONTROL, height=120, disabled=ab_file is not None)
    treatment_input = st.text_area("Treatment values", value=DEFAULT_TREATMENT, height=120, disabled=ab_file is not None)

    try:
        if ab_file is not None:
            control, treatment, ab_preview_df = load_ab_data_from_csv(ab_file)
        else:
            control = parse_numeric_series(control_input)
            treatment = parse_numeric_series(treatment_input)
            ab_preview_df = pd.DataFrame(
                {
                    "control": pd.Series(control, dtype=float),
                    "treatment": pd.Series(treatment, dtype=float),
                }
            )
        result = compare_group_means(
            control=control,
            treatment=treatment,
            confidence_level=confidence_level,
            num_bootstrap=num_bootstrap,
            num_permutations=num_permutations,
        )

        ab_cards = st.columns(4)
        ab_cards[0].metric("Control mean", f"{result.control_mean:.4f}")
        ab_cards[1].metric("Treatment mean", f"{result.treatment_mean:.4f}")
        ab_cards[2].metric("p-value", f"{result.p_value:.4f}")
        ab_cards[3].metric("Relative lift", f"{result.relative_lift:.2%}")

        summary_df = pd.DataFrame(
            [
                {"metric": "Control", "value": result.control_mean},
                {"metric": "Treatment", "value": result.treatment_mean},
            ]
        )
        summary_chart = (
            alt.Chart(summary_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("value:Q", title="Mean"),
                color=alt.Color("metric:N", legend=None),
                tooltip=["metric", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=220)
        )
        st.altair_chart(summary_chart, use_container_width=True)

        ci_low, ci_high = result.confidence_interval
        ci_df = pd.DataFrame(
            [
                {
                    "label": f"{int(confidence_level * 100)}% CI",
                    "diff": result.absolute_diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            ]
        )
        ci_chart = (
            alt.Chart(ci_df)
            .mark_point(filled=True, size=110)
            .encode(
                x=alt.X("ci_low:Q", title="Absolute difference with confidence interval"),
                x2="ci_high:Q",
                y=alt.Y("label:N", title=None),
                tooltip=[
                    alt.Tooltip("diff:Q", title="Absolute diff", format=".4f"),
                    alt.Tooltip("ci_low:Q", title="CI low", format=".4f"),
                    alt.Tooltip("ci_high:Q", title="CI high", format=".4f"),
                ],
            )
        )
        ci_rule = (
            alt.Chart(ci_df)
            .mark_rule(strokeWidth=4)
            .encode(x="ci_low:Q", x2="ci_high:Q", y="label:N")
        )
        zero_rule = alt.Chart(pd.DataFrame([{"x": 0.0}])).mark_rule(strokeDash=[6, 4]).encode(x="x:Q")
        st.altair_chart((ci_rule + ci_chart + zero_rule).properties(height=140), use_container_width=True)

        st.write(
            {
                "absolute_diff": round(result.absolute_diff, 4),
                "p_value": round(result.p_value, 4),
                "confidence_interval": tuple(round(bound, 4) for bound in result.confidence_interval),
            }
        )
        with st.expander("A/B data preview", expanded=False):
            st.dataframe(ab_preview_df, use_container_width=True, hide_index=True)
    except Exception as error:
        st.error(f"A/B input error: {error}")

st.divider()
st.markdown(
    """
    `Run command`

    ```powershell
    .\\.venv\\Scripts\\python.exe -m streamlit run .\\streamlit_app.py
    ```
    """
)
st.markdown(
    """
    `Example ranking CSV`

    ```csv
    query_id,ranked_items,relevant_items
    q1,"item_7|item_3|item_9|item_1","item_3"
    q2,"item_2|item_6|item_8|item_4","item_4|item_8"
    q3,"item_5|item_1|item_2|item_3","item_10"
    ```

    `Example A/B CSV`

    ```csv
    group,value
    control,0
    control,1
    treatment,1
    treatment,0
    ```
    """
)
