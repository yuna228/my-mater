import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 기본 설정
st.set_page_config(page_title="서울 연평균 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")

# 2. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # '날짜' 열을 datetime 타입으로 변환 후 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 조건 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & 
        (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    # 독립변수 X: 1908년 기준 경과 연수
    filtered_df["경과연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

df_filtered = load_and_preprocess_data()

# 3. 회귀 직선 및 상관계수 계산
X = df_filtered["경과연수"].values
Y = df_filtered["연평균기온"].values

# 선형 회귀 계수 (기울기, 절편)
slope, intercept = np.polyfit(X, Y, 1)

# 피어슨 상관계수
correlation = np.corrcoef(X, Y)[0, 1]

# 데이터 통계 정보 추출
data_count = len(df_filtered)
start_year = int(df_filtered["연도"].min())
end_year = int(df_filtered["연도"].max())

# 4. 연도 선택 및 기온 예측
st.subheader("🔮 미래 기온 예측")
selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

# 선택된 연도의 예측 기온 계산 (1908년 기준 경과연수 반영)
predicted_temp = slope * (selected_year - 1908) + intercept

st.metric(
    label=f"{selected_year}년 서울 예상 연평균 기온",
    value=f"{predicted_temp:.2f} °C"
)

st.divider()

# 5. 시각화 (Plotly)
# 전체 슬라이더 범위(1900~2100)에 대응하는 회귀선 데이터
line_years = np.arange(1900, 2101)
line_X = line_years - 1908
line_Y = slope * line_X + intercept

fig = go.Figure()

# 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_filtered["연도"],
    y=df_filtered["연평균기온"],
    mode="markers",
    name="관측 연평균기온",
    marker=dict(color="royalblue", size=7)
))

# 회귀 직선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_Y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="firebrick", width=2)
))

# 선택한 연도 강조 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택한 연도 ({selected_year}년)",
    marker=dict(color="orange", size=14, symbol="star")
))

fig.update_layout(
    title="서울 연도별 연평균 기온 추이 및 회귀선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 6. 통계 정보 출력
st.subheader("📊 분석 요약 정보")
col1, col2, col3, col4 = st.columns(4)
col1.metric("선형 상관계수", f"{correlation:.4f}")
col2.metric("분석 데이터 개수", f"{data_count} 개 해")
col3.metric("시작 연도", f"{start_year} 년")
col4.metric("끝 연도", f"{end_year} 년")
