import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# 3. 전체 기간 vs 최근 20년 데이터 분리 및 회귀분석
# (1) 전체 기간
X_all = df_filtered["경과연수"].values
Y_all = df_filtered["연평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)
corr_all = np.corrcoef(X_all, Y_all)[0, 1]

# (2) 최근 20년 (끝 연도 기준 최근 20개 관측 연도)
max_year = df_filtered["연도"].max()
df_recent20 = df_filtered[df_filtered["연도"] > (max_year - 20)].copy()

X_recent = df_recent20["경과연수"].values
Y_recent = df_recent20["연평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)

# 100년당 상승온도 계산 (1년당 기울기 * 100)
rate_100y_all = slope_all * 100
rate_100y_recent = slope_recent * 100

# 데이터 통계 정보
data_count = len(df_filtered)
start_year = int(df_filtered["연도"].min())
end_year = int(df_filtered["연도"].max())
recent_start_year = int(df_recent20["연도"].min())

# 4. 100년당 온난화 속도 비교 (화면에 크게 표시)
st.subheader("🔥 100년당 기온 상승 속도 비교")
col_rate1, col_rate2 = st.columns(2)

with col_rate1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year}년)",
        value=f"{rate_100y_all:+.2f} °C / 100년",
        delta=f"연간 약 {slope_all:+.3f} °C"
    )

with col_rate2:
    # 전체 기간 대비 최근 20년의 가속도 계산
    diff_rate = rate_100y_recent - rate_100y_all
    st.metric(
        label=f"최근 20년 ({recent_start_year}~{end_year}년)",
        value=f"{rate_100y_recent:+.2f} °C / 100년",
        delta=f"전체 평균 대비 {diff_rate:+.2f} °C/100년 빠르게 상승 중",
        delta_color="inverse"
    )

st.divider()

# 5. 연도 선택 및 기온 예측 (전체 기간 회귀선 기준)
st.subheader("🔮 미래 기온 예측")
selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

predicted_temp_all = slope_all * (selected_year - 1908) + intercept_all
predicted_temp_recent = slope_recent * (selected_year - 1908) + intercept_recent

pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    st.metric(
        label=f"{selected_year}년 예상 기온 (전체 기간 추세 기준)",
        value=f"{predicted_temp_all:.2f} °C"
    )
with pred_col2:
    st.metric(
        label=f"{selected_year}년 예상 기온 (최근 20년 추세 기준)",
        value=f"{predicted_temp_recent:.2f} °C"
    )

st.divider()

# 6. 시각화 (Plotly)
line_years = np.arange(1900, 2101)
line_X = line_years - 1908

# 추세선 Y값 계산
line_Y_all = slope_all * line_X + intercept_all
line_Y_recent = slope_recent * line_X + intercept_recent

fig = go.Figure()

# 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_filtered["연도"],
    y=df_filtered["연평균기온"],
    mode="markers",
    name="관측 연평균기온",
    marker=dict(color="gray", opacity=0.7, size=7)
))

# 전체 기간 회귀선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_Y_all,
    mode="lines",
    name=f"전체 기간 회귀선 ({rate_100y_all:+.2f}°C/100년)",
    line=dict(color="royalblue", width=2)
))

# 최근 20년 회귀선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_Y_recent,
    mode="lines",
    name=f"최근 20년 회귀선 ({rate_100y_recent:+.2f}°C/100년)",
    line=dict(color="firebrick", width=2, dash="dash")
))

# 선택한 연도 강조 표시 (전체 추세 기준)
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp_all],
    mode="markers",
    name=f"선택 연도 예측 ({selected_year}년)",
    marker=dict(color="orange", size=14, symbol="star")
))

fig.update_layout(
    title="서울 연도별 연평균 기온 추이 및 추세선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 7. 통계 정보 출력
st.subheader("📊 전체 데이터 요약 정보")
info1, info2, info3, info4 = st.columns(4)
info1.metric("선형 상관계수 (전체)", f"{corr_all:.4f}")
info2.metric("분석 데이터 개수", f"{data_count} 개 해")
info3.metric("시작 연도", f"{start_year} 년")
info4.metric("끝 연도", f"{end_year} 년")

# 1. 페이지 기본 설정
st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 선형회귀 모델 학습 및 평가")

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

df_all = load_and_preprocess_data()

# 3. 데이터셋 분할 및 모델 학습 함수
def train_and_eval(train_df, test_df):
    X_train = train_df[["경과연수"]]
    y_train = train_df["연평균기온"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 기울기 및 100년당 기온 상승량
    slope = model.coef_[0]
    slope_100y = slope * 100
    intercept = model.intercept_
    
    # 테스트 데이터 평가 (테스트 데이터가 있는 경우만)
    if test_df is not None and len(test_df) > 0:
        X_test = test_df[["경과연수"]]
        y_test = test_df["연평균기온"]
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    else:
        mae, mse, r2 = None, None, None
        
    return model, slope, slope_100y, intercept, mae, mse, r2

# 데이터셋 정의
# (1) 공통 테스트 데이터: 최근 20년 (2006 ~ 2025)
test_df = df_all[(df_all["연도"] >= 2006) & (df_all["연도"] <= 2025)]

# (2) 학습 데이터셋
train_50y = df_all[(df_all["연도"] >= 1956) & (df_all["연도"] <= 2005)]
train_100y = df_all[(df_all["연도"] >= 1906) & (df_all["연도"] <= 2005)]

# 모델 학습 및 평가
model_all, slope_all, slope100_all, ic_all, _, _, _ = train_and_eval(df_all, None)
model_50y, slope_50y, slope100_50y, ic_50y, mae_50y, mse_50y, r2_50y = train_and_eval(train_50y, test_df)
model_100y, slope_100y, slope100_100y, ic_100y, mae_100y, mse_100y, r2_100y = train_and_eval(train_100y, test_df)

# 전체 데이터 자체에 대한 평가 (Self-evaluation)
X_all_full = df_all[["경과연수"]]
y_all_full = df_all["연평균기온"]
y_pred_all = model_all.predict(X_all_full)
mae_all = mean_absolute_error(y_all_full, y_pred_all)
mse_all = mean_squared_error(y_all_full, y_pred_all)
r2_all = r2_score(y_all_full, y_pred_all)

# 4. 모델 성능 및 기울기 비교표 출력
st.subheader("📊 모델별 기울기 및 테스트 데이터(2006~2025년) 예측 성능 비교")

summary_data = {
    "모델 구분": ["전체 데이터 (전체 학습)", "과거 50년 학습 (1956~2005)", "과거 100년 학습 (1906~2005)"],
    "학습 기간": [f"{df_all['연도'].min()}~{df_all['연도'].max()}", "1956~2005 (50년)", "1906~2005 (100년)"],
    "학습 데이터 개수": [f"{len(df_all)}개", f"{len(train_50y)}개", f"{len(train_100y)}개"],
    "100년당 기온 상승량": [f"{slope100_all:+.2f} °C", f"{slope100_50y:+.2f} °C", f"{slope100_100y:+.2f} °C"],
    "MAE (평균 절대 오차)": [f"{mae_all:.4f} (전체 평가)", f"{mae_50y:.4f}", f"{mae_100y:.4f}"],
    "MSE (평균 제곱 오차)": [f"{mse_all:.4f} (전체 평가)", f"{mse_50y:.4f}", f"{mse_100y:.4f}"],
    "R² (결정계수)": [f"{r2_all:.4f} (전체 평가)", f"{r2_50y:.4f}", f"{r2_100y:.4f}"]
}

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.info("💡 **참고**: MAE/MSE는 낮을수록, R²는 1에 가까울수록 테스트 데이터(최근 20년)를 잘 예측함을 의미합니다.")

st.divider()

# 5. 미래 기온 예측 및 연도 선택
st.subheader("🔮 모델별 기온 예측")
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

year_elapsed = np.array([[selected_year - 1908]])
pred_val_all = model_all.predict(year_elapsed)[0]
pred_val_50y = model_50y.predict(year_elapsed)[0]
pred_val_100y = model_100y.predict(year_elapsed)[0]

col1, col2, col3 = st.columns(3)
col1.metric("전체 모델 예측치", f"{pred_val_all:.2f} °C")
col2.metric("과거 50년 모델 예측치", f"{pred_val_50y:.2f} °C")
col3.metric("과거 100년 모델 예측치", f"{pred_val_100y:.2f} °C")

st.divider()

# 6. Plotly 시각화
st.subheader("📈 회귀 직선 및 관측 데이터 비교 그래프")

line_years = np.arange(1900, 2101)
line_X = (line_years - 1908).reshape(-1, 1)

line_Y_all = model_all.predict(line_X)
line_Y_50y = model_50y.predict(line_X)
line_Y_100y = model_100y.predict(line_X)

fig = go.Figure()

# 과거 관측 데이터 (1908 ~ 2005)
train_mask = df_all["연도"] <= 2005
fig.add_trace(go.Scatter(
    x=df_all[train_mask]["연도"],
    y=df_all[train_mask]["연평균기온"],
    mode="markers",
    name="과거 관측 데이터 (1908~2005)",
    marker=dict(color="lightgray", opacity=0.8, size=6)
))

# 테스트 관측 데이터 (2006 ~ 2025)
fig.add_trace(go.Scatter(
    x=test_df["연도"],
    y=test_df["연평균기온"],
    mode="markers",
    name="테스트 데이터 (최근 20년: 2006~2025)",
    marker=dict(color="black", size=8, symbol="square")
))

# 전체 데이터 회귀선
fig.add_trace(go.Scatter(
    x=line_years, y=line_Y_all, mode="lines",
    name=f"전체 모델 ({slope100_all:+.2f}°C/100년)",
    line=dict(color="gray", width=1.5, dash="dot")
))

# 과거 50년 회귀선
fig.add_trace(go.Scatter(
    x=line_years, y=line_Y_50y, mode="lines",
    name=f"과거 50년 모델 ({slope100_50y:+.2f}°C/100년)",
    line=dict(color="firebrick", width=2.5)
))

# 과거 100년 회귀선
fig.add_trace(go.Scatter(
    x=line_years, y=line_Y_100y, mode="lines",
    name=f"과거 100년 모델 ({slope100_100y:+.2f}°C/100년)",
    line=dict(color="royalblue", width=2.5)
))

# 선택한 연도 표시
fig.add_trace(go.Scatter(
    x=[selected_year], y=[pred_val_50y], mode="markers",
    name=f"선택 연도 ({selected_year}년)",
    marker=dict(color="orange", size=12, symbol="star")
))

fig.update_layout(
    title="서울 연도별 기온 회귀선 및 테스트 데이터 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
