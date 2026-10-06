import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 페이지 기본 설정 (앱 실행 시 무조건 최상단에서 1회만 실행되어야 함)
st.set_page_config(page_title="서울 연평균 기온 예측 및 회귀 분석", layout="wide")
st.title("🌡️ 서울 연평균 기온 회귀 분석 및 예측기")

# 2. 데이터 불러오기 및 전처리 (공통 함수)
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

# 3. 탭 분리 구성
tab1, tab2 = st.tabs(["📏 1. 선형 회귀 모델 평가 (선형/기간별 비교)", "📈 2. 다항 회귀 모델 비교 (1차, 3차, 9차 곡선)"])

# ==========================================
# TAB 1: 선형 회귀 모델 평가 (기간별 비교)
# ==========================================
with tab1:
    st.header("선형 회귀 모델 평가 및 온난화 속도 비교")
    
    def train_and_eval(train_df, test_df):
        X_train = train_df[["경과연수"]]
        y_train = train_df["연평균기온"]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        slope = model.coef_[0]
        slope_100y = slope * 100
        intercept = model.intercept_
        
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

    # 데이터셋 정의 (최근 20년 공통 테스트)
    test_df_lin = df_all[(df_all["연도"] >= 2006) & (df_all["연도"] <= 2025)]
    train_50y = df_all[(df_all["연도"] >= 1956) & (df_all["연도"] <= 2005)]
    train_100y = df_all[(df_all["연도"] >= 1906) & (df_all["연도"] <= 2005)]

    # 모델 학습 및 평가
    model_all, slope_all, slope100_all, ic_all, _, _, _ = train_and_eval(df_all, None)
    model_50y, slope_50y, slope100_50y, ic_50y, mae_50y, mse_50y, r2_50y = train_and_eval(train_50y, test_df_lin)
    model_100y, slope_100y, slope100_100y, ic_100y, mae_100y, mse_100y, r2_100y = train_and_eval(train_100y, test_df_lin)

    # 전체 평가
    y_pred_all = model_all.predict(df_all[["경과연수"]])
    mae_all = mean_absolute_error(df_all["연평균기온"], y_pred_all)
    mse_all = mean_squared_error(df_all["연평균기온"], y_pred_all)
    r2_all = r2_score(df_all["연평균기온"], y_pred_all)

    # 성능 비교표
    st.subheader("📊 모델별 기울기 및 테스트 데이터(2006~2025년) 예측 성능 비교")
    summary_data = {
        "모델 구분": ["전체 데이터 (전체 학습)", "과거 50년 학습 (1956~2005)", "과거 100년 학습 (1906~2005)"],
        "학습 기간": [f"{df_all['연도'].min()}~{df_all['연도'].max()}", "1956~2005 (50년)", "1906~2005 (100년)"],
        "학습 데이터 개수": [f"{len(df_all)}개", f"{len(train_50y)}개", f"{len(train_100y)}개"],
        "100년당 기온 상승량": [f"{slope100_all:+.2f} °C", f"{slope100_50y:+.2f} °C", f"{slope100_100y:+.2f} °C"],
        "MAE (평균 절대 오차)": [f"{mae_all:.4f} (전체)", f"{mae_50y:.4f}", f"{mae_100y:.4f}"],
        "MSE (평균 제곱 오차)": [f"{mse_all:.4f} (전체)", f"{mse_50y:.4f}", f"{mse_100y:.4f}"],
        "R² (결정계수)": [f"{r2_all:.4f} (전체)", f"{r2_50y:.4f}", f"{r2_100y:.4f}"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.divider()

    # 연도 슬라이더 (key 지정으로 중복 문제 해결)
    st.subheader("🔮 선형 모델 연도별 예측 기온")
    selected_year_lin = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1, key="slider_tab1")

    year_elapsed_lin = np.array([[selected_year_lin - 1908]])
    pred_val_all = model_all.predict(year_elapsed_lin)[0]
    pred_val_50y = model_50y.predict(year_elapsed_lin)[0]
    pred_val_100y = model_100y.predict(year_elapsed_lin)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 모델 예측치", f"{pred_val_all:.2f} °C")
    col2.metric("과거 50년 모델 예측치", f"{pred_val_50y:.2f} °C")
    col3.metric("과거 100년 모델 예측치", f"{pred_val_100y:.2f} °C")

    # 시각화
    line_years = np.arange(1900, 2101)
    line_X = (line_years - 1908).reshape(-1, 1)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_all[df_all["연도"] <= 2005]["연도"], y=df_all[df_all["연도"] <= 2005]["연평균기온"], mode="markers", name="과거 관측 데이터", marker=dict(color="lightgray", opacity=0.8, size=6)))
    fig1.add_trace(go.Scatter(x=test_df_lin["연도"], y=test_df_lin["연평균기온"], mode="markers", name="테스트 데이터 (2006~2025)", marker=dict(color="black", size=8, symbol="square")))
    fig1.add_trace(go.Scatter(x=line_years, y=model_all.predict(line_X), mode="lines", name=f"전체 모델 ({slope100_all:+.2f}°C/100년)", line=dict(color="gray", width=1.5, dash="dot")))
    fig1.add_trace(go.Scatter(x=line_years, y=model_50y.predict(line_X), mode="lines", name=f"과거 50년 모델 ({slope100_50y:+.2f}°C/100년)", line=dict(color="firebrick", width=2.5)))
    fig1.add_trace(go.Scatter(x=line_years, y=model_100y.predict(line_X), mode="lines", name=f"과거 100년 모델 ({slope100_100y:+.2f}°C/100년)", line=dict(color="royalblue", width=2.5)))

    fig1.update_layout(title="서울 연도별 선형 회귀 비교 그래프", xaxis_title="연도", yaxis_title="평균기온 (°C)", hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)


# ==========================================
# TAB 2: 다항 회귀 모델 비교 (1차, 3차, 9차 곡선)
# ==========================================
with tab2:
    st.header("다항 회귀 곡선 모델 분석 (1차, 3차, 9차)")
    
    # 데이터 분할 (2005년 미만: 훈련용 / 2005년 이상: 테스트용)
    train_df_poly = df_all[df_all["연도"] < 2005].copy()
    test_df_poly = df_all[df_all["연도"] >= 2005].copy()

    train_count = len(train_df_poly)
    test_count = len(test_df_poly)

    degrees = [1, 3, 9]
    poly_models = {}
    poly_results = []

    X_train_p = train_df_poly["경과연수"].values
    y_train_p = train_df_poly["연평균기온"].values
    X_test_p = test_df_poly["경과연수"].values
    y_test_p = test_df_poly["연평균기온"].values

    year_2050_elapsed = 2050 - 1908

    for deg in degrees:
        coeffs = np.polyfit(X_train_p, y_train_p, deg)
        poly_models[deg] = coeffs
        
        y_pred_test_p = np.polyval(coeffs, X_test_p)
        mae_test_p = mean_absolute_error(y_test_p, y_pred_test_p)
        pred_2050 = np.polyval(coeffs, year_2050_elapsed)
        
        poly_results.append({
            "다항식 차수": f"{deg}차 ({'직선' if deg==1 else '곡선'})",
            "테스트 데이터 평균 오차 (MAE)": f"{mae_test_p:.3f} °C",
            "2050년 예상 기온": f"{pred_2050:.2f} °C"
        })

    col_info1, col_info2 = st.columns(2)
    col_info1.metric("훈련용 데이터 연도 수 (2005년 이전)", f"{train_count}개 연도", help=f"{train_df_poly['연도'].min()}년 ~ {train_df_poly['연도'].max()}년")
    col_info2.metric("테스트용 데이터 연도 수 (2005년 이후)", f"{test_count}개 연도", help=f"{test_df_poly['연도'].min()}년 ~ {test_df_poly['연도'].max()}년")

    st.markdown("##### 📊 모델별 테스트 데이터 오차 및 2050년 예측값 비교")
    st.dataframe(pd.DataFrame(poly_results), use_container_width=True, hide_index=True)

    st.warning("⚠️ **주의 (과적합 현상)**: 9차 다항식처럼 차수가 지나치게 높으면 훈련 데이터에는 완벽하게 맞춰지지만, 테스트 데이터나 미래 연도(2050년) 예측 시 극단적으로 왜곡될 수 있습니다.")

    st.divider()

    st.subheader("🔮 연도별 예측 기온 확인")
    selected_year_poly = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1, key="slider_tab2")
    selected_elapsed_poly = selected_year_poly - 1908

    cols_poly = st.columns(3)
    for idx, deg in enumerate(degrees):
        pred_val_p = np.polyval(poly_models[deg], selected_elapsed_poly)
        cols_poly[idx].metric(
            label=f"{deg}차 모델 ({selected_year_poly}년)",
            value=f"{pred_val_p:.2f} °C"
        )

    st.divider()

    # 시각화
    line_years_p = np.arange(1900, 2101)
    line_X_p = line_years_p - 1908

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train_df_poly["연도"], y=train_df_poly["연평균기온"], mode="markers", name="훈련 데이터 (<2005년)", marker=dict(color="steelblue", opacity=0.7, size=6)))
    fig2.add_trace(go.Scatter(x=test_df_poly["연도"], y=test_df_poly["연평균기온"], mode="markers", name="테스트 데이터 (>=2005년)", marker=dict(color="crimson", size=8, symbol="diamond")))

    colors_p = {1: "limegreen", 3: "orange", 9: "purple"}
    for deg in degrees:
        line_Y_p = np.polyval(poly_models[deg], line_X_p)
        fig2.add_trace(go.Scatter(x=line_years_p, y=line_Y_p, mode="lines", name=f"{deg}차 회귀 곡선", line=dict(color=colors_p[deg], width=2)))

    y_min = df_all["연평균기온"].min() - 2
    y_max = df_all["연평균기온"].max() + 5

    fig2.update_layout(
        title="서울 연도별 기온 다항회귀 곡선 비교 (1900~2100년)",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        yaxis=dict(range=[y_min, y_max]),
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)
