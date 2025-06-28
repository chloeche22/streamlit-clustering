"""dashboard_app.py – Streamlit 국회 표결 클러스터링 분석 전문 페이지
* Analytics 전용: 단일 Analysis 페이지
* .env 환경변수 적용 (DB, Azure 키)
* 세부 진행 상태 표시(progress bar & messages)
* UI 순서: 1.PCA 시각화 → 2.표결 내역 → 3.클러스터 요약 → 4.TTS
* 사이드바: 해석
* 페이지 하단 저작권 표시
작성: 2025‑06‑28 (수정: Sidebar Markdown 오류 해결)"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')
warnings.filterwarnings('ignore', message='Graph is not fully connected*')

import os
from datetime import datetime
from typing import Dict, Tuple, List

# 환경변수: Streamlit Secrets 사용 (환경변수 파일 의존 제거)
# Ensure to set secrets in .streamlit/secrets.toml or Streamlit Cloud settings

import pyodbc
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ───────────────────── DB 로드 함수 ───────────────────── #

def get_db_connection() -> pyodbc.Connection | None:
    """환경변수 기반 DB 연결 생성"""
    try:
        driver = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
        server = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        uid = os.getenv('DB_USER')
        pwd = os.getenv('DB_PASSWORD')
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={database};"
            f"UID={uid};PWD={pwd};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        return None

@st.cache_data(ttl=86400)
def load_vote_matrix() -> pd.DataFrame:
    conn = get_db_connection()
    if not conn:
        st.stop()
    try:
        df = pd.read_sql(
            "SELECT MEMBER_NO, BILL_NO, RESULT_VOTE_MOD"
            " FROM assembly_plenary_session_vote"
            " WHERE RESULT_VOTE_MOD IS NOT NULL",
            conn
        )
        return df.pivot(index="MEMBER_NO", columns="BILL_NO", values="RESULT_VOTE_MOD")
    finally:
        conn.close()

@st.cache_data(ttl=86400)
def load_vote_df() -> pd.DataFrame:
    conn = get_db_connection()
    if not conn:
        st.stop()
    try:
        return pd.read_sql(
            "SELECT MEMBER_NO, HG_NM, POLY_NM, BILL_NO, RESULT_VOTE_MOD"
            " FROM assembly_plenary_session_vote"
            " WHERE RESULT_VOTE_MOD IS NOT NULL",
            conn
        )
    finally:
        conn.close()

# ───────────────────── 클러스터링 함수 ───────────────────── #

def run_clustering(
    matrix: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, float], pd.DataFrame]:
    filled = matrix.fillna("미투표")
    le = LabelEncoder()
    encoded = filled.apply(le.fit_transform)

    algos = {
        "KMeans": KMeans(n_clusters=k, random_state=42, n_init="auto"),
        "DBSCAN": DBSCAN(eps=3, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=k),
        "GMM": GaussianMixture(n_components=k, random_state=42),
        "Spectral": SpectralClustering(n_clusters=k, assign_labels="kmeans", random_state=42),
    }

    results: Dict[str, np.ndarray] = {}
    scores: Dict[str, float] = {}
    counts: Dict[str, pd.Series] = {}

    for name, model in algos.items():
        try:
            labels = model.fit_predict(encoded)
            results[name] = labels
            if len(set(labels)) > 1:
                scores[name] = silhouette_score(encoded, labels)
            counts[name] = pd.Series(labels).value_counts().sort_index()
        except Exception:
            continue

    summary = pd.DataFrame(counts).fillna(0).astype(int)
    summary.index = [f"Cluster {i}" for i in summary.index]
    return encoded, results, scores, summary

# ───────────────────── Streamlit 헬퍼 ───────────────────── #

def plot_pca_scatter(
    encoded: pd.DataFrame,
    clusters: np.ndarray,
    party: pd.Series,
    names: pd.Series,
    title: str
) -> go.Figure:
    pca = PCA(n_components=2)
    comps = pca.fit_transform(encoded)
    df_plot = pd.DataFrame({
        'MEMBER_NO': encoded.index,
        'PCA1': comps[:, 0],
        'PCA2': comps[:, 1],
        'Cluster': clusters,
        'Party': party.loc[encoded.index].values,
        'Name': names.loc[encoded.index].values
    })
    shapes = ['circle','square','triangle-up','diamond','cross']
    fig = px.scatter(
        df_plot,
        x='PCA1', y='PCA2',
        color='Party', symbol='Cluster', symbol_sequence=shapes,
        hover_data=['MEMBER_NO','Name','Party','Cluster'],
        title=title, template='plotly_white'
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    return fig

# ───────────────────── Streamlit 애플리케이션 ───────────────────── #

def main():
    st.set_page_config(page_title="국회 표결 클러스터링 분석", layout="wide")
    st.title("🏛️ 국회 표결 클러스터링 분석")

    # 사이드바: 해석
    st.sidebar.header("🔍 분석 해석")
    st.sidebar.markdown(
        """
- **Silhouette Score**: 군집 간 거리 명확성을 나타내는 지표로, 값이 높을수록 품질이 우수합니다.
- **PCA**(주성분 분석): 고차원 데이터를 주요 특징 축으로 축소하여 시각화합니다.
- **DBSCAN**: 밀도 기반 클러스터링으로, 데이터 밀집 지역을 그룹으로 식별합니다.
- **GMM**: 가우시안 혼합 모델로, 각 군집을 확률 분포로 모델링합니다.
        """,
        unsafe_allow_html=False
    )

    # 클러스터링 설정 및 실행
    k = st.slider("클러스터 개수 선택", min_value=2, max_value=6, value=3)
    if not st.button("분석 실행"): return

    # 진행 상태 표시
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text("1/4 DB 연결 중…")
    progress_bar.progress(10)
    vote_matrix = load_vote_matrix()

    progress_text.text("2/4 DB에서 데이터 가져오는 중…")
    progress_bar.progress(40)
    vote_df = load_vote_df()
    party = vote_df[['MEMBER_NO','POLY_NM']].drop_duplicates().set_index('MEMBER_NO')['POLY_NM']
    names = vote_df[['MEMBER_NO','HG_NM']].drop_duplicates().set_index('MEMBER_NO')['HG_NM']

    progress_text.text("3/4 군집화 실행 중…")
    progress_bar.progress(70)
    encoded, results, scores, summary = run_clustering(vote_matrix, k)

    progress_text.text("4/4 표결 내용 준비 중…")
    progress_bar.progress(90)
    progress_bar.progress(100)
    progress_text.text("분석 완료!")

    # 최적 알고리즘 표시
    if scores:
        best = max(scores, key=scores.get)
        st.info(f"최적 알고리즘: **{best}** (Silhouette: {scores[best]:.3f})")

    # 1. PCA 2D 클러스터 시각화
    st.subheader("1. PCA 2D 클러스터 시각화")
    fig = plot_pca_scatter(encoded, results[best], party, names, title=f"{best} (k={k})")
    st.plotly_chart(fig, use_container_width=True)

    # 2. 클러스터별 의원 표결 내역
    st.subheader("2. 클러스터별 의원 표결 내역")
    cluster_series = pd.Series(results[best], index=vote_matrix.index, name='Cluster')
    df_full = vote_df[['HG_NM','MEMBER_NO','BILL_NO','RESULT_VOTE_MOD']].merge(
        cluster_series.reset_index().rename(columns={'index':'MEMBER_NO'}),
        on='MEMBER_NO', how='left'
    )
    clusters_list = sorted(df_full['Cluster'].unique())
    sel_cluster = st.selectbox("클러스터 선택", clusters_list)
    df_sel = df_full[df_full['Cluster'] == sel_cluster]
    st.dataframe(df_sel[['HG_NM','MEMBER_NO','BILL_NO','RESULT_VOTE_MOD']])

    # 3. 클러스터 그룹 요약
    st.subheader("3. 클러스터 그룹 요약")
    st.table(summary)
    st.markdown("**클러스터별 의원 수 분포 (알고리즘별)**")
    summary_reset = summary.reset_index().rename(columns={'index':'Cluster'})
    summary_melt = summary_reset.melt(id_vars='Cluster', var_name='Algorithm', value_name='Count')
    fig_summary = px.bar(
        summary_melt,
        x='Cluster', y='Count', color='Algorithm', barmode='group',
        title='클러스터 그룹 요약 - 알고리즘별 분포', template='plotly_white'
    )
    st.plotly_chart(fig_summary, use_container_width=True)

    # 4. 음성 설명 (TTS)
    st.subheader("4. 음성 설명 (TTS)")
    try:
        import azure.cognitiveservices.speech as speechsdk
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION", "koreacentral")
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_synthesis_voice_name = "ko-KR-SunHiNeural"

        cluster_desc = ". ".join([
            f"클러스터 {i}에는 {count}명의 의원이 있습니다" for i, count in summary[best].items()
        ])
        tts_text = (
            f"최적 알고리즘은 {best}이며, k 값은 {k} 입니다. "
            f"클러스터 그룹 요약 결과, {cluster_desc}. "
            "이상이 분석 결과 요약입니다."
        )

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        response = synthesizer.speak_text_async(tts_text).get()
        if response.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            st.audio(response.audio_data, format='audio/wav')
        else:
            st.warning(f"TTS 실패: {response.reason}")
    except Exception as e:
        st.warning(f"Azure TTS 처리 중 오류: {e}")

            # PDF 저장 (브라우저 Print)
    st.subheader("📄 PDF로 저장")
    st.markdown(
        """
        <style>
          @media print { .no-print { display: none; } }
        </style>
        <button class="no-print" onclick="window.print()">이 페이지를 PDF로 저장</button>
        """,
        unsafe_allow_html=True
    )

        # PDF 저장 버튼 (HTML 컴포넌트 사용)
    import streamlit.components.v1 as components
    st.subheader("📄 PDF로 저장")
    components.html(
        """
        <button onclick="window.print()" style="padding:8px 16px; font-size:16px; cursor:pointer;">
            이 페이지를 PDF로 저장
        </button>
        """,
        height=60
    )

    # 페이지 하단 저작권 표시
    st.markdown(
        """
        <div style='color:gray;font-size:12px;text-align:center;margin-top:2rem;'>
            © 2025 Outliers Team, Sesac Project. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
