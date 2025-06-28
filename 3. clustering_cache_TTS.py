"""dashboard_app.py – Streamlit 국회 표결 클러스터링 분석 전문 페이지
* Analytics 전용: 단일 Analysis 페이지
* Streamlit Secrets 사용 (dotenv 완전 제거)
* .env 관련 import 및 load_dotenv 완전 삭제
* DB 호출, TTS, PDF 저장, Copyright 포함
작성: 2025‑06‑28 (최종 수정 - dotenv 오류 해결)"""

import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')
warnings.filterwarnings('ignore', message='Graph is not fully connected*')

# Streamlit Secrets만 사용 - dotenv 관련 모든 import 제거
import streamlit as st
from datetime import datetime
from typing import Dict, Tuple, List

import pyodbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit.components.v1 as components

# ───────────────────── DB 로드 함수 ───────────────────── #

def get_db_connection() -> pyodbc.Connection | None:
    """Streamlit Secrets 기반 DB 연결 생성"""
    try:
        # Streamlit Secrets에서 직접 가져오기
        driver = st.secrets['DB_DRIVER']
        server = st.secrets['DB_HOST']
        database = st.secrets['DB_NAME']
        uid = st.secrets['DB_USER']
        pwd = st.secrets['DB_PASSWORD']
        
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};DATABASE={database};"
            f"UID={uid};PWD={pwd};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )
        return pyodbc.connect(conn_str)
    except KeyError as e:
        st.error(f"Streamlit Secrets에서 키를 찾을 수 없습니다: {e}")
        st.info("Streamlit Cloud의 Settings > Secrets에서 다음 키들을 설정하세요: DB_DRIVER, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD")
        return None
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        return None

@st.cache_data(ttl=86400)
def load_vote_matrix() -> pd.DataFrame:
    """투표 매트릭스 로드 (캐시 적용)"""
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
    except Exception as e:
        st.error(f"투표 매트릭스 로드 실패: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=86400)
def load_vote_df() -> pd.DataFrame:
    """투표 데이터프레임 로드 (캐시 적용)"""
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
    except Exception as e:
        st.error(f"투표 데이터 로드 실패: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ───────────────────── 클러스터링 함수 ───────────────────── #

def run_clustering(
    matrix: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, float], pd.DataFrame]:
    """다양한 클러스터링 알고리즘 실행"""
    if matrix.empty:
        return pd.DataFrame(), {}, {}, pd.DataFrame()
    
    # 결측값 처리 및 인코딩
    filled = matrix.fillna("미투표")
    le = LabelEncoder()
    encoded = filled.apply(le.fit_transform)

    # 클러스터링 알고리즘 정의
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

    # 각 알고리즘 실행
    for name, model in algos.items():
        try:
            labels = model.fit_predict(encoded)
            results[name] = labels
            
            # Silhouette Score 계산 (클러스터가 1개 이상일 때만)
            unique_labels = len(set(labels))
            if unique_labels > 1:
                scores[name] = silhouette_score(encoded, labels)
            else:
                scores[name] = -1  # 클러스터가 1개면 낮은 점수
                
            counts[name] = pd.Series(labels).value_counts().sort_index()
        except Exception as e:
            st.warning(f"{name} 알고리즘 실행 실패: {e}")
            continue

    # 결과 요약 테이블 생성
    summary = pd.DataFrame(counts).fillna(0).astype(int)
    summary.index = [f"Cluster {i}" for i in summary.index]
    
    return encoded, results, scores, summary

# ───────────────────── Streamlit 헬퍼 함수 ───────────────────── #

def plot_pca_scatter(
    encoded: pd.DataFrame,
    clusters: np.ndarray,
    party: pd.Series,
    names: pd.Series,
    title: str
) -> go.Figure:
    """PCA 2D 산점도 생성"""
    if encoded.empty:
        return go.Figure()
    
    # PCA 차원 축소
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(encoded)
    
    # 플롯용 데이터프레임 생성
    df_plot = pd.DataFrame({
        'MEMBER_NO': encoded.index,
        'PCA1': comps[:, 0],
        'PCA2': comps[:, 1],
        'Cluster': clusters.astype(str),
        'Party': party.loc[encoded.index].values,
        'Name': names.loc[encoded.index].values
    })
    
    # 클러스터별 마커 모양 정의
    shapes = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'star']
    
    # 산점도 생성
    fig = px.scatter(
        df_plot,
        x='PCA1', y='PCA2',
        color='Party', 
        symbol='Cluster', 
        symbol_sequence=shapes,
        hover_data=['MEMBER_NO', 'Name', 'Party', 'Cluster'],
        title=title, 
        template='plotly_white',
        width=800,
        height=600
    )
    
    # 마커 스타일 조정
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.update_layout(
        xaxis_title=f"PC1 (설명력: {pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 (설명력: {pca.explained_variance_ratio_[1]:.1%})"
    )
    
    return fig

def create_tts_audio(text: str) -> bytes | None:
    """Azure TTS를 사용한 음성 생성"""
    try:
        import azure.cognitiveservices.speech as speechsdk
        
        # Azure Speech 설정 (Streamlit Secrets에서 가져오기)
        speech_key = st.secrets['AZURE_SPEECH_KEY']
        speech_region = st.secrets['AZURE_SPEECH_REGION']
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_synthesis_voice_name = "ko-KR-SunHiNeural"
        
        # 음성 합성
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        response = synthesizer.speak_text_async(text).get()
        
        if response.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return response.audio_data
        else:
            st.warning(f"TTS 실패: {response.reason}")
            return None
            
    except KeyError as e:
        st.warning(f"Azure TTS 설정이 없습니다: {e}")
        st.info("Streamlit Secrets에 AZURE_SPEECH_KEY, AZURE_SPEECH_REGION을 설정하세요.")
        return None
    except ImportError:
        st.warning("Azure Speech SDK가 설치되지 않았습니다. requirements.txt에 'azure-cognitiveservices-speech'를 추가하세요.")
        return None
    except Exception as e:
        st.warning(f"Azure TTS 처리 중 오류: {e}")
        return None

# ───────────────────── Streamlit 메인 애플리케이션 ───────────────────── #

def main():
    """메인 Streamlit 애플리케이션"""
    
    # 페이지 설정
    st.set_page_config(
        page_title="국회 표결 클러스터링 분석", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 제목
    st.title("🏛️ 국회 표결 클러스터링 분석")
    st.markdown("---")

    # 사이드바: 분석 해석 및 설정
    st.sidebar.header("🔍 분석 해석")
    st.sidebar.markdown(
        """
        ### 📊 지표 설명
        - **Silhouette Score**: 군집 간 거리 명확성 (-1 ~ 1, 높을수록 좋음)
        - **PCA**: 고차원 데이터를 2차원으로 축소하여 시각화
        - **KMeans**: 중심점 기반 클러스터링 (일반적으로 가장 안정적)
        - **DBSCAN**: 밀도 기반 클러스터링 (이상치 탐지 가능)
        - **GMM**: 가우시안 혼합 모델 (확률적 클러스터링)
        
        ### 🎯 해석 팁
        - 정당별 색상으로 정치적 성향 파악 가능
        - 클러스터별 마커로 표결 패턴 그룹 확인
        - PCA 축은 주요 표결 특징을 나타냄
        """,
        unsafe_allow_html=False
    )

    # 메인 설정
    col1, col2 = st.columns([1, 3])
    
    with col1:
        k = st.slider("🎯 클러스터 개수 선택", min_value=2, max_value=8, value=3)
        run_analysis = st.button("🚀 분석 실행", type="primary")
    
    with col2:
        if not run_analysis:
            st.info("좌측에서 클러스터 개수를 선택하고 '분석 실행' 버튼을 클릭하세요.")
            return

    # 분석 실행
    with st.spinner("분석 진행 중..."):
        # 진행 상태 표시
        progress_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # 1단계: DB 연결 및 데이터 로드
            progress_text.text("1/4 📊 DB에서 투표 매트릭스 로드 중...")
            progress_bar.progress(25)
            vote_matrix = load_vote_matrix()
            
            if vote_matrix.empty:
                st.error("투표 데이터를 로드할 수 없습니다.")
                return

            # 2단계: 추가 데이터 로드
            progress_text.text("2/4 👥 의원 정보 로드 중...")
            progress_bar.progress(50)
            vote_df = load_vote_df()
            
            if vote_df.empty:
                st.error("의원 정보를 로드할 수 없습니다.")
                return
            
            # 의원별 정당 및 이름 정보 추출
            party = vote_df[['MEMBER_NO', 'POLY_NM']].drop_duplicates().set_index('MEMBER_NO')['POLY_NM']
            names = vote_df[['MEMBER_NO', 'HG_NM']].drop_duplicates().set_index('MEMBER_NO')['HG_NM']

            # 3단계: 클러스터링 실행
            progress_text.text("3/4 🔍 클러스터링 알고리즘 실행 중...")
            progress_bar.progress(75)
            encoded, results, scores, summary = run_clustering(vote_matrix, k)
            
            if not results:
                st.error("클러스터링을 실행할 수 없습니다.")
                return

            # 4단계: 완료
            progress_text.text("4/4 ✅ 분석 완료!")
            progress_bar.progress(100)
            
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            return
        finally:
            progress_text.empty()
            progress_bar.empty()

    # 결과 표시
    st.success(f"총 {len(vote_matrix)} 명의 의원, {len(vote_matrix.columns)} 개의 법안에 대한 분석 완료!")

    # 최적 알고리즘 표시
    if scores:
        best_algo = max(scores, key=scores.get)
        best_score = scores[best_algo]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 최적 알고리즘", best_algo)
        with col2:
            st.metric("📈 Silhouette Score", f"{best_score:.3f}")
        with col3:
            st.metric("🎯 클러스터 수", k)
    else:
        st.warning("유효한 클러스터링 결과가 없습니다.")
        return

    st.markdown("---")

    # 1. PCA 2D 클러스터 시각화
    st.subheader("1. 📊 PCA 2D 클러스터 시각화")
    
    # 알고리즘 선택 옵션
    selected_algo = st.selectbox(
        "시각화할 알고리즘 선택:", 
        list(results.keys()),
        index=list(results.keys()).index(best_algo) if best_algo in results else 0
    )
    
    if selected_algo in results:
        fig = plot_pca_scatter(
            encoded, 
            results[selected_algo], 
            party, 
            names, 
            title=f"{selected_algo} 클러스터링 결과 (k={k}, Score: {scores.get(selected_algo, 'N/A')})"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # 2. 클러스터별 의원 표결 내역
    st.subheader("2. 📋 클러스터별 의원 표결 내역")
    
    cluster_series = pd.Series(results[best_algo], index=vote_matrix.index, name='Cluster')
    df_full = vote_df[['HG_NM', 'MEMBER_NO', 'POLY_NM', 'BILL_NO', 'RESULT_VOTE_MOD']].merge(
        cluster_series.reset_index().rename(columns={'index': 'MEMBER_NO'}),
        on='MEMBER_NO', how='left'
    )
    
    clusters_list = sorted(df_full['Cluster'].dropna().unique())
    
    col1, col2 = st.columns([1, 2])
    with col1:
        sel_cluster = st.selectbox("🎯 클러스터 선택:", clusters_list)
    
    with col2:
        df_sel = df_full[df_full['Cluster'] == sel_cluster]
        st.metric("👥 해당 클러스터 의원 수", len(df_sel['MEMBER_NO'].unique()))
    
    # 클러스터별 의원 목록 및 표결 내역
    if not df_sel.empty:
        # 의원 목록
        members_in_cluster = df_sel[['HG_NM', 'POLY_NM', 'MEMBER_NO']].drop_duplicates()
        st.write(f"**클러스터 {sel_cluster} 의원 목록:**")
        st.dataframe(members_in_cluster, use_container_width=True)
        
        # 표결 내역 샘플
        st.write(f"**클러스터 {sel_cluster} 표결 내역 (최신 100건):**")
        sample_votes = df_sel[['HG_NM', 'BILL_NO', 'RESULT_VOTE_MOD']].tail(100)
        st.dataframe(sample_votes, use_container_width=True)
    
    st.markdown("---")

    # 3. 알고리즘별 성능 비교
    st.subheader("3. 📈 알고리즘별 성능 비교")
    
    if scores:
        # 성능 점수 바차트
        scores_df = pd.DataFrame(list(scores.items()), columns=['Algorithm', 'Silhouette_Score'])
        fig_scores = px.bar(
            scores_df,
            x='Algorithm', y='Silhouette_Score',
            title='알고리즘별 Silhouette Score 비교',
            template='plotly_white',
            color='Silhouette_Score',
            color_continuous_scale='viridis'
        )
        fig_scores.update_layout(showlegend=False)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    st.markdown("---")

    # 4. 클러스터 그룹 요약
    st.subheader("4. 📊 클러스터 그룹 요약")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**알고리즘별 클러스터 분포:**")
        st.dataframe(summary, use_container_width=True)
    
    with col2:
        # 클러스터 분포 시각화
        if not summary.empty:
            summary_reset = summary.reset_index().rename(columns={'index': 'Cluster'})
            summary_melt = summary_reset.melt(id_vars='Cluster', var_name='Algorithm', value_name='Count')
            
            fig_summary = px.bar(
                summary_melt,
                x='Cluster', y='Count', color='Algorithm', barmode='group',
                title='클러스터별 의원 수 분포', 
                template='plotly_white'
            )
            st.plotly_chart(fig_summary, use_container_width=True)
    
    st.markdown("---")

    # 5. 음성 설명 (TTS)
    st.subheader("5. 🔊 음성 설명 (TTS)")
    
    # TTS 텍스트 생성
    cluster_desc = ". ".join([
        f"클러스터 {i}에는 {count}명의 의원이 있습니다" 
        for i, count in summary[best_algo].items() if count > 0
    ])
    
    tts_text = (
        f"국회 표결 클러스터링 분석 결과를 말씀드리겠습니다. "
        f"총 {k}개의 클러스터로 분석했으며, "
        f"최적 알고리즘은 {best_algo}로 실루엣 점수는 {best_score:.3f}입니다. "
        f"{cluster_desc}. "
        "이상이 분석 결과 요약입니다."
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🎵 음성 생성"):
            with st.spinner("음성 생성 중..."):
                audio_data = create_tts_audio(tts_text)
                if audio_data:
                    st.success("음성 생성 완료!")
                    st.audio(audio_data, format='audio/wav')
    
    with col2:
        st.write("**음성 내용 미리보기:**")
        st.text_area("TTS 텍스트", tts_text, height=100, disabled=True)
    
    st.markdown("---")

    # 6. PDF 저장 기능
    st.subheader("6. 📄 보고서 저장")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PDF로 저장:**")
        components.html(
            """
            <button onclick="window.print()" 
                    style="padding:10px 20px; font-size:16px; cursor:pointer; 
                           background-color:#ff6b6b; color:white; border:none; 
                           border-radius:5px;" 
                    class="no-print">
                📄 이 페이지를 PDF로 저장
            </button>
            <style>
                @media print {
                    .no-print { display: none !important; }
                }
            </style>
            """,
            height=80
        )
    
    with col2:
        st.write("**데이터 다운로드:**")
        if st.button("📊 클러스터링 결과 CSV 다운로드"):
            # 결과 데이터 준비
            result_df = pd.DataFrame({
                'MEMBER_NO': vote_matrix.index,
                'NAME': names.loc[vote_matrix.index],
                'PARTY': party.loc[vote_matrix.index],
                'CLUSTER': results[best_algo]
            })
            
            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="CSV 파일 다운로드",
                data=csv,
                file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # 페이지 하단 저작권 및 정보
    st.markdown("---")
    st.markdown(
        """
        <div style='color:gray; font-size:12px; text-align:center; margin-top:2rem; padding:1rem; 
                    background-color:#f8f9fa; border-radius:10px;'>
            <p><strong>🏛️ 국회 표결 클러스터링 분석 시스템</strong></p>
            <p>© 2025 Outliers Team, Sesac Project. All rights reserved.</p>
            <p>데이터 출처: 국회 본회의 표결 데이터 | 분석 도구: Python, Streamlit, Scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()