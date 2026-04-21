import os
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ✅ (중요) src 폴더 안에서 실행되는 app.py 기준
# - models 폴더: 프로젝트 루트/models
# - 이미지 저장 위치: src/ (analysis.ipynb에서 저장한 png를 src에 둔다)

# --- [0. 경로 설정] ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../Deep-KPT/src
base_dir = os.path.dirname(current_dir)                   # .../Deep-KPT
models_dir = os.path.join(base_dir, "models")             # .../Deep-KPT/models

# ✅ 전처리 함수 import (프로젝트 구조: src/data_loader.py)
# - app.py가 src 폴더 안에 있으므로 "data_loader"로 바로 import 가능
from data_loader import preprocess_gene_data


# -----------------------------
# utils
# -----------------------------
def load_gene_list(models_dir: str):
    """train.py에서 저장한 gene_list.txt 로드"""
    path = os.path.join(models_dir, "gene_list.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return None


def get_image_path(filename: str) -> str | None:
    """
    ✅ 이미지 탐색 우선순위
    1) src/filename (권장: analysis.ipynb에서 savefig를 src로)
    2) 프로젝트 루트/filename (혹시 루트에 저장했을 때 대비)
    3) src/figures/filename (추가 폴더를 쓰고 싶으면 여기에 넣어도 됨)
    """
    candidates = [
        os.path.join(current_dir, filename),
        os.path.join(base_dir, filename),
        os.path.join(current_dir, "figures", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def read_uploaded_file(file):
    """
    업로드 파일을 DataFrame으로 로드
    - Excel: pd.read_excel
    - CSV: pd.read_csv
    """
    name = file.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. (xlsx/csv만 가능)")


def infer_group_from_sample(sample_name: str) -> str:
    """
    ✅ 내부 데이터 그룹 라벨링(발표용)
    - sample에 '12b'가 포함되면 Treated
    - 그렇지 않으면 Control
    """
    s = str(sample_name).upper()
    return "Treated (12b)" if "12B" in s else "Control"


# -----------------------------
# [1. 모델 정의 및 로드]
# -----------------------------
class DeepKPTModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


@st.cache_resource
def load_resources(models_dir: str):
    """
    ✅ 모델 + 학습 유전자 리스트 로드 (한 번만 로드)
    """
    model_path = os.path.join(models_dir, "best_model.pt")
    gene_list = load_gene_list(models_dir)

    if not (os.path.exists(model_path) and gene_list):
        return None, None

    model = DeepKPTModel(input_dim=len(gene_list))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, gene_list


# -----------------------------
# [2. UI 구성]
# -----------------------------
st.set_page_config(page_title="Deep-KPT Analysis", layout="wide")
st.title("🧬 Deep-KPT: 통합 데이터 분석 시스템")

# 리소스 로드
model, trained_genes = load_resources(models_dir)

# --- Sidebar: 업로드 UI ---
st.sidebar.header("📁 데이터 업로드")
my_file = st.sidebar.file_uploader("1. 내 실험 데이터 (Excel/CSV)", type=["xlsx", "csv"])
ext_file = st.sidebar.file_uploader("2. 외부 데이터 (CSV)", type=["csv"])

# --- 필수 파일 점검 ---
if model is None or trained_genes is None:
    st.error(
        "❌ 필수 파일을 찾을 수 없습니다.\n\n"
        f"경로 확인: {models_dir}\n"
        "1) train.py 실행 → models/gene_list.txt 생성\n"
        "2) models/best_model.pt 생성\n"
        "3) (선택) models/scaler_mean.npy, models/scaler_std.npy 생성"
    )
    st.stop()

# --- 파일 업로드 전 안내 ---
if not my_file:
    st.info("좌측 사이드바에서 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

# -----------------------------
# [3. 분석 파이프라인]
# -----------------------------
try:
    # (A) 내부 데이터 로드 & 전처리
    my_df = read_uploaded_file(my_file)
    X_int, samples_int, rep_int = preprocess_gene_data(my_df, trained_genes, models_dir=models_dir)

    st.sidebar.markdown("### ✅ 매칭 리포트(내 데이터)")
    st.sidebar.write(f"- 입력 형태 감지: **{rep_int['detected_format']}**")
    st.sidebar.write(f"- 키 사용: **{rep_int['key_used']}**")
    st.sidebar.write(f"- 매칭 유전자: **{rep_int['matched_genes']} / {rep_int['total_genes']}**")
    st.sidebar.write(f"- 매칭률: **{rep_int['match_ratio']*100:.1f}%**")
    st.sidebar.caption(rep_int["message"])

    with torch.no_grad():
        p_int = model(torch.tensor(X_int, dtype=torch.float32)).view(-1).numpy()

    # 안전장치(샘플 수 불일치 방지)
    if len(p_int) != len(samples_int):
        p_int = np.resize(p_int, len(samples_int))

    res_int = pd.DataFrame({
        "Sample Name": [str(s) for s in samples_int],
        "Sensitivity Score": p_int,
        "Group": [infer_group_from_sample(s) for s in samples_int],
        "Source": "Internal",
    })

    # (B) 외부 데이터 로드 & 전처리(선택)
    res_df = res_int.copy()

    if ext_file:
        ext_df = read_uploaded_file(ext_file)
        X_ext, samples_ext, rep_ext = preprocess_gene_data(ext_df, trained_genes, models_dir=models_dir)

        st.sidebar.markdown("### ✅ 매칭 리포트(외부 데이터)")
        st.sidebar.write(f"- 입력 형태 감지: **{rep_ext['detected_format']}**")
        st.sidebar.write(f"- 키 사용: **{rep_ext['key_used']}**")
        st.sidebar.write(f"- 매칭 유전자: **{rep_ext['matched_genes']} / {rep_ext['total_genes']}**")
        st.sidebar.write(f"- 매칭률: **{rep_ext['match_ratio']*100:.1f}%**")
        st.sidebar.caption(rep_ext["message"])

        with torch.no_grad():
            p_ext = model(torch.tensor(X_ext, dtype=torch.float32)).view(-1).numpy()

        if len(p_ext) != len(samples_ext):
            p_ext = np.resize(p_ext, len(samples_ext))

        res_ext = pd.DataFrame({
            "Sample Name": [f"Ext_{s}" for s in samples_ext],
            "Sensitivity Score": p_ext,
            "Group": ["External (Benchmark)"] * len(samples_ext),
            "Source": "External",
        })

        res_df = pd.concat([res_int, res_ext], ignore_index=True)
        st.sidebar.success("✅ 외부 데이터 결합 완료!")

    # -----------------------------
    # [4. 결과 요약 테이블]
    # -----------------------------
    st.subheader("✅ 1. 약물 반응성 예측 결과 요약")

    # 발표/가독성용 정렬: 점수 높은 순
    show_df = res_df.sort_values("Sensitivity Score", ascending=False).reset_index(drop=True)

    st.dataframe(
        show_df.style.background_gradient(cmap="Reds", subset=["Sensitivity Score"]),
        use_container_width=True
    )

    # -----------------------------
    # [5. 상세 시각화 리포트 (이미지 로딩)]
    # -----------------------------
    st.markdown("---")
    st.subheader("📊 2. 상세 시각화 리포트")

    # ✅ 네가 analysis.ipynb에서 만든 "5종 세트" 파일명 기준으로 탭 구성
    # (파일명만 맞춰서 src에 저장해두면 자동으로 뜸)
    tabs = st.tabs([
        "Dist (Internal vs External)",
        "Internal Group Diff",
        "Ranked Predictions",
        "Latent PCA",
        "Top Gene Importance",
    ])

    # ✅ 여기 파일명만 네 실제 저장 파일명과 동일하게 유지하면 됨
    viz_map = {
        "Dist (Internal vs External)": "viz01_dist_internal_vs_external.png",
        "Internal Group Diff": "viz02_internal_group_box.png",
        "Ranked Predictions": "viz03_samplewise_rank_scatter.png",
        "Latent PCA": "viz04_latent_pca_internal_vs_external.png",
        "Top Gene Importance": "viz05_top_genes_importance.png",
    }

    # 탭 순서대로 렌더
    tab_keys = list(viz_map.keys())
    for tab, key in zip(tabs, tab_keys):
        with tab:
            fname = viz_map[key]
            img_path = get_image_path(fname)

            # ✅ 이미지가 없으면, 어디를 찾았는지까지 안내(디버깅 편하게)
            if img_path is None:
                st.warning(
                    f"'{fname}' 파일을 찾지 못했습니다.\n\n"
                    "✅ 해결 방법:\n"
                    f"- 가장 추천: src 폴더에 '{fname}' 저장\n"
                    f"- 또는 src/figures 폴더에 '{fname}' 저장\n\n"
                    f"(현재 src 경로: {current_dir})"
                )
            else:
                st.image(img_path, use_container_width=True)
                # 발표용으로 파일명 표시(원하면 지워도 됨)
                st.caption(f"Loaded: {os.path.basename(img_path)}")

except Exception as e:
    st.error(f"분석 중 오류 발생: {e}")
