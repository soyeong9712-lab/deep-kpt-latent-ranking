import os
import re
import numpy as np
import pandas as pd


def _extract_entrez_id(x: str) -> str | None:
    """
    "TSPAN6 (7105)" -> "7105"
    "7105" -> "7105"
    "TSPAN6" -> None
    """
    s = str(x)
    m = re.search(r"\((\d+)\)", s)
    if m:
        return m.group(1)
    # 숫자만 단독인 경우
    if re.fullmatch(r"\d+", s.strip()):
        return s.strip()
    return None


def _extract_symbol(x: str) -> str:
    """
    "TSPAN6 (7105)" -> "TSPAN6"
    "TSPAN6" -> "TSPAN6"
    """
    s = str(x)
    s = s.split("(")[0].strip()
    return s


def _load_scaler(models_dir: str, input_dim: int):
    """
    train.py에서 저장한 mean/std를 로드 (없으면 None)
    """
    mean_path = os.path.join(models_dir, "scaler_mean.npy")
    std_path = os.path.join(models_dir, "scaler_std.npy")

    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)

        if mean.shape[0] == input_dim and std.shape[0] == input_dim:
            # std==0 방지
            std = np.where(std == 0, 1.0, std)
            return mean, std

    return None, None


def preprocess_gene_data(df: pd.DataFrame, gene_list: list[str], models_dir: str = "models"):
    """
    ✅ 목표: 어떤 입력이 오더라도 최종 결과를 (n_samples, 2000)으로 정렬해서 반환

    - gene_list: 학습 때 사용된 2000개 유전자 리스트 (예: "TSPAN6 (7105)")
    - df: 사용자가 업로드한 엑셀/CSV (형태가 다양할 수 있음)

    Returns:
      scaled_data: np.ndarray (n_samples, 2000)
      sample_names: list[str]
      report: dict (매칭률/형태/사용키 등)
    """

    report = {
        "status": "ok",
        "detected_format": None,    # "samples_x_genes" or "genes_x_samples"
        "key_used": None,           # "entrez" or "symbol"
        "matched_genes": 0,
        "total_genes": len(gene_list),
        "match_ratio": 0.0,
        "message": ""
    }

    try:
        if df is None or df.empty:
            raise ValueError("업로드한 데이터가 비어있습니다.")

        # -----------------------------
        # 1) 학습 유전자 리스트: entrez 우선 키 구성
        # -----------------------------
        trained_symbols = [_extract_symbol(g) for g in gene_list]
        trained_entrez = [_extract_entrez_id(g) for g in gene_list]

        has_entrez = sum([1 for x in trained_entrez if x is not None]) > (0.7 * len(gene_list))

        if has_entrez:
            trained_keys = [e if e is not None else s for e, s in zip(trained_entrez, trained_symbols)]
            report["key_used"] = "entrez"
        else:
            trained_keys = trained_symbols
            report["key_used"] = "symbol"

        input_dim = len(trained_keys)

        # -----------------------------
        # 2) 입력 df 형태 자동 감지
        # -----------------------------
        col_entrez = [_extract_entrez_id(c) for c in df.columns]
        col_symbols = [_extract_symbol(c) for c in df.columns]

        overlap_by_entrez = 0
        overlap_by_symbol = 0

        trained_key_set = set(trained_keys)

        for e in col_entrez:
            if e is not None and e in trained_key_set:
                overlap_by_entrez += 1

        for s in col_symbols:
            if s in trained_key_set:
                overlap_by_symbol += 1

        if overlap_by_entrez >= 50 or overlap_by_symbol >= 80:
            detected = "samples_x_genes"
        else:
            detected = "genes_x_samples"

        report["detected_format"] = detected

        # -----------------------------
        # 3) 입력 df에서 (genes x samples) 형태로 통일한 뒤 정렬
        # -----------------------------
        if detected == "samples_x_genes":
            # (samples, genes)
            sample_names = df.index.astype(str).tolist()

            if report["key_used"] == "entrez":
                gene_keys = [(_extract_entrez_id(c) or _extract_symbol(c)) for c in df.columns]
            else:
                gene_keys = [_extract_symbol(c) for c in df.columns]

            gxs = df.copy()
            gxs.columns = gene_keys
            gxs = gxs.T  # (genes, samples)

        else:
            # (genes, samples)
            gene_id_col = None
            gene_symbol_col = None

            for c in df.columns:
                cl = str(c).lower()
                if cl in ["gene_id", "entrez", "entrez_id", "geneid"]:
                    gene_id_col = c
                if cl in ["gene_symbol", "symbol", "gene", "gene_name", "genename"]:
                    gene_symbol_col = c

            if report["key_used"] == "entrez" and gene_id_col is not None:
                key_series = df[gene_id_col].astype(str).str.strip()
            else:
                gene_col = df.columns[0]
                if report["key_used"] == "entrez":
                    key_series = df[gene_col].astype(str).apply(lambda x: _extract_entrez_id(x) or _extract_symbol(x))
                else:
                    key_series = df[gene_col].astype(str).apply(_extract_symbol)

            # 샘플 컬럼 후보: gene_id_col / gene_symbol_col / 첫 컬럼 제외
            drop_cols = set()
            if gene_id_col is not None:
                drop_cols.add(gene_id_col)
            if gene_symbol_col is not None:
                drop_cols.add(gene_symbol_col)

            # 첫 컬럼(대개 gene 컬럼) 제외
            drop_cols.add(df.columns[0])

            sample_cols = [c for c in df.columns if c not in drop_cols]

            if len(sample_cols) == 0:
                raise ValueError("샘플 컬럼을 찾지 못했습니다. (유전자 컬럼 외에 값 컬럼이 없습니다)")

            # ✅✅✅ [최후의 방법] 메타데이터 컬럼을 '이름'으로 강제 제거
            META_COLS_EXACT = {
                "transcript_id", "description", "gene_biotype", "protein_id",
                "hgnc", "mim", "ensembl", "imgt/gene-db", "imgt", "gene-db"
            }
            META_COLS_CONTAINS = [
                "transcript", "description", "biotype", "protein",
                "hgnc", "mim", "ensembl", "imgt"
            ]

            def _is_meta_col(name: str) -> bool:
                s = str(name).strip().lower()
                if s in META_COLS_EXACT:
                    return True
                for kw in META_COLS_CONTAINS:
                    if kw in s:
                        return True
                return False

            sample_cols = [c for c in sample_cols if not _is_meta_col(c)]

            if len(sample_cols) == 0:
                raise ValueError(
                    "샘플 후보 컬럼이 전부 메타데이터(Transcript/Description/HGNC/MIM 등)로 판단되어 제거되었습니다."
                )

            # ✅✅✅ 숫자 컬럼 필터를 매우 빡세게 적용
            tmp = df[sample_cols].apply(pd.to_numeric, errors="coerce")

            valid_ratio = tmp.notna().mean(axis=0)   # 숫자 비율
            std = tmp.std(axis=0, skipna=True)       # 분산
            nuniq = tmp.nunique(dropna=True)         # 유니크 개수

            # 기준: 숫자비율 85% 이상 + 분산>0 + 유니크>=5
            keep_mask = (valid_ratio >= 0.85) & (std > 0) & (nuniq >= 5)

            # ✅✅✅ [FIX] keep_mask는 pandas Series로 유지한 상태에서 index 뽑기
            sample_cols = keep_mask[keep_mask].index.tolist()

            if len(sample_cols) == 0:
                raise ValueError(
                    "발현값(숫자) 샘플 컬럼을 찾지 못했습니다.\n"
                    "- 메타데이터 컬럼 제거 후에도\n"
                    "- 숫자비율(>=85%), 분산(>0), 유니크(>=5) 조건을 만족하는 컬럼이 없습니다.\n"
                    "※ 데이터가 너무 작거나(유전자 수가 적거나), 샘플 값이 상수에 가까우면 발생할 수 있습니다."
                )

            sample_names = [str(c) for c in sample_cols]

            gxs = df[sample_cols].copy()
            gxs.index = key_series.values

            # 숫자 변환
            gxs = gxs.apply(pd.to_numeric, errors="coerce")

            # 중복 gene key 평균
            gxs = gxs.groupby(gxs.index).mean()

        # -----------------------------
        # 4) trained_keys(2000개) 기준으로 강제 정렬 + 매칭률 체크
        # -----------------------------
        gxs = gxs.apply(pd.to_numeric, errors="coerce").fillna(0)

        aligned = gxs.reindex(index=trained_keys).fillna(0)  # (2000, n_samples)

        matched = int((aligned.sum(axis=1) != 0).sum())
        report["matched_genes"] = matched
        report["match_ratio"] = float(matched / input_dim)

        if report["match_ratio"] < 0.30:
            report["status"] = "fail"
            report["message"] = (
                f"유전자 매칭률이 너무 낮습니다. ({matched}/{input_dim}, {report['match_ratio']*100:.1f}%)\n"
                f"입력 데이터의 유전자 키(Entrez/심볼)와 학습 feature가 크게 다릅니다."
            )
            raise ValueError(report["message"])

        # -----------------------------
        # 5) log2 + scaler(transform only)
        # -----------------------------
        X = aligned.T.values  # (n_samples, 2000)
        X = np.log2(X + 1.0)

        mean, std = _load_scaler(models_dir=models_dir, input_dim=input_dim)
        if mean is not None and std is not None:
            X = (X - mean) / std
            report["message"] = "스케일러(mean/std) 로드 완료 → transform 적용"
        else:
            report["message"] = "스케일러 파일이 없어 log2만 적용(추론에서 fit 금지)"

        return X, sample_names, report

    except Exception as e:
        report["status"] = "fail"
        if not report["message"]:
            report["message"] = f"Preprocess Error: {e}"
        raise
