Deep-KPT: NGS 기반 항암 약물 반응 예측 모델

1. Overview:소규모 실험실 내부 RNA-seq 데이터와 대규모 외부 Omics 데이터를 통합하여
항암 약물 반응(Sensitivity)을 예측하고, 모델의 일반화 가능성을 분석한 프로젝트입니다.

2. Key Features
   
내부 데이터 기반 고신뢰 머신러닝 모델 구축 (LOOCV 100% 정확도)
딥러닝 기반 Latent Space 분석 및 외부 데이터 확장
Internal vs External 데이터 분포 비교 분석
샘플별 반응성 Ranking 시스템 구현
Streamlit 기반 시각화 분석 웹 서비스 제공

4. Model Architecture
   
Input: RNA-seq gene expression (~20,000 features)
Model: Fully Connected Deep Neural Network (PyTorch)
Hidden Layers: 512 → 256
Latent Space: 128-dim embedding
Output: Drug sensitivity probability (0~1)

5. Results
   
✔ Machine Learning (Internal Data)
LOOCV 기반 100% classification accuracy
300개 핵심 유전자 feature selection
✔ Deep Learning (Generalization)
외부 데이터 (n=1,684) 확장 분석
확률 분포 기반 반응 스펙트럼 분석
Latent space 기반 내부/외부 분포 차이 확인

핵심 인사이트:

내부 데이터에서는 ML이 더 안정적,
외부 데이터 확장에서는 DL이 더 유리

6. Visualization
   
Predicted probability distribution (Internal vs External)
Control vs Treated 비교
Sample ranking curve
Latent space PCA
Top gene importance analysis

7. System
   
Streamlit 기반 분석 UI
CSV / Excel 업로드 지원
실시간 예측 및 시각화 제공
NGROK 기반 외부 공유 가능

8. Limitations
   
내부 데이터 샘플 수 부족 (n=18)
Internal / External 데이터 분포 차이 (Domain Shift)
딥러닝 모델 확률 편향 (0 or 1 saturation)
실시간 데이터 파이프라인 부재

9. Future Work
    
Domain Adaptation 적용 (DANN / CORAL)
Probability Calibration (Temperature Scaling)
ML + DL Ensemble 전략
Bio pathway 기반 모델 해석 강화
Data pipeline 및 MLOps 구축

10. Tech Stack
    
Python
PyTorch
Scikit-learn
Pandas / NumPy
Streamlit
