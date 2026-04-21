import os
from pyngrok import ngrok

# 1. ngrok 토큰 설정 (본인 계정 토큰 필수)
ngrok.set_auth_token("37XgQVHDe8cIjAwV2Y1vP9WZWK2_33obqsDcxTtCVpWC8umT7")

# 2. Streamlit 실행 (포트 8501)
print("--- 서버 가동 중... 잠시만 기다려주세요 ---")
# 업로드 한도 1GB 적용 (.streamlit/config.toml 또는 여기서 지정)
os.system("streamlit run src/app.py --server.maxUploadSize=1000")

# 3. ngrok 터널 열기
public_url = ngrok.connect(8501)
print(f"✅ 배포 성공! 외부 접속 주소: {public_url}")