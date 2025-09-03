# 필요한 라이브러리 설치
# pip install transformers gradio

import gradio as gr
from transformers import pipeline

# 1️⃣ 한국어 감정 분석 파이프라인 생성
# https://huggingface.co/WhitePeak/bert-base-cased-Korean-sentiment
sentiment_pipeline = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")

# 2️⃣ 감정 분석 함수
def analyze_sentiment(review):
    result = sentiment_pipeline(review)[0]  # 결과는 리스트로 반환
    label = result['label']  # "positive" 또는 "negative"
    score = result['score']
    
    # 한글 레이블로 변환
    label_map = {"LABEL_0": "부정", "LABEL_1": "긍정"}
    return f"감정: {label_map.get(label, label)}, 확률: {score:.2f}"

# 3️⃣ Gradio 웹앱 인터페이스
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="영화 리뷰를 입력하세요..."),
    outputs=gr.Textbox(label="분석 결과"),
    title="한글 영화 리뷰 감정 분석",
    description="한국어 영화 리뷰 감정 분석 페이지입니다."
)

if __name__ == "__main__":
    iface.launch(share=True)
