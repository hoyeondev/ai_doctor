# 필요한 라이브러리 설치
# pip install transformers torch gradio

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 1️⃣ 한국어 감정 분석 모델 로드
# https://huggingface.co/WhitePeak/bert-base-cased-Korean-sentiment
model_name = "WhitePeak/bert-base-cased-Korean-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2️⃣ 감정 분석 함수
def analyze_sentiment(review):
    # 토크나이징
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    # 레이블 매핑
    labels = ["부정", "중립", "긍정"]
    max_idx = torch.argmax(probs).item()
    score = probs[0][max_idx].item()
    
    return f"감정: {labels[max_idx]}, 확률: {score:.2f}"

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
