# 필요한 라이브러리 설치
# pip install transformers gradio requests beautifulsoup4 matplotlib plotly pandas selenium

import gradio as gr
from transformers import pipeline
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import random
from urllib.parse import quote
import re
from ptpython.repl import embed

print("🤖 한국어 감정 분석 모델 로딩 중...")
# 1️⃣ 한국어 감정 분석 파이프라인 생성
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    print("대안 모델을 사용합니다...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

print("✅ 모델 로딩 완료!")

# 2️⃣ 네이버 영화 리뷰 크롤링 함수
def crawl_naver_movie_reviews(movie_title, max_reviews=10):
    """네이버 영화에서 리뷰를 크롤링하는 함수"""
    reviews = []
    
    try:

        # embed(globals(), locals())
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # 브라우저 안 띄우기
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=chrome_options)
        # 네이버 영화 검색 URL
        search_url = f"https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&mra=bkEw&pkid=68&os=36885745&qvt=0&query=영화 {quote(movie_title)} 평점"

        driver.get(search_url)
        driver.implicitly_wait(5)

        search_buttons = driver.find_elements(By.CSS_SELECTOR, "button.bt_search")
        if search_buttons:
            search_buttons[0].click()  # 첫 번째 버튼 클릭
        else:
            print("검색 버튼을 찾을 수 없습니다.")

        
        driver.implicitly_wait(5)

        review_elements = driver.find_elements(By.CSS_SELECTOR, ".area_review_content .desc._text")

        # 텍스트 추출(10개만)
        # reviews = [elem.text for elem in review_elements]
        reviews = [elem.text for elem in review_elements[:10]]

        # 결과 출력
        # for i, review in enumerate(reviews, 1):
        #     print(f"{i}: {review}")
                

        return reviews[:max_reviews], f"✅ {len(reviews)}개의 리뷰를 수집했습니다."
        
    except requests.RequestException as e:
        return [], f"❌ 네트워크 오류: {str(e)}"
    except Exception as e:
        return [], f"❌ 크롤링 오류: {str(e)}"

# 3️⃣ 샘플 리뷰 데이터 (크롤링 실패 시 사용)
sample_reviews = {
    "기생충": [
        "정말 충격적이고 인상 깊은 영화였어요. 봉준호 감독의 연출력이 돋보입니다.",
        "사회 계층 간의 갈등을 너무나 현실적으로 그려낸 작품이네요.",
        "예상치 못한 반전과 긴장감이 계속 이어져서 몰입도가 높았습니다.",
        "배우들의 연기력이 정말 뛰어나고 스토리텔링도 완벽해요.",
        "좀 과장된 면이 있긴 하지만 전체적으로는 만족스러운 영화입니다."
    ],
    "타이타닉": [
        "로맨틱한 사랑 이야기와 웅장한 스케일이 인상적이었어요.",
        "레오나르도 디카프리오와 케이트 윈슬렛의 케미가 정말 좋았습니다.",
        "너무 길고 뻔한 스토리라 조금 지루했어요.",
        "특수효과와 음악이 정말 대단하고 감동적이었습니다.",
        "클래식한 멜로 영화의 대표작이라고 할 수 있겠네요."
    ]
}

def get_sample_reviews(movie_title, max_reviews=10):
    """샘플 리뷰 반환 (크롤링 실패 시 대체용)"""
    for key in sample_reviews.keys():
        if key in movie_title or movie_title in key:
            return sample_reviews[key][:max_reviews], f"✅ 샘플 데이터 {len(sample_reviews[key][:max_reviews])}개를 사용합니다."
    
    # 기본 샘플 리뷰
    default_reviews = [
        "정말 재미있는 영화였어요. 추천합니다!",
        "스토리는 괜찮았는데 연출이 아쉬웠어요.",
        "배우들의 연기가 인상적이었습니다.",
        "예상보다 지루했지만 나쁘지 않았어요.",
        "볼만한 영화입니다. 시간 가는 줄 몰랐어요."
    ]
    return default_reviews, "✅ 기본 샘플 데이터를 사용합니다."

# 4️⃣ 감정 분석 함수 개선
def analyze_sentiment_batch(reviews):
    """여러 리뷰를 한번에 감정 분석"""
    results = []
    
    for review in reviews:
        try:
            result = sentiment_pipeline(review)[0]
            label = result['label']
            score = result['score']
            
            # 레이블 정규화
            if label in ["LABEL_1", "POSITIVE", "positive"]:
                sentiment = "긍정"
                emoji = "😊"
            else:
                sentiment = "부정"
                emoji = "😞"

            
            results.append({
                "review": review,
                "sentiment": sentiment,
                "emoji": emoji,
                "score": score,
                "confidence": f"{score:.1%}"
            })
            
        except Exception as e:
            results.append({
                "review": review,
                "sentiment": "오류",
                "emoji": "❓",
                "score": 0.0,
                "confidence": "0%"
            })
    
    return results

# 5️⃣ 시각화 함수
def create_sentiment_chart(results):
    """감정 분석 결과를 파이 차트로 시각화"""
    
    # 감정별 카운트
    sentiment_counts = {"긍정": 0, "부정": 0}
    
    for result in results:
        if result['sentiment'] in sentiment_counts:
            sentiment_counts[result['sentiment']] += 1
    
    # 0인 항목 제거
    sentiment_counts = {k: v for k, v in sentiment_counts.items() if v > 0}
    
    if not sentiment_counts:
        return None
    
    # 색상 설정
    colors = {"긍정": "#4CAF50", "부정": "#F44336"}
    
    # Plotly 파이 차트 생성
    fig = go.Figure(data=[
        go.Pie(
            labels=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            hole=0.4,  # 도넛 차트
            marker=dict(colors=[colors.get(k, "#999") for k in sentiment_counts.keys()]),
            textinfo='label+percent+value',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>개수: %{value}<br>비율: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': f"🎬 감정 분석 결과 (총 {sum(sentiment_counts.values())}개 리뷰)",
            'x': 0.5,
            'font': {'size': 18}
        },
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True,
        height=400,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig

# 6️⃣ 결과 테이블 생성 함수
def create_results_table(results):
    """분석 결과를 테이블 형태로 정리"""
    
    table_data = []
    for i, result in enumerate(results, 1):
        table_data.append([
            f" {i}",
            result['review'][:50] + ("..." if len(result['review']) > 50 else ""),
            f"{result['emoji']} {result['sentiment']}",
            result['confidence']
        ])
    
    return table_data

# 7️⃣ 메인 분석 함수
def analyze_movie_reviews(movie_title, max_reviews=10):
    """영화 리뷰 수집 및 감정 분석 메인 함수"""
    
    if not movie_title.strip():
        return "❓ 영화 제목을 입력해주세요.", None, None
    
    # 진행 상태 표시
    status_msg = f"🔍 '{movie_title}' 영화 리뷰 검색 중..."
    
    try:
        # 1단계: 리뷰 수집
        reviews, crawl_msg = crawl_naver_movie_reviews(movie_title, max_reviews)
        
        # 크롤링 실패 시 샘플 데이터 사용
        if not reviews:
            reviews, crawl_msg = get_sample_reviews(movie_title, max_reviews)
        
        if not reviews:
            return "❌ 리뷰를 찾을 수 없습니다. 다른 영화 제목을 시도해보세요.", None, None
        
        # 2단계: 감정 분석
        status_msg += f"\n🤖 {len(reviews)}개 리뷰 감정 분석 중..."
        results = analyze_sentiment_batch(reviews)
        
        # 3단계: 시각화
        chart = create_sentiment_chart(results)
        
        # 4단계: 결과 테이블
        table = create_results_table(results)
        
        # 5단계: 요약 메시지 생성
        positive_count = sum(1 for r in results if r['sentiment'] == '긍정')
        negative_count = sum(1 for r in results if r['sentiment'] == '부정')
        
        summary = f"""📊 '{movie_title}' 리뷰 감정 분석 결과

        {crawl_msg}

        📈 분석 결과:
        • 😊 긍정: {positive_count}개 ({positive_count/len(results)*100:.1f}%)
        
        • 😞 부정: {negative_count}개 ({negative_count/len(results)*100:.1f}%)  

        💡 종합 평가: {'긍정적' if positive_count > negative_count else '부정적'} 반응"""
        
        return summary, chart, table
        
    except Exception as e:
        return f"❌ 처리 중 오류가 발생했습니다: {str(e)}", None, None

# 8️⃣ Gradio 인터페이스 구성
def create_app():
    
    with gr.Blocks(
        title="🎬 영화 리뷰 감정 분석기",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🎬 영화 리뷰 감정 분석기
        
        **영화 제목을 입력하면 실제 사용자 리뷰를 수집하여 AI로 감정을 분석하고 결과를 시각적으로 보여드립니다.**
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                movie_input = gr.Textbox(
                    label="🎥 영화 제목",
                    placeholder="예: 좀비딸, 기생충, 타이타닉...",
                    lines=1
                )
                
                review_count = gr.Slider(
                    label="📊 분석할 리뷰 개수",
                    minimum=5,
                    maximum=10,
                    value=10,
                    step=1
                )
                
                analyze_btn = gr.Button("🔍 분석 시작", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 사용 팁
                - 분석 완료까지 10-30초 소요됩니다.
                """)
        
        # 결과 출력 영역
        with gr.Row():
            with gr.Column(scale=1):
                result_text = gr.Textbox(
                    label="📊 분석 결과 요약",
                    lines=8,
                    max_lines=12
                )
            
            with gr.Column(scale=1):
                sentiment_chart = gr.Plot(label="📈 감정 분포 차트")
        
        # 상세 결과 테이블
        gr.Markdown("**📝 리뷰별 상세 분석 결과**")
        result_table = gr.Dataframe(
            headers=["순번", "리뷰 내용", "감정", "신뢰도"],
            datatype=["str", "str", "str", "str"],
            wrap=True,
            interactive=False,
            row_count=(1, "dynamic") 
        )

        # 이벤트 핸들러
        analyze_btn.click(
            analyze_movie_reviews,
            inputs=[movie_input, review_count],
            outputs=[result_text, sentiment_chart, result_table]
        )
        
        movie_input.submit(
            analyze_movie_reviews,
            inputs=[movie_input, review_count],
            outputs=[result_text, sentiment_chart, result_table]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ 안내사항
        - 분석 결과는 참고용으로만 사용하세요
        - 웹 크롤링 시 해당 사이트의 이용약관을 준수합니다
        """)
    
    return app

# 앱 실행
if __name__ == "__main__":
    print("🚀 영화 리뷰 감정 분석기 시작...")
    app = create_app()
    app.launch(
        share=True,  # 외부 접근 허용 (필요 시 False로 변경)
        show_error=True
    )