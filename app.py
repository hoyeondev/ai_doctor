# 필요한 라이브러리 설치
# pip install transformers gradio requests beautifulsoup4 matplotlib plotly pandas

import gradio as gr
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import random
from urllib.parse import quote
import re

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
        # 네이버 영화 검색 URL
        search_url = f"https://movie.naver.com/movie/search/result.naver?query={quote(movie_title)}&section=movie"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 영화 검색
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 첫 번째 영화 결과의 링크 찾기
        movie_links = soup.find_all('a', href=True)
        movie_code = None
        
        for link in movie_links:
            href = link.get('href', '')
            if '/movie/bi/mi/basic.naver?code=' in href:
                movie_code = href.split('code=')[1].split('&')[0]
                break
        
        if not movie_code:
            return [], "영화를 찾을 수 없습니다."
        
        # 리뷰 페이지 URL
        review_url = f"https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code={movie_code}&type=after"
        
        review_response = requests.get(review_url, headers=headers, timeout=10)
        review_soup = BeautifulSoup(review_response.content, 'html.parser')
        
        # 리뷰 추출
        review_elements = review_soup.find_all('span', {'id': re.compile(r'_filtered_ment_\d+')})
        
        for element in review_elements[:max_reviews]:
            review_text = element.get_text(strip=True)
            if review_text and len(review_text) > 10:  # 너무 짧은 리뷰 제외
                reviews.append(review_text)
        
        if not reviews:
            # 대안: 더 일반적인 리뷰 선택자 시도
            review_elements = review_soup.find_all(['span', 'p'], class_=re.compile(r'comment|review|ment'))
            for element in review_elements[:max_reviews]:
                review_text = element.get_text(strip=True)
                if review_text and len(review_text) > 10:
                    reviews.append(review_text)
        
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
            elif label in ["LABEL_0", "NEGATIVE", "negative"]:
                sentiment = "부정"
                emoji = "😞"
            else:
                sentiment = "중립"
                emoji = "😐"
            
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
    sentiment_counts = {"긍정": 0, "부정": 0, "중립": 0}
    
    for result in results:
        if result['sentiment'] in sentiment_counts:
            sentiment_counts[result['sentiment']] += 1
    
    # 0인 항목 제거
    sentiment_counts = {k: v for k, v in sentiment_counts.items() if v > 0}
    
    if not sentiment_counts:
        return None
    
    # 색상 설정
    colors = {"긍정": "#4CAF50", "부정": "#F44336", "중립": "#FF9800"}
    
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
            f"리뷰 {i}",
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
        neutral_count = sum(1 for r in results if r['sentiment'] == '중립')
        
        summary = f"""📊 **'{movie_title}' 리뷰 감정 분석 결과**

{crawl_msg}

📈 **분석 결과:**
• 😊 긍정: {positive_count}개 ({positive_count/len(results)*100:.1f}%)
• 😞 부정: {negative_count}개 ({negative_count/len(results)*100:.1f}%)  
• 😐 중립: {neutral_count}개 ({neutral_count/len(results)*100:.1f}%)

💡 **종합 평가:** {'긍정적' if positive_count > negative_count else '부정적' if negative_count > positive_count else '중립적'} 반응"""
        
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
        
        **실시간 리뷰 수집 + AI 감정 분석 + 시각화**
        
        영화 제목을 입력하면 실제 사용자 리뷰를 수집하여 감정을 분석하고 결과를 시각적으로 보여드립니다.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                movie_input = gr.Textbox(
                    label="🎥 영화 제목",
                    placeholder="예: 기생충, 어벤져스, 타이타닉...",
                    lines=1
                )
                
                review_count = gr.Slider(
                    label="📊 분석할 리뷰 개수",
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1
                )
                
                analyze_btn = gr.Button("🔍 분석 시작", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 사용 팁
                - 한국어 영화 제목 권장
                - 유명한 영화일수록 더 많은 리뷰 수집
                - 분석 완료까지 10-30초 소요
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
        result_table = gr.Dataframe(
            label="📝 리뷰별 상세 분석 결과",
            headers=["순번", "리뷰 내용", "감정", "신뢰도"],
            datatype=["str", "str", "str", "str"],
            wrap=True
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
        server_name="127.0.0.1",
        server_port=7860,
        share=True,  # 외부 접근 허용 (필요 시 False로 변경)
        show_error=True
    )