import asyncio
from playwright.async_api import async_playwright

async def crawl_naver_movie_reviews(movie_title, max_reviews=10):
    reviews = []

    async with async_playwright() as p:
        # Chromium 브라우저 실행 (headless = True 기본값)
        # browser = await p.chromium.launch()
        browser = await p.chromium.launch(
            headless=True,        # GUI 없이 실행
            args=["--no-sandbox"] # Spaces에서 필수 옵션
        )
        page = await browser.new_page()

        # 영화 리뷰 검색 URL
        search_url = f"https://search.naver.com/search.naver?query=영화 {movie_title} 평점"
        await page.goto(search_url)

        # 페이지 로딩 대기
        await page.wait_for_timeout(2000)

        # 리뷰 요소 찾기 (최대 max_reviews 개)
        review_elements = await page.query_selector_all(".area_review_content .desc._text")

        for elem in review_elements[:max_reviews]:
            text = await elem.inner_text()
            reviews.append(text.strip())

        await browser.close()

    return reviews, f"✅ {len(reviews)}개의 리뷰를 수집했습니다."


# 실행 예시
if __name__ == "__main__":
    movie = "웡카"
    reviews, crawl_msg = asyncio.run(crawl_naver_movie_reviews(movie, max_reviews=10))
    print(crawl_msg)
    for i, r in enumerate(reviews, 1):
        print(f"{i}: {r}")
