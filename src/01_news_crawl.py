"""
뉴스 크롤링 모듈
네이버 뉴스에서 뉴스 데이터를 수집하는 기능을 제공합니다.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime
import json
import os
from typing import List, Dict, Optional


class NewsCrawler:
    """네이버 뉴스 크롤러 클래스"""
    
    def __init__(self, headers: Optional[Dict] = None):
        """
        Args:
            headers: HTTP 요청 헤더 (기본값: None)
        """
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def crawl_naver_news(self, keyword: str, max_pages: int = 10) -> List[Dict]:
        """
        네이버 뉴스에서 키워드로 검색하여 뉴스 데이터를 수집합니다.
        
        Args:
            keyword: 검색할 키워드
            max_pages: 최대 수집할 페이지 수
            
        Returns:
            수집된 뉴스 데이터 리스트
        """
        news_data = []
        
        for page in range(1, max_pages + 1):
            try:
                # 네이버 뉴스 검색 URL
                url = f"https://search.naver.com/search.naver?where=news&query={keyword}&start={(page-1)*10+1}"
                
                response = self.session.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # group_news 클래스 안의 뉴스 아이템들 찾기
                news_container = soup.find('div', class_='group_news')
                if not news_container:
                    print(f"페이지 {page}에서 뉴스 컨테이너를 찾을 수 없습니다.")
                    break
                
                # 네이버 뉴스 링크들 찾기
                all_links = news_container.find_all('a', href=True)
                naver_news_links = []
                
                for link in all_links:
                    href = link.get('href')
                    if href and 'news.naver.com' in href:
                        naver_news_links.append(href)
                
                # 중복 제거
                naver_news_links = list(set(naver_news_links))
                
                if not naver_news_links:
                    print(f"페이지 {page}에서 뉴스 링크를 찾을 수 없습니다.")
                    break
                
                print(f"페이지 {page}: {len(naver_news_links)}개 뉴스 링크 발견")
                
                # 각 뉴스 링크에서 상세 정보 추출
                for i, news_link in enumerate(naver_news_links):
                    try:
                        # 개별 뉴스 페이지 접근
                        news_response = self.session.get(news_link)
                        news_response.raise_for_status()
                        
                        news_soup = BeautifulSoup(news_response.text, 'html.parser')
                        
                        # 제목 추출
                        title = None
                        title_selectors = [
                            'h2#title_area span',
                            'h3#articleTitle',
                            '.media_end_head_headline',
                            'h2.media_end_head_headline',
                            '.article_title'
                        ]
                        
                        for selector in title_selectors:
                            try:
                                elem = news_soup.select_one(selector)
                                if elem:
                                    title = elem.get_text(strip=True)
                                    break
                            except:
                                continue
                        
                        # 다른 방법으로 제목 찾기
                        if not title:
                            h_tags = news_soup.find_all(['h1', 'h2', 'h3'])
                            for h in h_tags:
                                text = h.get_text(strip=True)
                                if text and len(text) > 10 and '뉴스' not in text:
                                    title = text
                                    break
                        
                        # 내용 추출
                        content = None
                        content_selectors = [
                            '#newsct_article',
                            '.news_end_body_container',
                            '#articleBodyContents',
                            '.article_body'
                        ]
                        
                        for selector in content_selectors:
                            try:
                                elem = news_soup.select_one(selector)
                                if elem:
                                    content = elem.get_text(strip=True)
                                    break
                            except:
                                continue
                        
                        # 언론사 추출
                        press = None
                        press_selectors = [
                            '.media_end_head_top_logo img',
                            '.press_logo img',
                            '.media_end_head_top_logo_text',
                            '.press_name'
                        ]
                        
                        for selector in press_selectors:
                            try:
                                elem = news_soup.select_one(selector)
                                if elem:
                                    if elem.name == 'img':
                                        press = elem.get('alt', '')
                                    else:
                                        press = elem.get_text(strip=True)
                                    if press:
                                        break
                            except:
                                continue
                        
                        # 날짜 추출
                        date = None
                        date_selectors = [
                            '.media_end_head_info_datestamp_time',
                            '.article_info span',
                            '.date'
                        ]
                        
                        for selector in date_selectors:
                            try:
                                elem = news_soup.select_one(selector)
                                if elem:
                                    date = elem.get_text(strip=True)
                                    break
                            except:
                                continue
                        
                        if title and content:
                            news_data.append({
                                'title': title,
                                'link': news_link,
                                'press': press or "",
                                'content': content,
                                'date': date or "",
                                'keyword': keyword,
                                'crawled_at': datetime.now().isoformat()
                            })
                            print(f"  뉴스 {i+1}/{len(naver_news_links)} 수집 완료: {title[:30]}...")
                        else:
                            print(f"  뉴스 {i+1}/{len(naver_news_links)} 스킵: 제목 또는 내용 없음")
                    
                    except Exception as e:
                        print(f"  뉴스 {i+1}/{len(naver_news_links)} 파싱 중 오류: {e}")
                        continue
                    
                    # 서버 부하 방지를 위한 딜레이
                    time.sleep(random.uniform(0.5, 1.5))
                
                print(f"페이지 {page} 완료: {len([n for n in news_data if n['keyword'] == keyword])}개 뉴스 수집")
                
                # 페이지 간 딜레이
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"페이지 {page} 크롤링 중 오류: {e}")
                continue
        
        return news_data
    
    def save_to_file(self, data: List[Dict], filename: str, format: str = 'json'):
        """
        수집된 데이터를 파일로 저장합니다.
        
        Args:
            data: 저장할 데이터
            filename: 파일명
            format: 저장 형식 ('json' 또는 'csv')
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if format.lower() == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format.lower() == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
        else:
            raise ValueError("지원하지 않는 형식입니다. 'json' 또는 'csv'를 사용하세요.")
        
        print(f"데이터가 {filename}에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    # 크롤러 초기화
    crawler = NewsCrawler()
    
    # 검색할 키워드들
    keywords = ['ETF','미국 주식']
    
    all_news_data = []
    
    for keyword in keywords:
        print(f"\n키워드 '{keyword}'로 뉴스 수집 시작...")
        news_data = crawler.crawl_naver_news(keyword, max_pages=1)
        all_news_data.extend(news_data)
        print(f"키워드 '{keyword}' 완료: {len(news_data)}개 뉴스 수집")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # JSON 형식으로 저장
    json_filename = f"data/raw/naver_news_{timestamp}.json"
    crawler.save_to_file(all_news_data, json_filename, 'json')
    
    # CSV 형식으로 저장 (주석처리)
    # csv_filename = f"data/raw/naver_news_{timestamp}.csv"
    # crawler.save_to_file(all_news_data, csv_filename, 'csv')
    
    print(f"\n총 {len(all_news_data)}개의 뉴스 데이터를 수집했습니다.")


if __name__ == "__main__":
    main() 