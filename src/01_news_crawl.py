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
                
                # 새로운 네이버 뉴스 구조에 맞게 수정
                # group_news 클래스 안의 뉴스 아이템들 찾기
                news_container = soup.find('div', class_='group_news')
                if not news_container:
                    print(f"페이지 {page}에서 뉴스 컨테이너를 찾을 수 없습니다.")
                    break
                
                # 뉴스 아이템들 찾기 (I6obO60yNcW8I32mDzvQ 클래스)
                news_items = news_container.find_all('div', class_='I6obO60yNcW8I32mDzvQ')
                
                if not news_items:
                    print(f"페이지 {page}에서 뉴스를 찾을 수 없습니다.")
                    break
                
                for item in news_items:
                    try:
                        # 제목 추출 (W035WwZVZIWyuG66e5iI 클래스의 a 태그)
                        title_element = item.find('a', class_='W035WwZVZIWyuG66e5iI')
                        title = ""
                        link = ""
                        if title_element:
                            title = title_element.get_text(strip=True)
                            link = title_element.get('href', "")
                        
                        # 요약 추출 (ti6bfMWvbomDA5J1fNOX 클래스의 a 태그)
                        summary_element = item.find('a', class_='ti6bfMWvbomDA5J1fNOX')
                        summary = summary_element.get_text(strip=True) if summary_element else ""
                        
                        # 언론사 추출 (sds-comps-profile-info-title-text 클래스)
                        press_element = item.find('span', class_='sds-comps-profile-info-title-text')
                        press = press_element.get_text(strip=True) if press_element else ""
                        
                        # 날짜 추출 (vVkv3w6EdDH42m2lfgVt 클래스의 span)
                        date_element = item.find('span', class_='vVkv3w6EdDH42m2lfgVt')
                        date = date_element.get_text(strip=True) if date_element else ""
                        
                        if title and link:
                            news_data.append({
                                'title': title,
                                'link': link,
                                'press': press,
                                'summary': summary,
                                'date': date,
                                'keyword': keyword,
                                'crawled_at': datetime.now().isoformat()
                            })
                    
                    except Exception as e:
                        print(f"뉴스 아이템 파싱 중 오류: {e}")
                        continue
                
                print(f"페이지 {page} 완료: {len(news_items)}개 뉴스 수집")
                
                # 서버 부하 방지를 위한 딜레이
                time.sleep(random.uniform(1, 3))
                
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
    keywords = ['ETF', '미국주식', '비트코인', '테슬라', '스테이블코인']
    
    all_news_data = []
    
    for keyword in keywords:
        print(f"\n키워드 '{keyword}'로 뉴스 수집 시작...")
        news_data = crawler.crawl_naver_news(keyword, max_pages=2)
        all_news_data.extend(news_data)
        print(f"키워드 '{keyword}' 완료: {len(news_data)}개 뉴스 수집")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # JSON 형식으로 저장
    json_filename = f"../data/raw/naver_news_{timestamp}.json"
    crawler.save_to_file(all_news_data, json_filename, 'json')
    
    # CSV 형식으로 저장 (주석처리)
    # csv_filename = f"../data/raw/naver_news_{timestamp}.csv"
    # crawler.save_to_file(all_news_data, csv_filename, 'csv')
    
    print(f"\n총 {len(all_news_data)}개의 뉴스 데이터를 수집했습니다.")


if __name__ == "__main__":
    main() 