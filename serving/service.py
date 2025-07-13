"""
BentoML 뉴스 요약 서비스 (간단한 함수 기반)
"""

# 프로젝트 루트 경로 추가
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

import bentoml
from typing import Dict, List, Any
import json
from datetime import datetime
import importlib

# 정적 모듈 import (숫자로 시작하는 파일명 처리)
news_crawl = importlib.import_module("01_news_crawl")
kobart_summarize = importlib.import_module("02_kobart_summarize")
keyword_summarize = importlib.import_module("03_keyword_summarize")

# 클래스 추출
NewsCrawler = news_crawl.NewsCrawler
KoBARTSummarizer = kobart_summarize.KoBARTSummarizer
KeywordSummarizer = keyword_summarize.KeywordSummarizer

# 글로벌 인스턴스 생성
print("🚀 뉴스 요약 서비스 시작 중...")

# 크롤러 초기화
crawler = NewsCrawler()
print("✅ 뉴스 크롤러 초기화 완료")

# KoBART 요약기 초기화
summarizer = KoBARTSummarizer()
print("✅ KoBART 요약기 초기화 완료")

# 키워드 요약기 초기화
keyword_summarizer = KeywordSummarizer()
print("✅ 키워드 요약기 초기화 완료")

print("🎉 모든 서비스 준비 완료!")


# BentoML 서비스 정의
@bentoml.service
class SimpleNewsService:
    
    @bentoml.api
    def health_check(self) -> Dict[str, Any]:
        """헬스 체크 API"""
        return {
            "status": "healthy",
            "service": "news_summarizer",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "crawler": "ready",
                "summarizer": "ready",
                "keyword_summarizer": "ready"
            }
        }
    

    @bentoml.api
    def crawl_news(self, keywords: List[str] = ["ETF"], pages: int = 3) -> Dict[str, Any]:
        """뉴스 크롤링 API"""
        try:
            print("🔍 뉴스 크롤링 시작...")
            all_news = []
            for keyword in keywords:
                news_data = crawler.crawl_naver_news(keyword, max_pages=pages)
                all_news.extend(news_data)
            
            return {
                "status": "success",
                "data": {
                    "news_count": len(all_news),
                    "news_data": all_news[:10],  # 처음 10개만 반환
                    "keywords": keywords
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @bentoml.api
    def summarize_news(self, news_data: List[Dict]) -> Dict[str, Any]:
        """뉴스 요약 API"""
        try:
            print("📝 뉴스 요약 시작...")
            summarized_data = summarizer.process_news_data(
                news_data,
                max_length=128,
                min_length=10,
                batch_size=4
            )
            
            return {
                "status": "success",
                "data": {
                    "original_count": len(news_data),
                    "summarized_count": len(summarized_data),
                    "summarized_data": summarized_data
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @bentoml.api
    def full_pipeline(self, keywords: List[str] = ["ETF"], pages: int = 3) -> Dict[str, Any]:
        """전체 파이프라인 API"""
        try:
            print("🔍 뉴스 크롤링 시작...")
            all_news = []
            for keyword in keywords:
                news_data = crawler.crawl_naver_news(keyword, max_pages=pages)
                all_news.extend(news_data)
            
            if not all_news:
                return {
                    "status": "error",
                    "message": "크롤링된 뉴스가 없습니다.",
                    "timestamp": datetime.now().isoformat()
                }
            
            print("📝 뉴스 요약 시작...")
            summarized_data = summarizer.process_news_data(
                all_news[:10],  # 처음 10개만 요약
                max_length=128,
                min_length=10,
                batch_size=4
            )
            
            print("🔑 키워드별 요약 시작...")
            keyword_summaries = keyword_summarizer.create_keyword_summary(
                summarized_data,
                max_length=300
            )
            
            return {
                "status": "success",
                "data": {
                    "crawl_count": len(all_news),
                    "summarized_count": len(summarized_data),
                    "keyword_summaries": keyword_summaries,
                    "stats": {
                        "total_news": len(all_news),
                        "summarized_news": len(summarized_data),
                        "keywords": list(keyword_summaries.keys()) if keyword_summaries else []
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            } 