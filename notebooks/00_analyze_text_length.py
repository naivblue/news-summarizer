#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
뉴스 텍스트 길이 분석 스크립트
"""

import json
import numpy as np
from collections import defaultdict

def analyze_text_length():
    """뉴스 텍스트 길이를 분석합니다."""
    
    print("=== 뉴스 텍스트 길이 분석 ===\n")
    
    # 데이터 로드
    with open('../data/raw/naver_news_20250628.json', 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    print(f"총 뉴스 개수: {len(news_data)}개\n")
    
    # 텍스트 길이 계산
    title_lengths = []
    summary_lengths = []
    combined_lengths = []
    
    for item in news_data:
        title = item.get('title', '')
        summary = item.get('summary', '')
        
        title_lengths.append(len(title))
        summary_lengths.append(len(summary))
        
        # 결합된 텍스트 (preprocess_text와 동일한 방식)
        combined_text = f"{title} {summary}".strip()
        combined_lengths.append(len(combined_text))
    
    # 기본 통계 출력
    print("=== 텍스트 길이 통계 ===")
    print(f"제목 평균 길이: {np.mean(title_lengths):.1f}자")
    print(f"제목 최대 길이: {np.max(title_lengths)}자")
    print(f"제목 최소 길이: {np.min(title_lengths)}자")
    print(f"제목 중간값: {np.median(title_lengths):.1f}자")
    print()
    
    print(f"요약 평균 길이: {np.mean(summary_lengths):.1f}자")
    print(f"요약 최대 길이: {np.max(summary_lengths)}자")
    print(f"요약 최소 길이: {np.min(summary_lengths)}자")
    print(f"요약 중간값: {np.median(summary_lengths):.1f}자")
    print()
    
    print(f"결합 텍스트 평균 길이: {np.mean(combined_lengths):.1f}자")
    print(f"결합 텍스트 최대 길이: {np.max(combined_lengths)}자")
    print(f"결합 텍스트 최소 길이: {np.min(combined_lengths)}자")
    print(f"결합 텍스트 중간값: {np.median(combined_lengths):.1f}자")
    print()
    
    # 1024자 제한 관련 분석
    over_1024 = [l for l in combined_lengths if l > 1024]
    under_1024 = [l for l in combined_lengths if l <= 1024]
    
    print("=== 1024자 제한 분석 ===")
    print(f"1024자 이하 텍스트: {len(under_1024)}개 ({len(under_1024)/len(combined_lengths)*100:.1f}%)")
    print(f"1024자 초과 텍스트: {len(over_1024)}개 ({len(over_1024)/len(combined_lengths)*100:.1f}%)")
    
    if over_1024:
        print(f"1024자 초과 텍스트 평균 길이: {np.mean(over_1024):.1f}자")
        print(f"1024자 초과 텍스트 최대 길이: {np.max(over_1024)}자")
        print(f"1024자 초과 텍스트 최소 길이: {np.min(over_1024)}자")
    print()
    
    # 키워드별 평균 텍스트 길이
    keyword_stats = defaultdict(list)
    for item in news_data:
        keyword = item.get('keyword', '')
        title = item.get('title', '')
        summary = item.get('summary', '')
        combined_text = f"{title} {summary}".strip()
        keyword_stats[keyword].append(len(combined_text))
    
    print("=== 키워드별 평균 텍스트 길이 ===")
    for keyword, lengths in sorted(keyword_stats.items()):
        avg_length = np.mean(lengths)
        print(f"{keyword}: {avg_length:.1f}자 ({len(lengths)}개)")
    print()
    
    # 길이 분포 요약
    print("=== 길이 분포 요약 ===")
    print("결합 텍스트 길이 구간별 분포:")
    ranges = [(0, 200), (201, 400), (401, 600), (601, 800), (801, 1024), (1025, 1500)]
    for start, end in ranges:
        count = len([l for l in combined_lengths if start <= l <= end])
        print(f"  {start}-{end}자: {count}개 ({count/len(combined_lengths)*100:.1f}%)")

if __name__ == "__main__":
    analyze_text_length() 