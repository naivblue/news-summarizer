import json
from collections import defaultdict

def analyze_text_length():
    """뉴스 텍스트 길이를 분석합니다."""
    
    # 뉴스 데이터 로드
    with open('data/raw/naver_news_20250628.json', 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    print(f"총 뉴스 개수: {len(news_data)}개")
    
    # 텍스트 길이 분석
    title_lengths = []
    summary_lengths = []
    combined_lengths = []
    
    for item in news_data:
        title = item.get('title', '')
        summary = item.get('summary', '')
        
        # 제목 길이
        title_lengths.append(len(title))
        
        # 요약 길이
        summary_lengths.append(len(summary))
        
        # 결합된 텍스트 길이 (preprocess_text와 동일한 방식)
        combined_text = f"{title} {summary}".strip()
        combined_lengths.append(len(combined_text))
    
    # 통계 계산
    print("\n=== 텍스트 길이 분석 ===")
    print(f"제목 평균 길이: {sum(title_lengths)/len(title_lengths):.1f}자")
    print(f"제목 최대 길이: {max(title_lengths)}자")
    print(f"제목 최소 길이: {min(title_lengths)}자")
    
    print(f"\n요약 평균 길이: {sum(summary_lengths)/len(summary_lengths):.1f}자")
    print(f"요약 최대 길이: {max(summary_lengths)}자")
    print(f"요약 최소 길이: {min(summary_lengths)}자")
    
    print(f"\n결합 텍스트 평균 길이: {sum(combined_lengths)/len(combined_lengths):.1f}자")
    print(f"결합 텍스트 최대 길이: {max(combined_lengths)}자")
    print(f"결합 텍스트 최소 길이: {min(combined_lengths)}자")
    
    # 1024자 제한 관련 분석
    over_1024 = [l for l in combined_lengths if l > 1024]
    print(f"\n1024자 초과 텍스트: {len(over_1024)}개 ({len(over_1024)/len(combined_lengths)*100:.1f}%)")
    
    if over_1024:
        print(f"1024자 초과 텍스트 평균 길이: {sum(over_1024)/len(over_1024):.1f}자")
        print(f"1024자 초과 텍스트 최대 길이: {max(over_1024)}자")
    
    # 키워드별 분석
    keyword_stats = defaultdict(list)
    for item in news_data:
        keyword = item.get('keyword', '')
        title = item.get('title', '')
        summary = item.get('summary', '')
        combined_text = f"{title} {summary}".strip()
        keyword_stats[keyword].append(len(combined_text))
    
    print(f"\n=== 키워드별 평균 텍스트 길이 ===")
    for keyword, lengths in keyword_stats.items():
        avg_length = sum(lengths) / len(lengths)
        print(f"{keyword}: {avg_length:.1f}자 ({len(lengths)}개)")

if __name__ == "__main__":
    analyze_text_length() 