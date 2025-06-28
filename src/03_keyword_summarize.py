"""
키워드별 뉴스 요약 모듈
개별 뉴스 요약 결과를 키워드별로 그룹화하여 추가 요약을 생성합니다.
"""

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import json
import os
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import glob
from collections import defaultdict


class KeywordSummarizer:
    """키워드별 요약기 클래스"""
    
    def __init__(self, model_name: str = "gogamza/kobart-base-v2"):
        """
        Args:
            model_name: 사용할 KoBART 모델명
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        # 토크나이저와 모델 로드
        print("토크나이저 로딩 중...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        
        print("KoBART 모델 로딩 중...")
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("모델 로딩 완료!")
    
    def preprocess_text(self, text: str, max_length: int = 1024) -> str:
        """
        텍스트를 전처리합니다.
        
        Args:
            text: 전처리할 텍스트
            max_length: 최대 길이
            
        Returns:
            전처리된 텍스트
        """
        # 특수문자 제거 및 정리
        text = text.strip()
        
        # 너무 긴 텍스트는 자르기
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def summarize_text(self, text: str, max_length: int = 500, min_length: int = 100) -> str:
        """
        텍스트를 요약합니다.
        
        Args:
            text: 요약할 텍스트
            max_length: 요약 최대 길이
            min_length: 요약 최소 길이
            
        Returns:
            요약된 텍스트
        """
        try:
            # 텍스트 전처리
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return ""
            
            # 토크나이징
            inputs = self.tokenizer(processed_text, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 요약 생성
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length,  # 새로 생성할 토큰 수 제한
                    min_new_tokens=min_length,  # 최소 새로 생성할 토큰 수
                    num_beams=5,  # 품질 개선을 위해 5로 증가
                    length_penalty=0.8,  # 짧은 요약을 위해 감소
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # 반복 방지
                    do_sample=True,  # 다양성 증가
                    temperature=0.8,  # 창의성 조절
                    top_k=50,  # 상위 토큰만 고려
                    top_p=0.9,  # 누적 확률 제한
                    repetition_penalty=1.2  # 반복 방지 강화
                )
            
            # 디코딩
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"요약 중 오류 발생: {e}")
            return ""
    
    def load_summarized_data(self, file_path: str) -> List[Dict]:
        """
        요약된 뉴스 데이터를 로드합니다.
        
        Args:
            file_path: 데이터 파일 경로
            
        Returns:
            요약된 뉴스 데이터 리스트
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def create_keyword_summary(self, processed_data: List[Dict], max_length: int = 500) -> Dict:
        """
        키워드별로 뉴스를 그룹화하고 추가 요약을 생성합니다.
        
        Args:
            processed_data: 처리된 뉴스 데이터
            max_length: 키워드별 요약 최대 길이
            
        Returns:
            키워드별 요약 딕셔너리
        """
        # 키워드별로 뉴스 그룹화
        keyword_groups = defaultdict(list)
        for item in processed_data:
            keyword = item.get('keyword', '')
            if keyword and item.get('kobart_summary'):
                keyword_groups[keyword].append(item)
        
        print(f"\n키워드별 뉴스 개수:")
        for keyword, items in keyword_groups.items():
            print(f"  {keyword}: {len(items)}개")
        
        # 키워드별 추가 요약 생성
        keyword_summaries = {}
        
        for keyword, items in keyword_groups.items():
            print(f"\n=== {keyword} 키워드 추가 요약 생성 중 ===")
            
            # 해당 키워드의 모든 요약 텍스트를 결합
            combined_summaries = []
            for item in items:
                summary = item.get('kobart_summary', '')
                title = item.get('title', '')
                if summary:
                    # 제목과 요약을 결합
                    combined_text = f"{title}: {summary}"
                    combined_summaries.append(combined_text)
            
            if combined_summaries:
                # 모든 요약을 하나의 긴 텍스트로 결합
                full_text = " ".join(combined_summaries)
                
                # 500자 정도의 키워드별 요약 생성
                keyword_summary = self.summarize_text(full_text, max_length, min_length=100)
                
                keyword_summaries[keyword] = {
                    'summary': keyword_summary,
                    'news_count': len(items),
                    'titles': [item.get('title', '') for item in items[:5]]  # 상위 5개 제목
                }
                
                print(f"생성된 요약 ({len(keyword_summary)}자):")
                print(f"{keyword_summary[:200]}...")
            else:
                print(f"{keyword}: 요약할 내용이 없습니다.")
        
        return keyword_summaries


def save_keyword_summary_md(keyword_summaries: Dict, output_file: str):
    """키워드별 요약을 마크다운 형식으로 저장합니다."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 키워드별 뉴스 요약\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y년 %m월 %d일')}\n\n")
        
        for keyword, summary_data in keyword_summaries.items():
            f.write(f"## 🔍 {keyword} ({summary_data['news_count']}개 뉴스)\n\n")
            f.write(f"### 📝 요약\n")
            f.write(f"{summary_data['summary']}\n\n")
            
            f.write(f"### 📰 주요 뉴스 제목\n")
            for i, title in enumerate(summary_data['titles'], 1):
                f.write(f"{i}. {title}\n")
            f.write("\n---\n\n")
    
    print(f"마크다운 요약이 {output_file}에 저장되었습니다.")


def main():
    # 설정
    max_length = 500  # 토큰 단위로 500개로 설정
    min_length = 100  # 최소 길이
    
    # 키워드 요약기 초기화
    summarizer = KeywordSummarizer()
    
    # 최신 요약된 뉴스 파일 찾기 (processed 폴더에서)
    news_files = glob.glob("data/processed/summarized_news_*.json")
    if not news_files:
        print("처리할 요약된 뉴스 파일이 없습니다.")
        print("먼저 python src/02_kobart_summarize.py 를 실행하여 개별 뉴스 요약을 생성하세요.")
        return
    
    latest_file = max(news_files, key=os.path.getctime)
    print(f"처리할 파일: {latest_file}")
    
    # 요약된 뉴스 데이터 로드
    news_data = summarizer.load_summarized_data(latest_file)
    print(f"로드된 뉴스: {len(news_data)}개")
    
    # 키워드별 요약 생성
    print(f"\n=== 키워드별 요약 생성 시작 ===")
    print(f"키워드별 요약 길이: {max_length} 토큰")
    keyword_summaries = summarizer.create_keyword_summary(news_data, max_length)
    
    # 키워드별 요약 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d")
    keyword_output_file = f"data/processed/keyword_summaries_{timestamp}.json"
    with open(keyword_output_file, 'w', encoding='utf-8') as f:
        json.dump(keyword_summaries, f, ensure_ascii=False, indent=2)
    
    print(f"키워드별 요약 저장: {keyword_output_file}")
    
    # 마크다운 형식으로도 저장
    md_output_file = f"data/processed/keyword_summaries_{timestamp}.md"
    save_keyword_summary_md(keyword_summaries, md_output_file)
    
    print(f"마크다운 요약 저장: {md_output_file}")
    
    print(f"\n=== 키워드별 요약 완료 ===")
    print(f"키워드별 요약: {len(keyword_summaries)}개")


if __name__ == "__main__":
    main() 