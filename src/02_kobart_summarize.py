"""
KoBART 뉴스 요약 모듈
KoBART 모델을 사용하여 뉴스 텍스트를 요약하는 기능을 제공합니다.
"""

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import pandas as pd
import json
import os
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
from datetime import datetime
import glob


class KoBARTSummarizer:
    """KoBART 요약기 클래스"""
    
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
    
    def summarize_text(self, text: str, max_length: int = 128, min_length: int = 10) -> str:
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
    
    # def summarize_batch(self, texts: List[str], max_length: int = 128, min_length: int = 10) -> List[str]:
    #     """
    #     여러 텍스트를 배치로 요약합니다. (사용하지 않음 - summarize_batch_real 사용)
    #     
    #     Args:
    #         texts: 요약할 텍스트 리스트
    #         max_length: 요약 최대 길이
    #         min_length: 요약 최소 길이
    #         
    #     Returns:
    #         요약된 텍스트 리스트
    #     """
    #     summaries = []
    #     
    #     for text in tqdm(texts, desc="텍스트 요약 중"):
    #         summary = self.summarize_text(text, max_length, min_length)
    #         summaries.append(summary)
    #     
    #     return summaries
    
    def summarize_batch_real(self, texts: List[str], max_length: int = 128, min_length: int = 10, batch_size: int = 8) -> List[str]:
        """
        여러 텍스트를 진짜 배치로 요약합니다.
        
        Args:
            texts: 요약할 텍스트 리스트
            max_length: 요약 최대 길이
            min_length: 요약 최소 길이
            batch_size: 배치 크기 (GPU 메모리에 따라 조정)
            
        Returns:
            요약된 텍스트 리스트
        """
        summaries = []
        
        # 배치 단위로 처리
        for i in tqdm(range(0, len(texts), batch_size), desc="배치 요약 중"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 배치 토크나이징
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    max_length=1024, 
                    truncation=True, 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 배치 요약 생성
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
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
                
                # 배치 디코딩
                batch_summaries = [
                    self.tokenizer.decode(ids, skip_special_tokens=True).strip() 
                    for ids in summary_ids
                ]
                summaries.extend(batch_summaries)
                
            except Exception as e:
                print(f"배치 {i//batch_size + 1} 처리 중 오류: {e}")
                # 오류 발생 시 개별 처리로 폴백
                for text in batch_texts:
                    try:
                        summary = self.summarize_text(text, max_length, min_length)
                        summaries.append(summary)
                    except:
                        summaries.append("")
        
        return summaries
    
    def load_news_data(self, file_path: str) -> List[Dict]:
        """
        뉴스 데이터를 로드합니다.
        
        Args:
            file_path: 데이터 파일 경로
            
        Returns:
            뉴스 데이터 리스트
        """
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path, encoding='utf-8-sig').to_dict('records')
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. JSON 또는 CSV 파일을 사용하세요.")
        
        return data
    
    def create_summary_text(self, news_item: Dict) -> str:
        """
        뉴스 아이템에서 요약할 텍스트를 생성합니다.
        
        Args:
            news_item: 뉴스 아이템 딕셔너리
            
        Returns:
            요약할 텍스트
        """
        title = news_item.get('title', '')
        summary = news_item.get('summary', '')
        
        # 제목과 요약을 결합
        if title and summary:
            return f"{title} {summary}"
        elif title:
            return title
        elif summary:
            return summary
        else:
            return ""
    
    def process_news_data(self, data: List[Dict], max_length: int = 128, min_length: int = 10, batch_size: int = 8) -> List[Dict]:
        """
        뉴스 데이터를 처리하여 요약을 추가합니다.
        
        Args:
            data: 뉴스 데이터 리스트
            max_length: 요약 최대 길이
            min_length: 요약 최소 길이
            batch_size: 배치 크기
            
        Returns:
            요약이 추가된 뉴스 데이터 리스트
        """
        # 요약할 텍스트들 준비
        texts_to_summarize = []
        valid_indices = []
        
        for i, item in enumerate(data):
            text_to_summarize = self.create_summary_text(item)
            if text_to_summarize:
                texts_to_summarize.append(text_to_summarize)
                valid_indices.append(i)
        
        print(f"요약할 텍스트: {len(texts_to_summarize)}개")
        
        # 배치 요약 실행
        summaries = self.summarize_batch_real(texts_to_summarize, max_length, min_length, batch_size)
        
        # 결과를 원본 데이터에 추가
        processed_data = []
        summary_idx = 0
        
        for i, item in enumerate(data):
            item_copy = item.copy()
            
            if i in valid_indices:
                item_copy['kobart_summary'] = summaries[summary_idx]
                item_copy['original_text'] = texts_to_summarize[summary_idx]
                summary_idx += 1
            else:
                item_copy['kobart_summary'] = ""
                item_copy['original_text'] = ""
            
            processed_data.append(item_copy)
        
        return processed_data
    
    def save_results(self, data: List[Dict], output_path: str, format: str = 'json'):
        """
        결과를 파일로 저장합니다.
        
        Args:
            data: 저장할 데이터
            output_path: 출력 파일 경로
            format: 저장 형식 ('json' 또는 'csv')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format.lower() == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            raise ValueError("지원하지 않는 형식입니다. 'json' 또는 'csv'를 사용하세요.")
        
        print(f"결과가 {output_path}에 저장되었습니다.")


def main():
    # 설정
    max_length = 500  # 토큰 단위로 500개로 설정
    min_length = 100  # 최소 길이도 늘려서 품질 개선
    batch_size = 4  # GPU 메모리에 따라 조정 (4, 8, 16 등)
    
    # KoBART 요약기 초기화
    summarizer = KoBARTSummarizer()
    
    # 최신 뉴스 파일 찾기 (raw 폴더에서)
    news_files = glob.glob("data/raw/*.json")
    if not news_files:
        print("처리할 뉴스 파일이 없습니다.")
        return
    
    latest_file = max(news_files, key=os.path.getctime)
    print(f"처리할 파일: {latest_file}")
    
    # 뉴스 데이터 로드
    news_data = summarizer.load_news_data(latest_file)
    print(f"로드된 뉴스: {len(news_data)}개")
    
    # 뉴스 데이터 처리 (배치 처리 사용)
    print(f"배치 크기: {batch_size}")
    print(f"개별 뉴스 요약 길이: {max_length} 토큰")
    processed_data = summarizer.process_news_data(news_data, max_length, min_length, batch_size)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = f"data/processed/summarized_news_{timestamp}.json"
    
    os.makedirs("data/processed", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"요약 완료! 결과 저장: {output_file}")
    
    # 요약 통계
    summaries = [item['kobart_summary'] for item in processed_data if item['kobart_summary']]
    print(f"성공적으로 요약된 뉴스: {len(summaries)}개")
    
    print(f"\n=== 개별 뉴스 요약 완료 ===")
    print(f"다음 단계: python src/03_keyword_summarize.py 를 실행하여 키워드별 요약을 생성하세요.")


if __name__ == "__main__":
    main() 