# my first dag :)
# cd ~/airflow/dags 에서 파일 생성 후 실행


import os
import glob
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from pendulum import datetime

# 프로젝트 경로
PROJECT_ROOT = '/Users/user/myproject/project_m_2507'


default_args = {
    'owner': 'naivblue',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

dag = DAG(
    dag_id='news_search_dag',
    description='뉴스 크롤링 및 KoBART 요약 파이프라인 (Simple)',
    schedule='0 9 * * *',  # 매일 오전 9시 실행
    start_date=datetime(2025, 7, 5),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=['news', 'summarization', 'kobart', 'simple'],
)


news_crawl_task = BashOperator(
    task_id='news_crawl',
    bash_command=f'''
        source {PROJECT_ROOT}/.venv/bin/activate && \
        cd {PROJECT_ROOT} && \
        python src/01_news_crawl.py
    ''',
    dag=dag,
)


kobart_summarize_task = BashOperator(
    task_id='kobart_summarize',
    bash_command=f'''
        source {PROJECT_ROOT}/.venv/bin/activate && \
        cd {PROJECT_ROOT} && \
        python src/02_kobart_summarize.py
    ''',
    dag=dag,
)

keyword_summarize_task = BashOperator(
    task_id='keyword_summarize',
    bash_command=f'''
        source {PROJECT_ROOT}/.venv/bin/activate && \
        cd {PROJECT_ROOT} && \
        python src/03_keyword_summarize.py
    ''',
    dag=dag,
)

def check_files():
    raw_files = glob.glob(f"{PROJECT_ROOT}/data/raw/*.json")
    processed_files = glob.glob(f"{PROJECT_ROOT}/data/processed/*.json")

    print(f"Raw 파일 수: {len(raw_files)}")
    print(f"Processed 파일 수: {len(processed_files)}")

    if raw_files:
        print(f"최신 raw 파일: {max(raw_files, key=os.path.getctime)}")
    if processed_files:
        print(f"최신 processed 파일: {max(processed_files, key=os.path.getctime)}")

check_files_task = PythonOperator(
    task_id='check_files',
    python_callable=check_files,
    dag=dag,
)

# 태스크 의존성 설정
news_crawl_task >> kobart_summarize_task >> keyword_summarize_task >> check_files_task

