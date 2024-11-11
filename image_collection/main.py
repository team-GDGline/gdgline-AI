import argparse
from selenium import webdriver
from image_collection.aquarium_crawling import crawling
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from image_collection.remove_duplicate_image import delete_image_from_path
from image_preprocessing import filtering
import json
import time


def retrieve_single_query(driver, query):
    crawling(driver, query, query)


def retrieve_multiple_queries(driver, json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        queries = json.load(file)

    for save_path, query_list in queries.items():
        start_time = time.time()
        
        for query in query_list:
            crawling(driver, query, save_path)

        folder_path = f'images/{save_path}'
        
        # 이미지 크기가 작은 것 삭제
        filtering(folder_path)
        
        # 중복 이미지 삭제
        delete_image_from_path(folder_path)
        
        print(f"Queries for '{save_path}' 처리 완료. 소요 시간: {time.time() - start_time:.2f}초")


if __name__ == '__main__':
    # argparse를 사용하여 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="Aquarium Crawling Script")
    parser.add_argument(
        '--mode', 
        choices=['single', 'multiple'], 
        required=True, 
        help="single: 단일 쿼리 검색, multiple: 여러 쿼리 검색"
    )
    parser.add_argument(
        '--query', 
        type=str, 
        help="단일 쿼리 검색 시 사용할 검색어"
    )
    parser.add_argument(
        '--json_file_path', 
        type=str, 
        help="여러 쿼리 검색 시 사용할 JSON 파일 경로"
    )
    parser.add_argument(
        '--no_quit', 
        action='store_true', 
        help="작업 완료 후 브라우저를 종료하지 않음"
    )

    args = parser.parse_args()

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    if args.mode == 'single':
        if not args.query:
            print("단일 쿼리 모드를 선택했지만 쿼리를 제공하지 않았습니다. --query 인수를 입력하세요.")
        else:
            retrieve_single_query(driver, args.query)
    
    elif args.mode == 'multiple':
        if not args.json_file_path:
            print("여러 쿼리 모드를 선택했지만 JSON 파일 경로를 제공하지 않았습니다. --json_file_path 인수를 입력하세요.")
        else:
            retrieve_multiple_queries(driver, args.json_file_path)
    
    if not args.no_quit:
        driver.quit()
