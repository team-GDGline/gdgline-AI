from selenium import webdriver
from aquarium_crwaling import crawling
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from remove_duplicate_image import find_duplicate_images, delete_duplicates
import json
import time

def retrieve_single_query(driver):
    query = input("입력 : ")
    
    crawling(driver, query, query)

def retrieve_multiple_queries(driver, json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        queries = json.load(file)

    for save_path, query_list in queries.items():
        query_list = query_list 
        for query in query_list:
            crawling(driver, query, save_path)

        # 중복 이미지 삭제
        folder_path = f'image/{save_path}'
        duplicate_files, total_files = find_duplicate_images(folder_path)
        if duplicate_files:
            print(f"{save_path} 폴더에서 중복 이미지 파일이 발견되었습니다. 삭제를 시작합니다.")
            deleted_files = delete_duplicates(duplicate_files)
            print(f"{save_path} 폴더의 중복 이미지 삭제 완료 (삭제된 파일 수: {deleted_files})")
        else:
            print(f"{save_path} 폴더에 중복 이미지 파일이 없습니다.")
        
        print(f"전체 파일 수: {total_files}, 중복 제거 후 남은 파일 수: {total_files - deleted_files}")


if __name__ == '__main__':
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    # # 1개 query 검색
    # retrieve_single_query(driver)
    
    # 여러 개 query 검색
    json_file_path = "queries.json"
    retrieve_multiple_queries(driver, json_file_path)
    
    driver.quit()
    
