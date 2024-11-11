from selenium import webdriver
from aquarium_crwaling import crawling
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from image_collection.remove_duplicate_image import delete_image_from_path
from image_preprocessing import filtering
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
        
        start_time = time.time()
        
        for query in query_list:
            
            crawling(driver, query, save_path)

        folder_path = f'image/{save_path}'
        
        # 이미지 크기가 작은 것 삭제
        filtering(folder_path)
        
        # 중복 이미지 삭제
        delete_image_from_path(folder_path)
        
        print(f"Queries for '{save_path}' 처리 완료. 소요 시간: {time.time() - start_time:.2f}초")


if __name__ == '__main__':
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    # # 1개 query 검색
    # retrieve_single_query(driver)
    
    # 여러 개 query 검색
    json_file_path = "queries.json"
    retrieve_multiple_queries(driver, json_file_path)
    
    driver.quit()
    
