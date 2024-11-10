import os
import time
import base64
import random
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    ElementNotInteractableException,
    WebDriverException
)
from datetime import datetime

# 크롬 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')
options.add_argument(
    'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36'
)
options.add_argument('ignore-certificate-errors')

# 전역 변수 초기화
crawled_count = 0

# 현재 날짜와 시간을 파일명에 사용
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def scroll_down(driver):
    print("ㅡ 스크롤 다운 시작 ㅡ")
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(2, 4)) 
        new_height = driver.execute_script("return document.body.scrollHeight")
        if last_height == new_height:
            print("ㅡ 스크롤 다운 종료 ㅡ")
            driver.execute_script("window.scrollTo(0, 0);")
            break
        last_height = new_height

def click_and_retrieve(driver, save_path, index, img, img_list_length):
    global crawled_count
    try:
        img.click()
        time.sleep(random.uniform(0.3, 0.7)) # 이미지 로드를 위해 대기
        
        src = None
        _format = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            _src = driver.find_element(By.XPATH, '//img[@class="sFlh5c FyHeAf iPVvYb"]')
            src = _src.get_attribute('src')
            print(f"Image source URL or Base64 data: {src}")
            
            if src.startswith("data:image"):
                _format = src.split(';')[0].split('/')[-1]
            else:
                _format = src.split('.')[-1].split('?')[0]
                if _format not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    _format = None
            
            if not _format:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(src, headers=headers)
                if response.status_code == 200:
                    _format = "png"
                else:
                    print(f"Unable to retrieve content type: {response.status_code}")
                    return
            
        except NoSuchElementException:
            print("이미지 요소를 찾을 수 없어 건너뜁니다.")
            return
        except WebDriverException as e:
            print(f"Error while retrieving src: {e}")
            return

        os.makedirs(f"image/{save_path}", exist_ok=True)
        
        if src.startswith("data:image"):
            base64_data = src.split(",")[1]
            with open(f"image/{save_path}/{timestamp}_{crawled_count + 1}.{_format}", "wb") as f:
                f.write(base64.b64decode(base64_data))
            print(f"{index + 1} / {img_list_length} 번째 사진 저장 (Base64 형식)")
            crawled_count += 1
        else:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(src, headers=headers)
                
                if response.status_code == 200:
                    with open(f"image/{save_path}/{timestamp}_{crawled_count + 1}.{_format}", "wb") as f:
                        f.write(response.content)
                    print(f"{index + 1} / {img_list_length} 번째 사진 저장 (URL 형식)")
                    crawled_count += 1
                else:
                    print(f"HTTPError 발생, 패스: {response.status_code}")
            
            except requests.exceptions.RequestException as e:
                print(f"요청 오류 발생, 건너뜀: {e}")
        
    except WebDriverException as e:
        print("Error during click and retrieve:", e)


def crawling(driver, query, save_path):
    
    global crawled_count
    
    print("ㅡ 크롤링 시작 ㅡ")
    crawled_count = 0 
    url = f"https://www.google.com/search?as_st=y&tbm=isch&hl=ko&as_q={query}&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=itp:photo"
    driver.get(url)

    time.sleep(10) 
    scroll_down(driver)

    div = driver.find_element(By.XPATH, """//*[@id="rso"]/div/div/div[1]/div/div""")
    img_list = div.find_elements(By.CSS_SELECTOR, ".F0uyec")
    print(len(img_list))

    os.makedirs(f'image/{save_path}', exist_ok=True)
    print(f"ㅡ {save_path} 생성 ㅡ")

    current_img = img_list[0].find_element(By.CSS_SELECTOR, "img.YQ4gaf")
    current_img.click()
    
    time.sleep(random.uniform(2, 5))  # 2초 쉬고 시작

    for index in range(len(img_list)):
        try:  
            click_and_retrieve(driver, save_path, index, current_img, len(img_list))
        except (ElementClickInterceptedException, NoSuchElementException):
            print("ㅡ ElementClickInterceptedException 또는 NoSuchElementException 발생 ㅡ")
            driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
            time.sleep(3)
            current_img.click()
            click_and_retrieve(driver, save_path, index, current_img, len(img_list))
        except ElementNotInteractableException:
            print("ㅡ ElementNotInteractableException ㅡ")
        except requests.exceptions.RequestException as e:
            print(f"네트워크 오류 발생: {e}")

        current_img = current_img.find_element(By.XPATH, """//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[1]/div/div[2]/button[2]/div""")

    try:
        print("ㅡ 크롤링 종료 (성공률: %.2f%%) ㅡ" % (crawled_count / len(img_list) * 100.0))
    except ZeroDivisionError:
        print("ㅡ img_list 가 비어있음 ㅡ")
