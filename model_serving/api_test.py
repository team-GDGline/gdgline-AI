import base64
import requests
import json
import time

# 1. 이미지 파일을 Base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

# 2. TorchServe 예측 요청 함수 (응답 시간 측정 추가)
def send_prediction_request(image_path, server_url):
    # Base64로 인코딩된 이미지 생성
    base64_image = encode_image_to_base64(image_path)
    
    # 요청 데이터 생성
    data = {"image": base64_image}
    
    # 서버로 POST 요청 보내기
    headers = {"Content-Type": "application/json"}
    
    # 요청 시간 측정 시작
    start_time = time.time()
    
    # 서버에 POST 요청
    response = requests.post(server_url, headers=headers, json=data)
    
    # 요청 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time  # 경과 시간 계산
    
    # 응답 결과 반환
    if response.status_code == 200:
        return {
            "response": response.json(),
            "time_taken": elapsed_time
        }
    else:
        return {
            "error": response.status_code,
            "message": response.text,
            "time_taken": elapsed_time
        }

# 3. 실행
if __name__ == "__main__":
    # 이미지 파일 경로
    image_path = "image4-Photoroom.png"  # 업로드할 이미지 파일 경로
    
    # TorchServe 서버 URL
    server_url = "http://127.0.0.1:8080/predictions/yolo_model"
    
    # 예측 요청 보내기
    result = send_prediction_request(image_path, server_url)
    
    # 결과 출력
    print("Prediction Result:", result["response"] if "response" in result else result)
    print(f"Time Taken: {result['time_taken']:.4f} seconds")
