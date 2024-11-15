import base64
import requests
import json

# 1. 이미지 파일을 Base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

# 2. TorchServe 예측 요청 함수
def send_prediction_request(image_path, server_url):
    # Base64로 인코딩된 이미지 생성
    base64_image = encode_image_to_base64(image_path)
    
    # 요청 데이터 생성 (body 없이)
    data = {"image": base64_image}
    
    # 서버로 POST 요청 보내기
    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, headers=headers, json=data)  # 여기서 `json` 매개변수 사용
    
    # 응답 결과 반환
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}

# 3. 실행
if __name__ == "__main__":
    # 이미지 파일 경로
    image_path = "image4-Photoroom.png"  # 업로드할 이미지 파일 경로
    
    # TorchServe 서버 URL
    server_url = "http://127.0.0.1:8080/predictions/yolo_model"
    
    # 예측 요청 보내기
    result = send_prediction_request(image_path, server_url)
    print("Prediction Result:", result)
