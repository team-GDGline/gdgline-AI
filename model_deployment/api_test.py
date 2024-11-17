import requests
import base64
import json

# 이미지 파일을 읽어 Base64로 인코딩
with open('input_image.png', 'rb') as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

# 요청 데이터 생성
payload = {
    "image": image_base64
}

# 헤더 설정
headers = {
    'Content-Type': 'application/json'
}

# 추론 요청 보내기
response = requests.post('http://127.0.0.1:8080/predictions/yolo', headers=headers, json=payload)

# 응답 상태 코드 확인 및 처리
if response.status_code == 200:
    try:
        # JSON 응답 파싱
        result = response.json()
        print("추론 결과:", json.dumps(result, indent=4, ensure_ascii=False))
    except json.JSONDecodeError:
        print("응답이 JSON 형식이 아닙니다.")
        print("응답 내용:", response.text)
else:
    print(f"요청 실패: 상태 코드 {response.status_code}")
    print("응답 내용:", response.text)
