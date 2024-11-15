import base64
import json
import cv2

# YOLOHandler 클래스 임포트
from yolo_handler import YOLOHandler

# 테스트 이미지 로드 및 Base64로 변환
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Base64로 인코딩된 이미지를 생성
image_path = "image4-Photoroom.png"  # 테스트할 이미지 파일 경로
base64_image = encode_image_to_base64(image_path)

# YOLOHandler 인스턴스 생성
handler = YOLOHandler()

# 모델 초기화
context = {}  # Context는 빈 dict로 설정
handler.initialize(context)

# 테스트 데이터 생성
data = [{"image": f"data:image/jpeg;base64,{base64_image}"}]

# 모델 처리
result = handler.handle(data, context)

# 결과 출력
print(json.dumps(result, indent=4))
