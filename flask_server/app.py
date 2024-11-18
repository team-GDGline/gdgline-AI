import torch
import torchvision
import torchvision.ops
import yaml
import base64
import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # 추가된 부분
from io import BytesIO

app = Flask(__name__)

# CORS 설정 추가
CORS(app, resources={r"/*": {"origins": "http://mulmung.s3-website.ap-northeast-2.amazonaws.com"}})

# 모델 로드 및 초기화
def initialize_model():
    try:
        model_dir = "./"  # 모델 디렉토리 경로
        model_path = os.path.join(model_dir, "best.torchscript")
        data_yaml_path = os.path.join(model_dir, "data.yaml")
        
        # TorchScript 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = torch.jit.load(model_path)
        model.eval()
        
        # 클래스 이름 로드
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data configuration file not found at: {data_yaml_path}")
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            class_names = data.get("names", [])
            if not isinstance(class_names, list):
                raise ValueError("The 'names' field in data.yaml must be a list of class names.")
        
        return model, class_names
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

# 이미지 전처리
# 이미지 전처리
def preprocess(image_data):
    try:
        # Base64 디코딩 및 이미지 로드
        img_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 원본 이미지 크기
        original_height, original_width = img_rgb.shape[:2]

        # 패딩을 추가하여 비율 유지하면서 정사각형 만들기
        scale = 640 / max(original_width, original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)
        resized_img = cv2.resize(img_rgb, (resized_width, resized_height))

        # 패딩 계산
        pad_x = (640 - resized_width) // 2
        pad_y = (640 - resized_height) // 2

        # 패딩 추가
        padded_img = cv2.copyMakeBorder(
            resized_img, 
            pad_y, 
            640 - resized_height - pad_y, 
            pad_x, 
            640 - resized_width - pad_x, 
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)  # YOLO의 기본 패딩 값
        )

        # 텐서로 변환 및 정규화
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return img_tensor

    except Exception as e:
        raise ValueError(f"Failed to process the image data: {e}")


# 모델 추론
def inference(model, inputs):
    with torch.no_grad():
        return model(inputs)

# 후처리
def postprocess(outputs, class_names):
    conf_threshold = 0.5
    iou_threshold = 0.5
    
    outputs = outputs[0]
    boxes = outputs[:4, :].transpose(0, 1)
    class_scores = outputs[4:, :]
    
    class_confidences, class_labels = torch.max(class_scores, dim=0)
    mask = class_confidences > conf_threshold
    
    filtered_boxes = boxes[mask]
    filtered_scores = class_confidences[mask]
    filtered_labels = class_labels[mask]

    if filtered_boxes.size(0) > 0:
        indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold)
        final_boxes = filtered_boxes[indices]
        final_scores = filtered_scores[indices]
        final_labels = filtered_labels[indices]

        final_boxes[:, 0] = final_boxes[:, 0] - final_boxes[:, 2] / 2
        final_boxes[:, 1] = final_boxes[:, 1] - final_boxes[:, 3] / 2
        final_boxes[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]
        final_boxes[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]

        results = list(set([class_names[class_id.item()] for class_id in final_labels]))
        
        return {"detections": results}
    else:
        return {"detections": []}

# 모델 초기화
model, class_names = initialize_model()

@app.route("/", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            return jsonify({"status": "OK"}), 200
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "'image' field is required."}), 400
        
        image_data = data.get("image")

        model_input = preprocess(image_data)
        
        model_output = inference(model, model_input)

        result = postprocess(model_output, class_names)

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)