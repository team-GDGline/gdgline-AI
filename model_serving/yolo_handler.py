import torch
import torchvision
import yaml
import base64
import cv2
import numpy as np
import os
from ts.torch_handler.base_handler import BaseHandler

class YOLOHandler(BaseHandler):
    def __init__(self):
        super(YOLOHandler, self).__init__()
        self.model = None
        self.class_names = []

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir", ".")
        model_path = os.path.join(model_dir, "best.torchscript")
        data_yaml_path = os.path.join(model_dir, "data.yaml")
        
        # TorchScript 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # 클래스 이름 로드
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data configuration file not found at: {data_yaml_path}")
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.class_names = data.get("names", [])
            if not isinstance(self.class_names, list):
                raise ValueError("The 'names' field in data.yaml must be a list of class names.")

    def preprocess(self, data):
        if not data or "body" not in data[0]:
            raise ValueError("No data found in the request.")
        body = data[0]["body"]
        if isinstance(body, (bytes, bytearray)):
            body = body.decode('utf-8')
        image_data = body.get("image")
        if not image_data:
            raise ValueError("'image' field is required in the request.")
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        try:
            img_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            return img_tensor
        except Exception as e:
            raise ValueError(f"Failed to process the image data: {e}")

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def postprocess(self, outputs):
        conf_threshold = 0.5
        iou_threshold = 0.5
        
        outputs = outputs[0]  # 배치 차원 제거

        # 바운딩 박스 좌표와 클래스 확률 분리
        boxes = outputs[:4, :].transpose(0, 1)  # 첫 4개 값은 바운딩 박스 좌표
        class_scores = outputs[4:, :]  # 클래스 확률

        # 각 바운딩 박스에서 최고 확률의 클래스 ID 및 해당 확률 추출
        class_confidences, class_labels = torch.max(class_scores, dim=0)
        mask = class_confidences > conf_threshold  # 신뢰도 임계값 적용
        
        # 필터링된 바운딩 박스 및 해당 클래스 정보
        filtered_boxes = boxes[mask]
        filtered_scores = class_confidences[mask]
        filtered_labels = class_labels[mask]

        # 비최대 억제 적용
        if filtered_boxes.size(0) > 0:
            indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold)
            final_boxes = filtered_boxes[indices]
            final_scores = filtered_scores[indices]
            final_labels = filtered_labels[indices]

            # 바운딩 박스 좌표 변환 (x_center, y_center, w, h → x1, y1, x2, y2)
            final_boxes[:, 0] = final_boxes[:, 0] - final_boxes[:, 2] / 2  # x_center - width / 2
            final_boxes[:, 1] = final_boxes[:, 1] - final_boxes[:, 3] / 2  # y_center - height / 2
            final_boxes[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]  # x_center + width / 2
            final_boxes[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]  # y_center + height / 2

            # 결과 준비
            results = list(set([self.class_names[class_id.item()] for class_id in final_labels]))
            
            return {"detections": results}
        else:
            return {"detections": []}

    def handle(self, data, context):
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            result = self.postprocess(model_output)
            return [result]
        except Exception as e:
            return [{"error": str(e)}]
