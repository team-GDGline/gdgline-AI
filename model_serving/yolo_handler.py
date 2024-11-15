import torch
import torchvision
import yaml
import base64
import cv2
import numpy as np
import json
import os
from ts.torch_handler.base_handler import BaseHandler

class YOLOHandler(BaseHandler):
    def __init__(self):
        super(YOLOHandler, self).__init__()
        self.model = None
        self.class_names = []

    def initialize(self, context):
        try:
            # TorchScript 모델 경로 설정
            model_path = "best.torchscript"
            
            # TorchScript 모델 로드
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            self.model = torch.jit.load(model_path)
            self.model.eval()
            
            # 데이터 설정 파일 경로
            data_yaml_path = "data.yaml"
            
            # 클래스 이름 로드
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"Data configuration file not found at: {data_yaml_path}")
            
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get("names", [])
                if not isinstance(self.class_names, list):
                    raise ValueError("The 'names' field in data.yaml must be a list of class names.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
        
    def preprocess(self, data):
        """
        Decode Base64 image and prepare tensor input.
        """
        image_data = data[0].get("image")
        if not image_data:
            raise ValueError("'image' field is required in the request.")
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        img_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        return torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def inference(self, inputs):
        """
        Run inference on the model.
        """
        with torch.no_grad():
            return self.model(inputs)

    def postprocess(self, outputs):
        """
        Extract class names and bounding boxes from the model output.
        Args:
            outputs (torch.Tensor): Model outputs, assumed shape [batch_size, num_outputs].
            conf_threshold (float): Confidence threshold for filtering predictions.
            iou_threshold (float): IoU threshold for non-max suppression (NMS).
        Returns:
            dict: Detected classes, scores, and bounding boxes.
        """
        conf_threshold=0.5
        iou_threshold=0.5
        
        outputs = outputs[0]  # Assuming batch size of 1
        boxes = outputs[:4, :].T  # Extract bounding boxes [N, 4]
        scores = outputs[4:, :].T  # Extract class scores [N, num_classes]
        
        # Find the best class for each box and its corresponding confidence score
        class_scores, class_ids = torch.max(scores, dim=1)  # [N]
        mask = class_scores > conf_threshold  # Filter by confidence threshold
        
        # Apply the mask to filter boxes, scores, and class IDs
        filtered_boxes = boxes[mask]  # [M, 4]
        filtered_scores = class_scores[mask]  # [M]
        filtered_class_ids = class_ids[mask]  # [M]
        
        # Apply Non-Maximum Suppression (NMS)
        if filtered_boxes.size(0) > 0:
            nms_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold)
            final_boxes = filtered_boxes[nms_indices]
            final_scores = filtered_scores[nms_indices]
            final_class_ids = filtered_class_ids[nms_indices]
            
            # Convert bounding boxes from center format to corner format (x_center, y_center, w, h → x1, y1, x2, y2)
            final_boxes[:, 0] = final_boxes[:, 0] - final_boxes[:, 2] / 2  # x1
            final_boxes[:, 1] = final_boxes[:, 1] - final_boxes[:, 3] / 2  # y1
            final_boxes[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]  # x2
            final_boxes[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]  # y2
            
            # Prepare the results
            results = list(set([self.class_names[class_id.item()] for class_id in final_class_ids]))
            
            return {"detections": results}
        else:
            return {"detections": []}

    def handle(self, data, context):
        """
        Main handler entry point.
        """
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            return self.postprocess(model_output)
        except Exception as e:
            return json.dumps({"error": str(e)})