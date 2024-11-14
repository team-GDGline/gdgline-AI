import torch
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image, ImageOps
import io
import json

class YOLOHandler(BaseHandler):
    def __init__(self):
        super(YOLOHandler, self).__init__()
        self.model = None

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        model_path = f"{model_dir}/yolo_model_scripted.pt"
        
        # TorchScript 모델 로드
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def preprocess(self, data):
        # 이미지 데이터 불러오기
        image = Image.open(io.BytesIO(data[0].get("data") or data[0].get("body"))).convert("RGB")
        
        # 1. Auto-Orient: 이미지 방향 자동 수정
        image = ImageOps.exif_transpose(image)

        # 2. Resize to 640x640
        image = image.resize((640, 640))

        # 3. Auto-Adjust Contrast (Contrast Stretching)
        # 픽셀 값 범위를 0~255로 스트레칭
        image = ImageOps.autocontrast(image)

        # 텐서 변환 및 배치 차원 추가
        image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # [1, C, H, W]
        return image

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def postprocess(self, outputs):
        # 출력에서 클래스 종류만 추출하여 JSON으로 반환
        detected_classes = set()
        for output in outputs[0]:  # 모델 출력 구조에 따라 수정 가능
            detected_class = output['class']
            detected_classes.add(detected_class)

        return json.dumps({"detected_classes": list(detected_classes)})
