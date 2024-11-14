import torch
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image, ImageOps
import io
import json
import yaml

class YOLOHandler(BaseHandler):
    def __init__(self):
        super(YOLOHandler, self).__init__()
        self.model = None
        self.class_names = {}

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        model_path = f"{model_dir}/model/best.torchscript"
        
        # TorchScript 모델 로드
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # 클래스 이름 로드
        with open(f"{model_dir}/data.yaml", 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.class_names = data.get("names", {})

    def preprocess(self, data):
        # 이미지 데이터 불러오기
        image = Image.open(io.BytesIO(data[0].get("data") or data[0].get("body"))).convert("RGB")
        
        # 1. Auto-Orient: 이미지 방향 자동 수정
        image = ImageOps.exif_transpose(image)

        # 2. Resize to 640x640
        image = image.resize((640, 640))

        # 3. Auto-Adjust Contrast (Contrast Stretching)
        image = ImageOps.autocontrast(image)

        # 텐서 변환 및 배치 차원 추가
        image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return image

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def postprocess(self, outputs):
        # 출력에서 클래스 ID를 이름으로 변환하여 JSON으로 반환
        detected_classes = set()
        for output in outputs[0]:
            detected_class_id = output['class']
            detected_class_name = self.class_names.get(detected_class_id, "Unknown")
            detected_classes.add(detected_class_name)

        return json.dumps({"detected_classes": list(detected_classes)})
