from ultralytics import YOLO

# 예측 함수
def predict_model(model_path, source):
    model = YOLO(model_path)
    results = model.predict(source=source)
    print("Prediction Results:", results)

# 모델 내보내기 함수
def export_model(model_path, format):
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Model exported as {format} format for deployment")

# 모델 평가 함수
def evaluate_model(model_path, data_path):
    model = YOLO(model_path)
    metrics = model.val(data=data_path)
    print("Evaluation Metrics:")
    print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.precision:.4f}")
    print(f"  - Recall: {metrics.box.recall:.4f}")
    print(f"  - F1-Score: {metrics.box.f1:.4f}")

# 예측 수행
predict_model(model_path='yolo11n.pt', source='path/to/your/images_or_video') # 수정해서 사용

# # 모델 내보내기
# export_model(model_path='yolo11n.pt', format='onnx') # 수정해서 사용

# # 모델 평가
# evaluate_model(model_path='yolo11n.pt', data_path='data.yaml') # 수정해서 사용
