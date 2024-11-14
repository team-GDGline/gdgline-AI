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
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category

# 예측 수행
predict_model(model_path='yolo11n.pt', source='path/to/your/images_or_video') # 수정해서 사용

# # 모델 내보내기
# export_model(model_path='yolo11n.pt', format='onnx') # 수정해서 사용

# # 모델 평가
# evaluate_model(model_path='yolo11n.pt', data_path='data.yaml') # 수정해서 사용
