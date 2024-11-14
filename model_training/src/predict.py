from ultralytics import YOLO

def predict_model(model_path, source):
    model = YOLO(model_path)
    results = model.predict(source=source)
    print("Prediction Results:", results)
