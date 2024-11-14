from ultralytics import YOLO

def evaluate_model(model_path, data_path):
    model = YOLO(model_path)
    metrics = model.val(data=data_path)
    print("Evaluation Metrics:")
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category