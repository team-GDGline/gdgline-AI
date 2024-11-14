from ultralytics import YOLO

def evaluate_model(model_path, data_path):
    model = YOLO(model_path)
    metrics = model.val(data=data_path)
    print("Evaluation Metrics:")
    print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.precision:.4f}")
    print(f"  - Recall: {metrics.box.recall:.4f}")
    print(f"  - F1-Score: {metrics.box.f1:.4f}")
