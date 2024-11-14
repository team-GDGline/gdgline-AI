from ultralytics import YOLO

def export_model(model_path, format):
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Model exported as {format} format for deployment")
