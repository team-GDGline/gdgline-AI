from ultralytics import YOLO

def resume_train_model(model_path):
    model = YOLO(model_path)
    model.train(resume=True)
    print("Resumed training complete.")
