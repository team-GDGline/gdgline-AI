from ultralytics import YOLO

def train_model(model_path, data_path, imgsz, epochs, batch, optimizer, patience, cos_lr, verbose):
    model = YOLO(model_path)
    model.train(data=data_path, imgsz=imgsz, epochs=epochs,
                batch=batch, optimizer=optimizer, 
                patience=patience, cos_lr=cos_lr, verbose=verbose)
    print("Training complete.")
