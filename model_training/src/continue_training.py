from ultralytics import YOLO

def continue_train_model(model_path, data_path, additional_epochs, imgsz, batch, optimizer, patience, cos_lr, verbose):
    model = YOLO(model_path)
    model.train(data=data_path, imgsz=imgsz, epochs=additional_epochs,
                batch=batch, optimizer=optimizer, 
                patience=patience, cos_lr=cos_lr, verbose=verbose)
    print("Continued training complete.")
