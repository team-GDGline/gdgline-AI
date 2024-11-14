import argparse
from train import train_model
from evaluate import evaluate_model
from predict import predict_model
from export import export_model
from settings import show_settings
from resume_training import resume_train_model
from resume_training import resume_train_model
from continue_training import continue_train_model

def main():
    parser = argparse.ArgumentParser(description="YOLO model operations")

    # 필수 인자 (모드와 모델 경로)
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'predict', 'export', 'settings', 'resume', 'continue'],
                        help="Mode to run: 'train', 'evaluate', 'predict', 'export', 'settings', 'resume', or 'continue'")
    parser.add_argument('--model', type=str, required=True, help="Path to the YOLO model weights file")
    
    # 선택적 인자 (학습 및 평가에 필요한 인자들)
    parser.add_argument('--data', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training and validation")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch', type=float, default=16, help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default='auto', help="Optimizer for training")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--cos_lr', action='store_true', help="Use cosine learning rate scheduler")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output during training")
    parser.add_argument('--additional_epochs', type=int, default=50, help="Number of additional epochs for continue training")

    # 예측 및 내보내기 시 필요한 인자
    parser.add_argument('--source', type=str, help="Source path for prediction images")
    parser.add_argument('--format', type=str, default='torchscript', help="Export format for the model")

    args = parser.parse_args()

    # mode에 따른 함수 호출
    if args.mode == 'train':
        train_model(
            model_path=args.model,
            data_path=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            optimizer=args.optimizer,
            patience=args.patience,
            cos_lr=args.cos_lr,
            verbose=args.verbose
        )
    elif args.mode == 'resume':
        resume_train_model(model_path=args.model)
    elif args.mode == 'continue':
        continue_train_model(
            model_path=args.model,
            data_path=args.data,
            additional_epochs=args.additional_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            patience=args.patience,
            cos_lr=args.cos_lr,
            verbose=args.verbose
        )
    elif args.mode == 'evaluate':
        evaluate_model(model_path=args.model, data_path=args.data)
    elif args.mode == 'predict':
        if not args.source:
            print("Error: --source is required for prediction.")
            return
        predict_model(model_path=args.model, source=args.source)
    elif args.mode == 'export':
        export_model(model_path=args.model, format=args.format)
    elif args.mode == 'settings':
        show_settings()
    else:
        print("Invalid mode. Please select 'train', 'resume', 'continue', 'predict', 'export', 'evaluate', or 'settings'.")

if __name__ == "__main__":
    main()
