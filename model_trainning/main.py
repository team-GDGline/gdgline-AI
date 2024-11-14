import argparse
from train import train_model
from evaluate import evaluate_model
from predict import predict_model
from export import export_model
from settings import show_settings

def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="YOLO model operations")

    # 필수 인자 (모드와 모델 경로)
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'predict', 'export', 'settings'],
                        help="Mode to run: 'train', 'evaluate', 'predict', 'export', or 'settings'")
    parser.add_argument('--model', type=str, required=True, help="Path to the YOLO model weights file")
    
    # 선택적 인자
    parser.add_argument('--data', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--imgsz', type=int, help="Image size for training and validation")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch', type=float, default=16, help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default='auto', help="Optimizer for training")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--cos_lr', type=bool, default=False, help="Use cosine learning rate scheduler")
    parser.add_argument('--verbose', type=bool, default=False, help="Enable verbose output during training")
    parser.add_argument('--source', type=str, help="Source path for prediction images")
    parser.add_argument('--format', type=str, default='torchscript', help="Export format for the model")

    # 인자 파싱
    args = parser.parse_args()

    # mode에 따른 함수 호출
    if args.mode == 'train':
        train_model(model_path=args.model, data_path=args.data, imgsz=args.imgsz, 
                    epochs=args.epochs, batch=args.batch, optimizer=args.optimizer, 
                    patience=args.patience, cos_lr=args.cos_lr, verbose=args.verbose)
        
    elif args.mode == 'evaluate':
        evaluate_model(model_path=args.model, data_path=args.data)
        
    elif args.mode == 'predict':
        predict_model(model_path=args.model, source=args.source)
        
    elif args.mode == 'export':
        export_model(model_path=args.model, format=args.format)
        
    elif args.mode == 'settings':
        show_settings()
        
    else:
        print("Invalid mode. Please select 'train', 'predict', 'export', 'evaluate', or 'settings'.")

# main() 함수 호출
if __name__ == "__main__":
    main()
