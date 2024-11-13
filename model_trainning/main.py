import argparse
from ultralytics import YOLO
from ultralytics import settings

def main(args):
    # YOLO 모델 초기화
    model = YOLO(args.model)

    if args.mode == 'train':
        # 학습 실행
        model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs,
                    batch=args.batch, optimizer=args.optimizer, 
                    patience=args.patience, cos_lr=args.cos_lr,  verbose=args.verbose)
        print("Training complete.")

    elif args.mode == 'evaluate':
        # 모델 평가 실행
        metrics = model.val(data=args.data)
        
        # 평가 지표 출력
        print("Evaluation Metrics:")
        print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
        print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  - Precision: {metrics.box.precision:.4f}")
        print(f"  - Recall: {metrics.box.recall:.4f}")
        print(f"  - F1-Score: {metrics.box.f1:.4f}")
        
    elif args.mode == 'predict':
        # 예측 실행
        results = model.predict(source=args.source)
        print("Prediction Results:", results)
        
    elif args.mode == 'export':
        # 모델 내보내기, format 옵션 지원
        model.export(format=args.format)
        print(f"Model exported as {args.format} format for deployment")

    elif args.mode == 'settings':
        # settings 값 출력
        print("YOLO Settings:")
        print(settings)

    else:
        print("Invalid mode. Please select 'train', 'predict', 'export', 'evaluate', or 'settings'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO model operations")


    parser.add_argument('--mode', type=str, required = True, choices=['train', 'evaluate', 'predict', 'export', 'settings'],
                        help="Mode to run: 'train', 'evaluate', 'predict', 'export', or 'settings'")
    parser.add_argument('--model', type=str, required = True, help="Path to the YOLO model weights file")
    
    parser.add_argument('--data', type=str, help="Path to the dataset configuration file")
    parser.add_argument('--imgsz', type=int, help="Image size for training and validation")

    # 학습 파라미터 인자
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default='auto', help="Optimizer for training (e.g., 'auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp')")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--cos_lr', type=bool, default=False, help="Use cosine learning rate scheduler")
    parser.add_argument('--verbose', type=bool, default=False, help="Enable verbose output during training")
    
    # 예측 데이터 인자
    parser.add_argument('--source', type=str, help="Source path for prediction images")
    
    # export 인자
    parser.add_argument('--format', type=str, default='torchscript', help="Export format for the model (e.g., 'torchscript', 'onnx', 'coreml')")

    args = parser.parse_args()

    # 각 모드에서 필요한 인자 확인
    if args.mode in ['train', 'evaluate'] and not args.data:
        parser.error("The --data argument is required when mode is set to 'train' or 'evaluate'.")
    if args.mode == 'train' and not args.imgsz:
        parser.error("The --imgsz argument is required when mode is set to 'train'.")
    if args.mode == 'predict' and not args.source:
        parser.error("The --source argument is required when mode is set to 'predict'.")
    if args.mode != 'settings' and not args.model:
        parser.error("The --model argument is required for all modes except 'settings'.")

    main(args)
