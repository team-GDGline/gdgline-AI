import yaml
from ultralytics import YOLO

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 모델 학습 함수
def train_model(config):
    model = YOLO(config.get('model', 'yolo11n.pt'))
    train_params = {
        'data': config.get('data'),
        'epochs': config.get('epochs', 100),
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'patience': config.get('patience', 100),
        'device': config.get('device'),
        'optimizer': config.get('optimizer', 'auto'),
        'seed': config.get('seed', 0),
        'pretrained': config.get('pretrained', True),
        'save': config.get('save', True),
        'val': config.get('val', True),
        'amp': config.get('amp', True),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'cos_lr': config.get('cos_lr', False)
    }
    train_params = {k: v for k, v in train_params.items() if v is not None}
    model.train(**train_params)
    print("Training complete.")

# 학습 재개 함수 (resume)
def resume_training(config):
    model = YOLO(config.get('model', 'yolo11n.pt'))  # 가중치 경로만 필요
    model.train(resume=True)
    print("Resumed training complete.")

# 추가 학습 함수 (continue)
def continue_training(config):
    model = YOLO(config.get('model', 'yolo11n.pt'))
    additional_epochs = config.get('additional_epochs', 50)
    train_params = {
        'data': config.get('data'),
        'epochs': additional_epochs,
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'patience': config.get('patience', 100),
        'device': config.get('device'),
        'optimizer': config.get('optimizer', 'auto'),
        'seed': config.get('seed', 0),
        'pretrained': config.get('pretrained', True),
        'save': config.get('save', True),
        'val': config.get('val', True),
        'amp': config.get('amp', True),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'cos_lr': config.get('cos_lr', False)
    }
    train_params = {k: v for k, v in train_params.items() if v is not None}
    model.train(**train_params)
    print("Continued training complete.")

# 설정 파일 경로 및 로드
config_path = 'config.yaml'
config = load_config(config_path)

# 모드 선택에 따른 기능 수행
mode = config.get('mode', 'train')

if mode == 'train':
    train_model(config)
elif mode == 'resume':
    resume_training(config)
elif mode == 'continue':
    continue_training(config)
else:
    print("Invalid mode specified in config.yaml. Please select 'train', 'resume', or 'continue'.")
