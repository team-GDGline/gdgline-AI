import yaml
from ultralytics import YOLO

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 모델 학습 함수
def train_model(config):
    # 모델 로드 및 설정
    model = YOLO(config.get('model', 'yolo11n.pt'))
    
    # 기본 파라미터 딕셔너리
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
        'cos_lr' : config.get('cos_lr', False)
    }
    
    # None이 아닌 파라미터만 필터링하여 전달
    train_params = {k: v for k, v in train_params.items() if v is not None}

    # 학습 시작
    model.train(**train_params)

# 설정 파일 경로
config_path = 'config.yaml'

# 설정 파일 로드 및 학습 시작
config = load_config(config_path)
train_model(config)
