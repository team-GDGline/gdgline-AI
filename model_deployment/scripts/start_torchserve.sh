# TorchServe 종료
./scripts/stop_torchserve.sh

# 모델 저장소 디렉토리 생성
mkdir -p model-store

# TorchServe 시작 및 모델 등록
torchserve --start --model-store model-store --models yolo.mar --ts-config config/config.properties

echo "TorchServe가 시작되었습니다."
