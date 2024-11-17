#!/bin/bash

# 모델 아카이브 생성
torch-model-archiver \
    --model-name yolo \
    --version 1.0 \
    --serialized-file best.torchscript \
    --handler yolo_handler.py \
    --extra-files "data.yaml" \
    --export-path model-store \
    --force

echo "YOLO 모델 아카이브 생성 완료: model-store/yolo.mar"