import torch
import torchvision
import cv2
import time

# TorchScript 모델 로드
model_path = "best.torchscript"  # TorchScript 파일 경로
torchscript_model = torch.jit.load(model_path)
torchscript_model.eval()

# 이미지 경로
image_path = "image4-Photoroom.png"

# 원본 이미지를 로드하여 후에 바운딩 박스를 표시할 때 사용
original_img = cv2.imread(image_path)

# 이미지 로드 및 전처리 (cv2 사용)
def preprocess_image(image_path):
    # cv2로 이미지 로드 (BGR 형식)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # BGR → RGB 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 크기 조정
    img_resized = cv2.resize(img_rgb, (640, 640))

    # numpy 배열 → torch.Tensor 변환
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # (H, W, C) → (C, H, W)
    img_tensor = img_tensor / 255.0  # 정규화 [0, 1]

    # 배치 차원 추가
    img_tensor = img_tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
    return img_tensor

# 전처리 시작
start_time = time.time()
img_tensor = preprocess_image(image_path)
preprocess_time = time.time() - start_time  # 전처리 시간 측정

# 추론 실행
start_infer_time = time.time()
with torch.no_grad():
    outputs = torchscript_model(img_tensor)
inference_time = time.time() - start_infer_time  # 추론 시간 측정

# 후처리 함수 정의
def postprocess(outputs, conf_threshold=0.5, iou_threshold=0.4, num_classes=14):
    outputs = outputs[0]  # 배치 차원 제거 (shape: [84, 8400])

    # 바운딩 박스 좌표와 클래스 확률 분리
    boxes = outputs[:4, :].transpose(0, 1)  # 첫 4개 값은 바운딩 박스 좌표 ([8400, 4])
    class_scores = outputs[4:, :]  # 클래스 확률 ([80, 8400])

    # 각 바운딩 박스에서 최고 확률의 클래스 ID 및 해당 확률 추출
    class_confidences, class_labels = torch.max(class_scores, dim=0)
    mask = class_confidences > conf_threshold  # 신뢰도 임계값 적용
    
    # 필터링된 바운딩 박스 및 해당 클래스 정보
    filtered_boxes = boxes[mask]
    filtered_scores = class_confidences[mask]
    filtered_labels = class_labels[mask]

    # 비최대 억제 적용
    if filtered_boxes.size(0) > 0:
        indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold)
        final_boxes = filtered_boxes[indices]
        final_scores = filtered_scores[indices]
        final_labels = filtered_labels[indices]

        # 바운딩 박스 좌표 변환 (x_center, y_center, w, h → x1, y1, x2, y2)
        final_boxes[:, 0] = final_boxes[:, 0] - final_boxes[:, 2] / 2  # x_center - width / 2
        final_boxes[:, 1] = final_boxes[:, 1] - final_boxes[:, 3] / 2  # y_center - height / 2
        final_boxes[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]  # x_center + width / 2
        final_boxes[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]  # y_center + height / 2

        return final_boxes, final_scores, final_labels
    else:
        return [], [], []

# 후처리 시작
start_post_time = time.time()
boxes, scores, labels = postprocess(outputs, conf_threshold=0.5, iou_threshold=0.4, num_classes=80)
postprocess_time = time.time() - start_post_time  # 후처리 시간 측정

# 바운딩 박스 결과를 원본 이미지에 표시
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표
    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 바운딩 박스 그리기
    label_text = f"Class {label.item()}: {score:.2f}"  # 클래스 및 신뢰도 표시
    cv2.putText(original_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 이미지 출력
cv2.imshow("Detection Result", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 전체 처리 시간 출력
total_time = preprocess_time + inference_time + postprocess_time
print(f"Preprocessing Time: {preprocess_time:.4f} seconds")
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"Postprocessing Time: {postprocess_time:.4f} seconds")
print(f"Total Time: {total_time:.4f} seconds")
