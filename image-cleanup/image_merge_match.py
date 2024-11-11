import os
import shutil
import cv2


def merge_folders(folder1_path, folder2_path, output_folder_path):
    # 출력 폴더 생성
    os.makedirs(output_folder_path, exist_ok=True)

    # 첫 번째 폴더의 이미지 복사
    for image_name in os.listdir(folder1_path):
        src_path = os.path.join(folder1_path, image_name)
        dst_path = os.path.join(output_folder_path, image_name)
        shutil.copy(src_path, dst_path)
        print(f"{image_name}이(가) 첫 번째 폴더에서 복사되었습니다.")

    # 두 번째 폴더의 이미지 복사
    for image_name in os.listdir(folder2_path):
        src_path = os.path.join(folder2_path, image_name)
        dst_path = os.path.join(output_folder_path, image_name)
        shutil.copy(src_path, dst_path)
        print(f"{image_name}이(가) 두 번째 폴더에서 복사되었습니다.")


def template_match_images(base_image_path, image_folder_path, output_folder_path, threshold=0.8):
    # 기준 이미지 읽기
    base_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
    if base_image is None:
        print("기준 이미지를 불러올 수 없습니다.")
        return

    # 출력 폴더 생성
    os.makedirs(output_folder_path, exist_ok=True)

    # 이미지 폴더의 모든 이미지 순회
    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 이미지 로드 확인
        if image is None or image.shape[0] < base_image.shape[0] or image.shape[1] < base_image.shape[1]:
            continue

        # 템플릿 매칭 수행
        result = cv2.matchTemplate(image, base_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 유사도 확인 후 저장
        if max_val >= threshold:
            shutil.copy(image_path, output_folder_path)
            print(f"{image_name} 이미지가 기준 이미지와 유사하여 저장되었습니다.")
