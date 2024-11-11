import os
import hashlib

# 중복 이미지 탐색 및 삭제 함수
def find_duplicate_images(folder_path):
    hash_dict = {}
    duplicate_files = []
    total_files = 0

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', 'webp')):
                total_files += 1
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # 첫 번째로 발견된 파일은 남기고 이후 중복 파일만 추가
                if file_hash in hash_dict:
                    duplicate_files.append(file_path)
                else:
                    hash_dict[file_hash] = file_path  # 해시값과 파일 경로 저장 (첫 파일만 남김)

    return duplicate_files, total_files


def delete_duplicates(duplicate_files):
    deleted_files = 0
    for file_path in duplicate_files:
        os.remove(file_path)
        deleted_files += 1
        print(f"Deleted: {file_path}")
    return deleted_files


def delete_image_from_path(folder_path):
    duplicate_files, total_files = find_duplicate_images(folder_path)
    if duplicate_files:
        print(f"{folder_path} 폴더에서 중복 이미지 파일이 발견되었습니다. 삭제를 시작합니다.")
        deleted_files = delete_duplicates(duplicate_files)
        print(f"{folder_path} 폴더의 중복 이미지 삭제 완료 (삭제된 파일 수: {deleted_files})")
    else:
        print(f"{folder_path} 폴더에 중복 이미지 파일이 없습니다.")
        return
    
    print(f"전체 파일 수: {total_files}, 중복 제거 후 남은 파일 수: {total_files - deleted_files}")