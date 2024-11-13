import os
from PIL import Image

def filtering(path):
    print("ㅡ 필터링 시작 ㅡ")
    filtered_count = 0
    dir_name = os.path.join(path)
    os.makedirs(dir_name, exist_ok=True)
    for index, file_name in enumerate(os.listdir(dir_name)):
        try:
            file_path = os.path.join(dir_name, file_name)
            img = Image.open(file_path)
            if img.width < 351 and img.height < 351:
                img.close()
                os.remove(file_path)
                print(f"{index} 번째 사진 삭제")
                filtered_count += 1
        except OSError:
            os.remove(file_path)
            filtered_count += 1

    print(f"ㅡ 필터링 종료 (총 갯수: {filtered_count}) ㅡ")
    