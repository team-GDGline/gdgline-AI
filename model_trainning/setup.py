from setuptools import setup, find_packages

setup(
    name="model_training",
    version="0.1",
    packages=find_packages("src"),  # src 디렉터리를 기준으로 패키지 탐색
    package_dir={"": "src"},        # src 디렉터리를 루트로 인식
    install_requires=[
        "numpy>=1.23.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.6.0",
        "pillow>=7.1.2",
        "pyyaml>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "tqdm>=4.64.0",
        "psutil",
        "py-cpuinfo",
        "pandas>=1.1.4",
        "seaborn>=0.11.0",
        "ultralytics-thop>=2.0.0"
    ],
    entry_points={
        'console_scripts': [
            'model_training=main:main',  # main.py의 main() 함수를 실행하는 엔트리 포인트
        ]
    },
)
