markdown# 호환성에 어려움을 겪다가 성공한 설치 순서 및 과정

## 설치 환경
- Anaconda 사용
- 파이썬 3.9
- GPU 사용
- PIP 사용

## 설치가 잘 되지 않을 때, 캐시 정리 및 env 초기화
```bash
conda clean --all
pip cache purge
설치 순서
1. 가상환경 생성
bashconda create -n capstone_yolo python=3.9
conda activate capstone_yolo
2. PyTorch 설치(GPU)
bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. ultralytics 설치
bashpip install ultralytics
4. 추가 라이브러리 설치
bashpip install opencv-python pillow matplotlib