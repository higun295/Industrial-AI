# 호환성에 어려움을 겪다가 성공한 설치 순서 및 과정
- docker를 사용하지 않았을 때의 방법
- docker를 사용하면 훨씬 편하게 할 수 있음(참고용!!)

## 설치 환경
- Anaconda 사용
- 파이썬 3.9
- GPU 사용
- PIP 사용

## 설치가 잘 되지 않을 때

#### 캐시 정리 및 env 초기화
```bash
conda clean --all
pip cache purge
```

#### 설치 순서
```
1. 가상환경 생성
conda create -n capstone_yolo python=3.9
conda activate capstone_yolo

2. PyTorch 설치(GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. ultralytics 설치
pip install ultralytics

4. 추가 라이브러리 설치
pip install opencv-python pillow matplotlib
```
