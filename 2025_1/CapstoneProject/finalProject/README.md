
# 자동차 부품용 나사 이종품 검출을 위한 데이터 드리프트 대응형 딥러닝 모델

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Inception-ResNet](https://img.shields.io/badge/Model-Inception--ResNet-blue)

> **Inception-ResNet 기반 듀얼 분류 모델을 활용한 자동차 나사 이종품 검출 시스템**

## 📋 프로젝트 개요

자동차 제조 공정에서 발생하는 나사 이종품 혼입 문제를 해결하기 위한 AI 기반 비전 검사 시스템입니다. 환경 변화로 인한 데이터 드리프트 문제를 듀얼 모델 구조로 해결하여 안정적인 분류 성능을 달성했습니다.

### 🎯 주요 성과
- **정확도 향상**: 99.42% → 99.93% (0.51% 개선)
- **오검출 감소**: NG 이미지 326장 → 40장 (87.73% 감소)
- **안정성 확보**: 데이터 드리프트 상황에서도 안정적인 성능 유지

## 🔍 문제 정의

### 기존 문제점
1. **모델 드리프트 현상**: 카메라 위치 변동으로 인한 성능 저하
2. **미세한 분류 어려움**: 나사 각도 차이 인식률 저하
3. **단일 모델 한계**: 환경 변화에 대한 대응력 부족

### 해결 목표
- 27산 30도, 27산 45도, 28산, 31산 나사의 정확한 분류
- 카메라 위치 변화에 강건한 모델 구축
- 실시간 검사 환경에 적합한 시스템 개발

## 🏗️ 시스템 아키텍처

### 듀얼 모델 프로세스
```
[입력 이미지] 
    ↓
[1차 모델: 기본 분류기]
    ↓
[신뢰도 검사 (confidence < 0.9 or NG)]
    ↓
[2차 모델: 보완 분류기] → [최종 결과]
```

### 모델 구조
- **기본 모델**: Inception-ResNet v2
- **특징**: 
  - 멀티 스케일 특징 추출 (1x1, 3x3, 5x5 필터)
  - 잔차 연결(Residual Connection)로 학습 안정화
  - Fine-grained classification에 최적화

## 📊 데이터셋

### 학습 데이터
- **수집 기간**: 2023-11-22 ~ 2024-01-27, 2024-12-02 ~ 2024-12-12
- **총 데이터**: 114,217장
- **클래스별 분포**:
  - 27산 30도: 33,431장
  - 27산 45도: 13,255장
  - 28산: 60,939장
  - 31산: 6,592장

### 테스트 데이터
- **수집 기간**: 2024-12-18 ~ 2025-01-10
- **총 데이터**: 56,792장

## ⚙️ 실험 설정

### 하드웨어 환경
- **OS**: Ubuntu 22.04.4 LTS
- **CPU**: AMD Ryzen Threadripper PRO 5975WX 32-Cores
- **GPU**: NVIDIA RTX 5000 Ada Generation 32GB
- **RAM**: 256GB

### 하이퍼파라미터
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam
- **Input Size**: 299x299 (Inception-ResNet 입력 크기)

## 📈 실험 결과

### 성능 비교

| 구분 | 적용 전 | 모델 1개 적용 | 모델 2개 적용 |
|------|---------|---------------|---------------|
| 평균 정확도 | 99.42% | 99.91% | 99.93% |
| NG 이미지 수 | 326장 | 52장 | 40장 |

### Confusion Matrix 분석
- **27산 각도 분류 개선**: 가장 큰 성능 향상 확인
- **모델 1개 → 2개**: 각도 분류 NG 수 12개 → 6개 (50% 감소)

## 🔧 설치 및 사용법

### 필요 라이브러리
```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
```

### 기본 사용 예시
```python
# 모델 로드
model_1 = load_model('first_model.h5')
model_2 = load_model('second_model.h5')

# 듀얼 모델 추론
def dual_model_predict(image):
    # 1차 분류
    pred_1 = model_1.predict(image)
    confidence = np.max(pred_1)
    
    # 신뢰도 검사
    if confidence < 0.9 or np.argmax(pred_1) == NG_CLASS:
        # 2차 분류
        pred_2 = model_2.predict(image)
        return pred_2
    
    return pred_1
```

## 📝 주요 기여점

1. **데이터 드리프트 대응**: 실제 제조 환경에서 발생하는 드리프트 현상 분석 및 해결
2. **듀얼 모델 시스템**: 신뢰도 기반 2단계 분류로 정확도 향상
3. **미세 분류 성능**: Inception-ResNet 구조로 나사 각도 차이 정확 분류
4. **실용성 검증**: 실제 생산 라인 데이터로 성능 검증

## 🚀 향후 연구 방향

### 단기 목표
- **온라인 학습**: 실시간 환경 변화 적응 시스템 개발
- **모델 최적화**: 하이퍼파라미터 튜닝을 통한 성능 향상
- **신뢰도 기반 앙상블**: 다양한 조합 방식 연구

### 장기 목표
- **적응형 분류 모델**: 지속 학습 구조 설계
- **경량화**: 실시간 검사를 위한 모델 경량화
- **다른 부품 확장**: 나사 외 다른 자동차 부품 검출 시스템 확장

## 🔗 참고 자료

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (2016)](https://arxiv.org/abs/1602.07261)
- [A Study on Efficient AI Model Drift Detection Methods for MLOps (2023)](https://example.com)
- [Development of the AI-fact Image Inspection System (2024)](https://example.com)

## 👥 팀 정보

**2조 - 충북대학교 산업인공지능학과**
- **팀장**: 신희권 (2024254008@cbnu.ac.kr)
- **팀원**: 박금나 (pgn@cbnu.ac.kr)
- **팀원**: 김다현 (dahyeon713@cbnu.ac.kr)

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ 태그

`Computer Vision` `Quality Control` `Manufacturing` `Deep Learning` `Data Drift` `Inception-ResNet` `Automotive Industry`
