# Face Shape Classification

얼굴형을 분류하는 Vision Transformer 기반 딥러닝 모델입니다.

## Recognition part

ViT(Vision Transformer)를 활용한 얼굴형 분류 모델로, 정면 얼굴 이미지를 입력받아 아래 5가지 얼굴형을 분류합니다:
- 타원형(Oval)
- 하트형(Heart)
- 둥근형(Round)
- 각진형(Square)
- 긴형(Oblong)

## 모델 정보

이 저장소는 [metadome/face_shape_classification](https://huggingface.co/metadome/face_shape_classification) Hugging Face 모델을 사용하기 쉽도록 구성되었습니다.

- 모델 정확도: 85.3% (자체 평가 기준)
- 추론 시간: 평균 0.05초 (GPU 기준)

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/SoftwareProjectssu/AI.git
cd AI/face_classification

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 모델 파일 다운로드

이 코드를 실행하기 전에 Hugging Face에서 모델 파일을 다운로드해야 합니다:

```bash
# 모델 디렉토리 생성
mkdir -p models

# 모델 다운로드
curl -L https://huggingface.co/metadome/face_shape_classification/resolve/main/pytorch_model.bin -o models/model_85_nn_.pth
