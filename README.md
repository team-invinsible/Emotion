# 감정 인식 프로젝트 (Emotion Recognition)

딥러닝을 활용한 실시간 얼굴 표정 인식 시스템

## 📖 프로젝트 소개

이 프로젝트는 컴퓨터 비전과 딥러닝 기술을 사용하여 얼굴 표정에서 감정을 인식하는 시스템입니다. 7가지 감정(기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립)을 실시간으로 분류할 수 있습니다.

## 🎯 주요 기능

- **실시간 감정 인식**: 웹캠을 통한 실시간 얼굴 표정 분석
- **7가지 감정 분류**: 기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립
- **높은 정확도**: EfficientNet-b5 모델로 83% 이상의 정확도 달성
- **다양한 입력 방식**: 이미지 파일, 웹캠, 비디오 지원

## 📁 프로젝트 구조

```
Emotion/
├── README.md                 # 프로젝트 설명서
├── (원)README.md            # 원본 기술 문서
├── model_eff.pth           # 학습된 EfficientNet 모델
├── export/                 # 실행 가능한 메인 프로그램
│   ├── emotion.py          # 이미지 감정 인식
│   ├── webcam_emotion.py   # 웹캠 실시간 인식
│   ├── last.py            # 통합 인식 프로그램
│   ├── model.pth          # 모델 파일
│   ├── requirements.txt   # 필요 라이브러리
│   └── utils.py           # 유틸리티 함수
├── 표정분류_v.2/           # 개발 버전
├── mv/                    # 이동된 파일들
└── 임시파일/               # 임시 작업 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 클론 후 export 디렉토리로 이동
cd export

# 가상환경 생성 및 활성화 (선택사항)
python -m venv myvenv
source myvenv/bin/activate  # Windows: myvenv\Scripts\activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2. 이미지 파일로 감정 인식

```bash
python emotion.py --img 이미지파일.jpg
```

### 3. 웹캠으로 실시간 감정 인식

```bash
python webcam_emotion.py
```

### 4. 통합 프로그램 실행

```bash
python last.py
```

## 💻 사용 예시

### 이미지 파일 분석
```bash
$ python emotion.py --img happy.jpg
[{'label': '기쁨', 'probs': [0.0, 0.0, 1.16, 0.09, 0.27, 98.48, 0.0]}]
```

### 얼굴 검출과 함께 분석
```bash
$ python emotion.py --img sample.jpg --detect_face
[{'rect': '(886, 292, 1324, 1324)', 'label': '기쁨', 'probs': [0.0, 0.0, 1.16, 0.09, 0.27, 98.48, 0.0]}]
```

## 🎨 지원하는 감정

| 번호 | 감정 | 설명 |
|------|------|------|
| 0 | 기쁨 | 행복하고 즐거운 표정 |
| 1 | 당황 | 놀라거나 당황한 표정 |
| 2 | 분노 | 화나고 격분한 표정 |
| 3 | 불안 | 걱정되고 불안한 표정 |
| 4 | 상처 | 아프고 상처받은 표정 |
| 5 | 슬픔 | 슬프고 우울한 표정 |
| 6 | 중립 | 무표정하고 평온한 표정 |

## 📊 모델 성능

| 모델 | 검증 정확도 | 테스트 정확도 | 모델 크기 |
|------|-------------|---------------|-----------|
| EfficientNet-b5 | 83.356% | 83.028% | 350MB |
| VGG24 (deep-wide) | 83.272% | 81.393% | 610MB |
| EfficientNet-b4 | 83.328% | 82.112% | 223MB |
| ResNet18 | 82.351% | 80.941% | 128MB |

## 🛠️ 주요 매개변수

### emotion.py 옵션
- `--img`: 입력 이미지 파일명
- `--model_path`: 모델 파일 경로 (기본값: model.pt)
- `--gpu`: GPU 사용 여부 (기본값: True)
- `--detect_face`: 얼굴 검출 수행 여부 (기본값: False)

### webcam_emotion.py 옵션
- 실시간 웹캠 입력으로 감정 인식
- ESC 키로 종료
- 검출된 얼굴에 감정 라벨과 확률 표시

## 🔧 개발 환경

- **운영체제**: Ubuntu 18.04 (개발), macOS (호환)
- **Python**: 3.7+
- **PyTorch**: 1.7.1+
- **OpenCV**: 얼굴 검출용
- **PIL**: 8.1.0+ (이미지 처리)

## 📚 추가 문서

자세한 기술 문서와 학습 과정은 `(원)README.md` 파일을 참조하세요.

## 🏷️ 태그

`#감정인식` `#딥러닝` `#컴퓨터비전` `#PyTorch` `#OpenCV` `#실시간` `#얼굴표정`
