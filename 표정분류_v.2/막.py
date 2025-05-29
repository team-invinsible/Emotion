#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
from torchvision import transforms
from PIL import Image
from google.colab.patches import cv2_imshow

import sys
sys.path.append('/content/drive/MyDrive/jaein')
from models import getModel  # 여러분 __init__.py에 있는 함수

# —————————— 사용자 설정 ——————————
VIDEO_PATH   = '/content/drive/MyDrive/jaein/VideoData/2.mov'
MODEL_PATH   = '/content/drive/MyDrive/jaein/model_eff.pth'
CASCADE_PATH = '/content/drive/MyDrive/jaein/face_classifier.xml'
MODEL_NAME   = 'efficientnet-b5'
IMAGE_SIZE   = 224
# 한글 라벨 
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
# ————————————————————————————————


def build_model(model_name, ckpt_path, device):
    # 모델 생성
    model = getModel(model_name).to(device)
    # 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    # 0) 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) 모델
    model = build_model(MODEL_NAME, MODEL_PATH, device)

    # 2) 전처리
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # 3) 얼굴 검출기
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # 4) 비디오 열기
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # (웹캠용이라면 좌우 반전, 동영상이라면 주석 처리)
            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # 얼굴 하나씩
            for i, (x, y, w, h) in enumerate(faces):
                # ROI 자르고 PIL→Tensor
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE)))
                inp = transform(pil).unsqueeze(0).to(device)

                # 예측
                logits = model(inp)
                probs  = torch.softmax(logits, dim=1)[0]
                p, idx = probs.max(0)
                label = f"{class_labels[idx]} ({p*100:.1f}%)"

                # 콘솔에도 출력
                print(f"[Frame] face#{i}: {label}")

                # 화면에 출력
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if len(faces) == 0:
                cv2.putText(frame, 'No Face Found', (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # Colab 에서 화면 띄우기
            cv2_imshow(frame)

            # 'q' 누르면 루프 탈출
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()