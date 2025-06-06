#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
from torchvision import transforms
from PIL import Image
# from google.colab.patches import cv2_imshow  # Colab 전용
from models import getModel  
import os
import time  # FPS 측정을 위해 추가

# —————————— 사용자 설정 ——————————
VIDEO_PATH   = '/Users/ijaein/Desktop/Emotion/export/video/화면 기록 2025-05-26 오후 2.45.18.mp4'
MODEL_PATH   = '/Users/ijaein/Desktop/Emotion/model_eff.pth'  # EfficientNet 모델 가중치 파일
CASCADE_PATH = '/Users/ijaein/Desktop/Emotion/export/face_classifier.xml'
MODEL_NAME   = 'efficientnet-b5'  # EfficientNet-b5 사용
IMAGE_SIZE   = 224  

# 속도 최적화 설정 (시간 기반 버전)
ANALYSIS_INTERVAL = 1.0  # 1초마다 1번 분석
PLAYBACK_SPEED = 5     # 비디오 재생 속도 (2.5배속)

# 추가 최적화 옵션들
FAST_FACE_DETECTION = True  # 빠른 얼굴 검출 모드
USE_LIGHTER_MODEL = False   # 더 가벼운 CNN 모델 사용 (True로 설정하면 CNN 사용)

# 한글 라벨 
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
# ————————————————————————————————


def build_model(model_name, ckpt_path):
    # 모델 생성
    model = getModel(model_name)
    
    # 체크포인트 로드 (가중치 파일이 있는 경우만)
    try:
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('model', ckpt)
            model.load_state_dict(state)
            print(f"✓ 가중치를 로드했습니다: {ckpt_path}")
        else:
            print(f"⚠️ 가중치 파일을 찾을 수 없습니다. 랜덤 초기화된 모델을 사용합니다.")
    except Exception as e:
        print(f"⚠️ 가중치 로딩 실패: {str(e)}")
        print("랜덤 초기화된 모델을 사용합니다.")
    
    model.eval()
    return model


def main():
    # 0) 디바이스
    device = 'cpu'
    print(f"Using device: {device}")
    print(f"고속 최적화 설정:")
    print(f"  - 분석 간격: {ANALYSIS_INTERVAL}초")
    print(f"  - 재생속도: {PLAYBACK_SPEED}x")
    print(f"  - 빠른 얼굴검출: {FAST_FACE_DETECTION}")
    print(f"  - 가벼운 모델: {USE_LIGHTER_MODEL}")

    # 1) 모델 (가벼운 모델 옵션)
    if USE_LIGHTER_MODEL:
        model_name = 'cnn'
        model_path = '/Users/ijaein/Desktop/Emotion/export/model.pth'
        image_size = 48
        print("CNN 모델 사용 (빠른 처리)")
    else:
        model_name = MODEL_NAME
        model_path = MODEL_PATH
        image_size = IMAGE_SIZE
        print(f"{MODEL_NAME} 모델 사용")
    
    model = build_model(model_name, model_path)

    # 2) 전처리 (모델에 따라 다르게)
    if USE_LIGHTER_MODEL:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])

    # 3) 얼굴 검출기 (빠른 모드)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if FAST_FACE_DETECTION:
        # 빠른 검출을 위한 파라미터 조정 (중복 검출 방지)
        scale_factor = 1.2  # 더 정밀한 스케일 (이전 1.5에서 1.2로)
        min_neighbors = 4   # 더 엄격한 이웃 수 (이전 3에서 4로)
    else:
        scale_factor = 1.1  # 매우 정밀한 스케일
        min_neighbors = 5   # 기본값

    # 4) 비디오 열기
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return

    # 속도 최적화 변수들
    frame_count = 0
    processed_frames = 0  # 실제 처리된 프레임 수
    
    # FPS 측정을 위한 변수들
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    # 재생 속도 조절을 위한 지연 시간 계산
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / (fps * PLAYBACK_SPEED)))
    
    # 1초에 해당하는 프레임 수 계산 (진짜 시간 기반)
    frames_per_interval = int(fps * ANALYSIS_INTERVAL)
    
    print(f"원본 비디오 FPS: {fps}")
    print(f"분석 간격: {ANALYSIS_INTERVAL}초 = {frames_per_interval} 프레임마다")
    print(f"이론적 처리 FPS: {fps / frames_per_interval:.1f}")
    print(f"재생 속도 적용: {fps / frames_per_interval * PLAYBACK_SPEED:.1f} (체감 FPS)")
    print("-" * 50)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # 1초 간격 프레임 스킵 적용 (진짜 시간 기반)
            if frame_count % frames_per_interval != 0:
                continue

            processed_frames += 1
            fps_frame_count += 1
            
            # FPS 계산 (1초마다 업데이트)
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (current_time - fps_start_time)
                fps_start_time = current_time
                fps_frame_count = 0

            # (웹캠용이라면 좌우 반전, 동영상이라면 주석 처리)
            frame = cv2.flip(frame, 1)

            # 얼굴 검출 (매번 수행 - 단순화)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            
            # 중복 얼굴 제거 및 가장 큰 얼굴만 선택
            if len(detected_faces) > 0:
                # 얼굴 크기(면적) 기준으로 정렬하여 가장 큰 얼굴만 선택
                faces_with_area = [(x, y, w, h, w*h) for x, y, w, h in detected_faces]
                faces_with_area.sort(key=lambda x: x[4], reverse=True)  # 면적 기준 내림차순
                
                # 가장 큰 얼굴만 선택 (중복 제거)
                largest_face = faces_with_area[0]
                faces = [(largest_face[0], largest_face[1], largest_face[2], largest_face[3])]
            else:
                faces = []

            # 얼굴 하나씩 처리
            for i, (x, y, w, h) in enumerate(faces):
                # ROI 자르고 PIL→Tensor
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(cv2.resize(face, (image_size, image_size)))
                inp = transform(pil).unsqueeze(0)

                # 예측
                logits = model(inp)
                probs  = torch.softmax(logits, dim=1)[0]
                p, idx = probs.max(0)
                label = f"{class_labels[idx]} ({p*100:.1f}%)"

                # 콘솔에도 출력
                print(f"[Frame {frame_count}] face#{i}: {label}")

                # 화면에 출력
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if len(faces) == 0:
                cv2.putText(frame, 'No Face Found', (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # 성능 정보 표시
            cv2.putText(frame, f'Analysis Interval: {ANALYSIS_INTERVAL}s', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f'Processing FPS: {current_fps:.1f}', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f'Processed: {processed_frames}/{frame_count}', (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # 로컬 환경에서 화면 띄우기
            cv2.imshow('Emotion Analysis (Optimized)', frame)

            # 재생 속도 조절된 대기 시간
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    # 최종 통계 출력
    total_time = time.time() - start_time
    average_fps = processed_frames / total_time if total_time > 0 else 0
    
    print("-" * 50)
    print(f"총 실행 시간: {total_time:.1f}초")
    print(f"총 프레임 수: {frame_count}")
    print(f"처리된 프레임 수: {processed_frames}")
    print(f"평균 처리 FPS: {average_fps:.1f}")
    print(f"처리 효율: {processed_frames/frame_count*100:.1f}% ({ANALYSIS_INTERVAL}초마다 1번)")
    print(f"이론적 최대 FPS: {fps / frames_per_interval:.1f}")
    print("-" * 50)


if __name__ == "__main__":
    main()