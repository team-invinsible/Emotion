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
import glob
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading

# —————————— 사용자 설정 ——————————
# 다중 동영상 처리 설정
VIDEO_FOLDER = '/Users/ijaein/Desktop/Emotion/export/video/'  # 비디오 폴더 경로
MODEL_PATH   = '/Users/ijaein/Desktop/Emotion/model_eff.pth'  # EfficientNet 모델 가중치 파일
CASCADE_PATH = '/Users/ijaein/Desktop/Emotion/export/face_classifier.xml'
MODEL_NAME   = 'efficientnet-b5'  # EfficientNet-b5 사용
IMAGE_SIZE   = 224  

# 처리 방식 설정
PARALLEL_PROCESSING = True  # True: 병렬처리, False: 순차처리
MAX_WORKERS = 4              # 병렬 처리시 최대 워커 수 (CPU 코어수에 맞게 조정)
SHOW_VIDEO = False           # 동영상 화면 표시 여부 (병렬처리시 False 권장)

# 속도 최적화 설정 (시간 기반 버전)
ANALYSIS_INTERVAL = 1.0  # 1초마다 1번 분석
PLAYBACK_SPEED = 5     # 비디오 재생 속도 (2.5배속)

# 추가 최적화 옵션들
FAST_FACE_DETECTION = True  # 빠른 얼굴 검출 모드
USE_LIGHTER_MODEL = False   # 더 가벼운 CNN 모델 사용 (True로 설정하면 CNN 사용)

# 한글 라벨 
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
# ————————————————————————————————


def get_video_files(folder_path):
    """비디오 파일 목록을 가져오는 함수"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.m4v', '*.webm']
    
    if not os.path.isdir(folder_path):
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return []
    
    video_files = set()  # 중복 제거를 위해 set 사용
    for ext in video_extensions:
        # 현재 폴더만 검색 (하위 폴더 제외로 중복 방지)
        video_files.update(glob.glob(os.path.join(folder_path, ext)))
    
    return sorted(list(video_files))


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


def process_single_video(video_path):
    """단일 비디오 처리 함수 (병렬 처리용)"""
    try:
        print(f"🎬 처리 시작: {os.path.basename(video_path)}")
        
        # 모델 로드 (각 프로세스마다)
        if USE_LIGHTER_MODEL:
            model_name = 'cnn'
            model_path = '/Users/ijaein/Desktop/Emotion/export/model.pth'
            image_size = 48
        else:
            model_name = MODEL_NAME
            model_path = MODEL_PATH
            image_size = IMAGE_SIZE
        
        model = build_model(model_name, model_path)
        
        # 전처리
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
        
        # 얼굴 검출기
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if FAST_FACE_DETECTION:
            scale_factor = 1.2
            min_neighbors = 4
        else:
            scale_factor = 1.1
            min_neighbors = 5
        
        # 비디오 처리
        return process_video_core(video_path, model, transform, face_cascade, 
                                scale_factor, min_neighbors, image_size)
    
    except Exception as e:
        print(f"❌ {os.path.basename(video_path)} 처리 중 오류: {str(e)}")
        return None


def process_video_core(video_path, model, transform, face_cascade, scale_factor, min_neighbors, image_size):
    """비디오 처리 핵심 로직"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

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

            # 화면 표시 (선택적)
            if SHOW_VIDEO:
                cv2.imshow(f'Emotion Analysis - {os.path.basename(video_path)}', frame)
                # 재생 속도 조절된 대기 시간
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    
    # 최종 통계
    total_time = time.time() - start_time
    average_fps = processed_frames / total_time if total_time > 0 else 0
    
    print(f"✅ {os.path.basename(video_path)} 완료!")
    print(f"   처리시간: {total_time:.1f}초, 프레임: {processed_frames}/{frame_count}, FPS: {average_fps:.1f}")
    
    # 결과 반환
    return {
        'video_path': video_path,
        'total_time': total_time,
        'total_frames': frame_count,
        'processed_frames': processed_frames,
        'average_fps': average_fps,
        'success': True
    }


def main():
    """메인 함수 - 다중 동영상 처리"""
    print("🎬 다중 동영상 감정 분석 시작")
    print("=" * 70)
    
    # 설정 출력
    device = 'cpu'
    print(f"Using device: {device}")
    print(f"처리 방식: {'병렬 처리' if PARALLEL_PROCESSING else '순차 처리'}")
    if PARALLEL_PROCESSING:
        print(f"최대 워커 수: {MAX_WORKERS}")
    print(f"최적화 설정:")
    print(f"  - 분석 간격: {ANALYSIS_INTERVAL}초")
    print(f"  - 재생속도: {PLAYBACK_SPEED}x")
    print(f"  - 화면 표시: {SHOW_VIDEO}")
    print(f"  - 빠른 얼굴검출: {FAST_FACE_DETECTION}")
    print(f"  - 가벼운 모델: {USE_LIGHTER_MODEL}")

    # 비디오 파일 목록 가져오기
    video_files = get_video_files(VIDEO_FOLDER)
    if not video_files:
        print("❌ 처리할 동영상 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n📁 찾은 동영상 파일 ({len(video_files)}개):")
    for i, video_path in enumerate(video_files, 1):
        file_size = os.path.getsize(video_path) / (1024*1024)  # MB
        print(f"  {i}. {os.path.basename(video_path)} ({file_size:.1f}MB)")

    # 처리 시작
    total_start_time = time.time()
    results = []
    
    if PARALLEL_PROCESSING:
        # 병렬 처리
        print(f"\n🚀 병렬 처리 시작 (워커 수: {MAX_WORKERS})")
        if SHOW_VIDEO:
            print("⚠️ 병렬 처리시 화면 표시는 비활성화됩니다.")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_video, video_path) 
                      for video_path in video_files]
            
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
    
    else:
        # 순차 처리
        print(f"\n📹 순차 처리 시작")
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] 처리 중...")
            result = process_single_video(video_path)
            if result:
                results.append(result)
    
    # 전체 결과 요약
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    print("\n" + "=" * 70)
    print("🎯 전체 처리 결과 요약")
    print("=" * 70)
    print(f"총 처리 시간: {total_processing_time:.1f}초")
    print(f"처리된 동영상: {len(results)}/{len(video_files)}개")
    
    if results:
        total_frames = sum(r['total_frames'] for r in results)
        total_processed = sum(r['processed_frames'] for r in results)
        avg_fps = sum(r['average_fps'] for r in results) / len(results)
        
        print(f"총 프레임 수: {total_frames:,}")
        print(f"처리된 프레임: {total_processed:,}")
        print(f"전체 처리 효율: {total_processed/total_frames*100:.1f}%")
        print(f"평균 처리 FPS: {avg_fps:.1f}")
        
        if PARALLEL_PROCESSING:
            theoretical_sequential_time = sum(r['total_time'] for r in results)
            speedup = theoretical_sequential_time / total_processing_time
            print(f"병렬 처리 가속도: {speedup:.2f}x")
        
        print(f"\n📊 개별 동영상 결과:")
        for i, result in enumerate(results, 1):
            video_name = os.path.basename(result['video_path'])
            print(f"  {i}. {video_name[:40]+'...' if len(video_name) > 40 else video_name}")
            print(f"     처리시간: {result['total_time']:.1f}초, "
                  f"프레임: {result['processed_frames']}/{result['total_frames']}, "
                  f"FPS: {result['average_fps']:.1f}")
    
    print("🏁 모든 처리 완료!")


if __name__ == "__main__":
    main()