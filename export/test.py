#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from models import getModel  
import os
import time  # FPS 측정을 위해 추가
import glob
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
import numpy as np

# —————————— 사용자 설정 ——————————
# 다중 동영상 처리 설정
# VIDEO_FOLDER = '/Users/ijaein/Desktop/Emotion/export/video/'  # 비디오 폴더 경로
VIDEO_PATH   = '/Users/ijaein/Desktop/Emotion/export/video/이승무원.mp4' # 단일 비디오 경로 (예시)
MODEL_PATH   = '/Users/ijaein/Desktop/Emotion/model_eff.pth'  # EfficientNet 모델 가중치 파일
CASCADE_PATH = '/Users/ijaein/Desktop/Emotion/export/face_classifier.xml'
MODEL_NAME   = 'efficientnet-b5'  # EfficientNet-b5 사용
IMAGE_SIZE   = 224  

# 처리 방식 설정
PARALLEL_PROCESSING = False  # 단일 비디오는 순차 처리
MAX_WORKERS = 4              # 병렬 처리시 최대 워커 수 (CPU 코어수에 맞게 조정)
SHOW_VIDEO = True            # 동영상 화면 표시 여부 (단일 비디오는 True 권장)

# 속도 최적화 설정 (시간 기반 버전)
ANALYSIS_INTERVAL = 1.0  # 1초마다 1번 분석
PLAYBACK_SPEED = 5     # 비디오 재생 속도 (2.5배속)

# 추가 최적화 옵션들
FAST_FACE_DETECTION = True  # 빠른 얼굴 검출 모드
USE_LIGHTER_MODEL = False   # 더 가벼운 CNN 모델 사용 (True로 설정하면 CNN 사용)

# 한글 라벨 
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

# 감정 매핑 (평가용)
EMOTION_MAPPING = {
    '기쁨': 'happy',
    '당황': 'surprise', 
    '분노': 'angry',
    '불안': 'fear',
    '상처': 'disgust',
    '슬픔': 'sad',
    '중립': 'neutral'
}

# 면접 평가 기준
POSITIVE_EMOTIONS = ['happy', 'neutral']
NEGATIVE_EMOTIONS = ['sad', 'angry', 'fear', 'surprise', 'disgust']

# 폰트 설정 (AppleGothic.ttf 경로 및 크기)
FONT_PATH = '/System/Library/Fonts/AppleGothic.ttf' # AppleGothic 폰트 경로 (macOS 기본 경로)
FONT_SIZE = 30 # 폰트 크기

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


def calculate_interview_score(emotion_data):
    """면접 평가 점수 계산 함수"""
    if not emotion_data:
        return 0, {}
    
    # 감정별 통계 계산
    total_frames = len(emotion_data)
    emotion_counts = defaultdict(int)
    emotion_confidences = defaultdict(list)
    emotion_transitions = 0
    prev_emotion = None
    
    for frame_data in emotion_data:
        emotion = frame_data['emotion']
        confidence = frame_data['confidence']
        
        emotion_counts[emotion] += 1
        emotion_confidences[emotion].append(confidence)
        
        # 감정 전이 계산
        if prev_emotion is not None and prev_emotion != emotion:
            emotion_transitions += 1
        prev_emotion = emotion
    
    # 1. 긍정 감정 비율 (최대 25점)
    positive_count = sum(emotion_counts[emotion] for emotion in POSITIVE_EMOTIONS)
    positive_ratio = positive_count / total_frames
    positive_score = positive_ratio * 100 * 0.25
    
    # 2. 부정 감정 비율 (최대 15점)
    negative_count = sum(emotion_counts[emotion] for emotion in NEGATIVE_EMOTIONS)
    negative_ratio = negative_count / total_frames
    negative_score = (1 - negative_ratio) * 100 * 0.15
    
    # 3. happy 평균 confidence (최대 20점)
    if 'happy' in emotion_confidences and emotion_confidences['happy']:
        happy_confidence = sum(emotion_confidences['happy']) / len(emotion_confidences['happy'])
        happy_score = happy_confidence * 100 * 0.2
    else:
        happy_confidence = 0
        happy_score = 0
    
    # 총점 계산 (60점 만점)
    total_score = min(60, positive_score + negative_score + happy_score)
    
    # 상세 분석 결과
    analysis = {
        'total_frames': total_frames,
        'emotion_counts': dict(emotion_counts),
        'emotion_ratios': {emotion: count/total_frames for emotion, count in emotion_counts.items()},
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'happy_confidence': happy_confidence,
        'emotion_transitions': emotion_transitions,
        'scores': {
            'positive_score': positive_score,
            'negative_score': negative_score, 
            'happy_score': happy_score,
            'total_score': total_score
        }
    }
    
    return total_score, analysis


def print_interview_report(video_name, analysis):
    """면접 평가 리포트 출력"""
    print(f"\n📊 === {video_name} 면접 평가 리포트 ===")
    print("=" * 60)
    
    scores = analysis['scores']
    print(f"🎯 **최종 점수: {scores['total_score']:.1f}/60점**")
    print(f"📈 **평가 등급: {get_grade(scores['total_score'])}**")
    
    print(f"\n📋 **세부 평가:**")
    print(f"  1. 긍정 감정 비율: {scores['positive_score']:.1f}/25점 ({analysis['positive_ratio']*100:.1f}%)")
    print(f"  2. 부정 감정 제어: {scores['negative_score']:.1f}/15점 ({analysis['negative_ratio']*100:.1f}%)")
    print(f"  3. 미소 신뢰도: {scores['happy_score']:.1f}/20점 ({analysis['happy_confidence']*100:.1f}%)")
    
    print(f"\n📈 **감정 분포:**")
    for emotion, count in analysis['emotion_counts'].items():
        ratio = analysis['emotion_ratios'][emotion]
        print(f"  - {emotion}: {count}회 ({ratio*100:.1f}%)")
    
    print(f"\n💡 **개선 제안:**")
    suggestions = get_improvement_suggestions(analysis)
    for suggestion in suggestions:
        print(f"  • {suggestion}")
    
    print("=" * 60)


def get_grade(score):
    """점수를 등급으로 변환 (60점 만점 기준)"""
    if score >= 54:  # 90% 이상
        return "A+ (우수)"
    elif score >= 48:  # 80% 이상
        return "A (양호)"
    elif score >= 42:  # 70% 이상
        return "B+ (보통)"
    elif score >= 36:  # 60% 이상
        return "B (미흡)"
    else:
        return "C (개선 필요)"


def get_improvement_suggestions(analysis):
    """개선 제안 생성"""
    suggestions = []
    scores = analysis['scores']
    
    if scores['positive_score'] < 15:
        suggestions.append("더 자주 미소를 짓고 긍정적인 표정을 유지하세요")
    
    if scores['negative_score'] < 12:
        suggestions.append("부정적인 감정 표현을 줄이고 중립적인 표정을 유지하세요")
    
    if scores['happy_score'] < 15:
        suggestions.append("미소의 진정성을 높이고 자연스러운 표정을 연습하세요")
    
    if not suggestions:
        suggestions.append("전반적으로 우수한 표정 관리를 보여주었습니다!")
    
    return suggestions


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
    
    # 감정 데이터 수집
    emotion_data = []
    
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

    # 폰트 로드
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        # 작은 폰트 (성능 정보용)
        font_small = ImageFont.truetype(FONT_PATH, FONT_SIZE - 10) 
    except IOError:
        print(f"⚠️ 폰트 파일을 찾을 수 없거나 로드할 수 없습니다: {FONT_PATH}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

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
            # frame = cv2.flip(frame, 1) # 이전에 주석 처리되어 있지 않았다면 이 줄도 주석 처리합니다.

            # Convert frame to PIL Image to draw text
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

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
                emotion_korean = class_labels[idx]
                emotion_english = EMOTION_MAPPING[emotion_korean]
                label = f"{emotion_korean} ({p*100:.1f}%)"

                # 감정 데이터 저장
                emotion_data.append({
                    'frame': frame_count,
                    'emotion': emotion_english,
                    'emotion_korean': emotion_korean,
                    'confidence': p.item()
                })

                # 콘솔에도 출력
                print(f"[Frame {frame_count}] face#{i}: {label}")

                # 화면에 출력 (PIL 사용)
                # 얼굴 바운딩 박스 그리기 (PIL 사용)
                draw.rectangle([(x, y), (x+w, y+h)], outline=(0, 255, 0), width=2) # 초록색 테두리
                
                # 텍스트 위치 계산 (얼굴 위) - textbbox 사용
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x
                text_y = y - text_height - 5 # 얼굴 위 5픽셀
                if text_y < 0: # 화면 상단 벗어나면 얼굴 아래로
                    text_y = y + h + 5

                draw.text((text_x, text_y), label, font=font, fill=(0, 255, 0, 255)) # 초록색 텍스트 (RGBA)

            if len(faces) == 0:
                draw.text((20, 60), 'No Face Found', font=font, fill=(255, 0, 0, 255)) # 빨간색 텍스트

            # 성능 정보 표시 (PIL 사용)
            draw.text((20, 30), f'Analysis Interval: {ANALYSIS_INTERVAL}s', font=font_small, fill=(0, 255, 255, 255)) # 청록색 텍스트
            draw.text((20, 60), f'Processing FPS: {current_fps:.1f}', font=font_small, fill=(0, 255, 255, 255))
            draw.text((20, 90), f'Processed: {processed_frames}/{frame_count}', font=font_small, fill=(0, 255, 255, 255))

            # PIL Image를 다시 OpenCV (NumPy 배열)로 변환
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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
    
    # 면접 평가 점수 계산
    interview_score = 0
    interview_analysis = {}
    if emotion_data:
        interview_score, interview_analysis = calculate_interview_score(emotion_data)
        print_interview_report(os.path.basename(video_path), interview_analysis)
    
    # 결과 반환
    return {
        'video_path': video_path,
        'total_time': total_time,
        'total_frames': frame_count,
        'processed_frames': processed_frames,
        'average_fps': average_fps,
        'emotion_data': emotion_data,
        'interview_score': interview_score,
        'interview_analysis': interview_analysis,
        'success': True
    }


def main(video_path=None):
    """메인 함수 - 단일 동영상 처리"""
    print("🎬 단일 동영상 감정 분석 시작")
    print("=" * 70)
    
    # 설정 출력
    device = 'cpu'
    print(f"Using device: {device}")
    print(f"최적화 설정:")
    print(f"  - 분석 간격: {ANALYSIS_INTERVAL}초")
    print(f"  - 재생속도: {PLAYBACK_SPEED}x")
    print(f"  - 화면 표시: {SHOW_VIDEO}")
    print(f"  - 빠른 얼굴검출: {FAST_FACE_DETECTION}")
    print(f"  - 가벼운 모델: {USE_LIGHTER_MODEL}")

    # 비디오 경로 설정
    if video_path is None:
        video_path = VIDEO_PATH # 기본 경로 사용

    if not os.path.exists(video_path):
        print(f"❌ 동영상 파일을 찾을 수 없습니다: {video_path}")
        return
    
    print(f"\n📁 처리할 동영상 파일: {os.path.basename(video_path)}")
    file_size = os.path.getsize(video_path) / (1024*1024)  # MB
    print(f"  크기: {file_size:.1f}MB")

    # 처리 시작
    total_start_time = time.time()
    result = process_single_video(video_path)
    
    # 전체 결과 요약
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    print("\n" + "=" * 70)
    print("🎯 처리 결과 요약")
    print("=" * 70)
    print(f"총 처리 시간: {total_processing_time:.1f}초")
    
    if result and result['success']:
        video_name = os.path.basename(result['video_path'])
        print(f"  {video_name[:40]+'...' if len(video_name) > 40 else video_name}")
        print(f"     처리시간: {result['total_time']:.1f}초, "
              f"프레임: {result['processed_frames']}/{result['total_frames']}, "
              f"FPS: {result['average_fps']:.1f}")
        
        # 면접 점수 표시
        if 'interview_score' in result and result['interview_score'] > 0:
            score = result['interview_score']
            grade = get_grade(score)
            print(f"     🎯 면접점수: {score:.1f}/60점 ({grade})")
    else:
        print("❌ 동영상 처리 실패")
    
    print("🏁 처리 완료!")


if __name__ == "__main__":
    # 여기에서 처리할 단일 동영상 파일 경로를 지정하거나, 기본값(VIDEO_PATH)을 사용합니다.
    # 예시: main('/Users/ijaein/Desktop/Emotion/export/video/my_interview.mp4')
    main()