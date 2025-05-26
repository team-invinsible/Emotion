#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from collections import deque
import time
import argparse
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def get_font_path():
    """시스템에 설치된 한글 폰트 경로 반환"""
    # macOS 기본 한글 폰트
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "/System/Library/Fonts/PingFang.ttc",          # macOS
        "/Library/Fonts/NanumGothic.ttf",              # 나눔고딕
        "/Library/Fonts/MalgunGothic.ttf",             # 맑은 고딕
        "/Library/Fonts/NotoSansCJKkr-Regular.otf",    # Noto Sans
        "/Users/ijaein/Desktop/my_emotion_api/export/NanumGothic.ttf"  # 사용자 지정 경로
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return path
    
    return None

class EmotionAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', font_path=None):
        self.device = device
        self.transform = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        self.emotion_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recent_emotions = deque(maxlen=30)

        # Placeholder for FACS-related components if needed
        # self.facs_model = None

        # 한글 폰트 설정
        self.font = None
        self.font_size = 25
        self.font_path = font_path if font_path else get_font_path()
        
        try:
            if self.font_path and os.path.exists(self.font_path):
                self.font = ImageFont.truetype(self.font_path, self.font_size)
                print(f"✓ 한글 폰트를 로드했습니다: {os.path.basename(self.font_path)}")
            else:
                print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
                self.font = ImageFont.loadDefault()
        except Exception as e:
            print(f"⚠️ 폰트 로드 중 오류 발생: {str(e)}")
            print("기본 폰트를 사용합니다.")
            self.font = ImageFont.loadDefault()

        # 감정별 가중치 설정 (100점 만점 기준)
        self.emotion_weights = {
            '기쁨': 100,    # 가장 높은 점수
            '중립': 95,     # 긍정적 감정 (상향 조정)
            '당황': 60,     # 중립적 감정 (상향 조정)
            '불안': 50,     # 부정적 감정 (상향 조정)
            '상처': 60,     # 부정적 감정 (상향 조정)
            '슬픔': 60,     # 부정적 감정 (상향 조정)
            '분노': 50      # 가장 낮은 점수 (상향 조정)
        }
        
        # 감정 확률 조정을 위한 가중치
        self.emotion_prob_weights = {
            '기쁨': 1.3,    # 기쁨 확률 30% 증가
            '중립': 1.5,    # 중립 확률 50% 증가
            '당황': 0.6,    # 30% 감소
            '불안': 0.6,    # 30% 감소
            '상처': 0.6,    # 30% 감소
            '슬픔': 0.6,    # 30% 감소
            '분노': 0.6     # 30% 감소
        }
        
        # 결과 저장을 위한 변수들
        self.results = {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'emotions': [],
            'scores': [],
            'frames_processed': 0,
            'total_faces_detected': 0,
            'emotion_scores': {emotion: [] for emotion in self.emotion_labels}  # 감정별 점수 저장
        }

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

    def filter_faces(self, faces, frame_shape):
        filtered = []
        h, w = frame_shape[:2]
        for (x, y, fw, fh) in faces:
            if fw > w * 0.1 and fh > h * 0.1:
                filtered.append((x, y, fw, fh))
        return filtered

    def analyze_emotion(self, face_img):
        # 기본 확률 계산 (시뮬레이션)
        probs = F.softmax(torch.randn(len(self.emotion_labels)), dim=0)
        probs = probs.cpu().numpy()
        
        # 감정별 가중치 적용
        weighted_probs = probs * np.array([self.emotion_prob_weights[emotion] for emotion in self.emotion_labels])
        
        # 확률 정규화 (합이 1이 되도록)
        weighted_probs = weighted_probs / np.sum(weighted_probs)
        
        # 감정별 점수 계산
        scores = {}
        for i, emotion in enumerate(self.emotion_labels):
            scores[emotion] = weighted_probs[i] * self.emotion_weights[emotion]
        
        return weighted_probs, scores

    def calculate_score(self, probs, scores):
        """100점 만점 기준으로 점수 계산 (완화된 기준)"""
        total_score = 0
        total_weight = 0
        
        # 기본 점수 계산
        for emotion, prob in zip(self.emotion_labels, probs):
            if prob > 0:
                total_score += scores[emotion]
                total_weight += prob
        
        if total_weight > 0:
            base_score = total_score / total_weight
            
            # 점수 보정 (최소 점수 보장)
            min_score = 60  # 최소 점수 설정
            if base_score < min_score:
                # 낮은 점수에 대해 완화된 보정 적용
                corrected_score = min_score + (base_score - min_score) * 0.5
                return min(100, max(min_score, corrected_score))
            
            # 높은 점수에 대해 추가 보너스
            if base_score > 80:
                bonus = (base_score - 80) * 0.2  # 80점 이상에 대해 20% 보너스
                return min(100, base_score + bonus)
            
            return min(100, base_score)
        return 60  # 기본 최소 점수

    def grade(self, score):
        """100점 만점 기준 등급 계산 (완화된 기준)"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        else:
            return 'C'

    def draw_korean_text(self, img, text, position, color=(255, 255, 255), font_size=None):
        """한글 텍스트를 이미지에 그리는 함수"""
        try:
            # PIL Image로 변환
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 폰트 크기 조정이 필요한 경우
            if font_size and font_size != self.font_size:
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                except:
                    font = self.font
            else:
                font = self.font
            
            # 텍스트 배경을 위한 크기 계산
            text_bbox = draw.textbbox(position, text, font=font)
            padding = 5
            bg_rect = [
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding
            ]
            
            # 텍스트 배경 그리기 (반투명)
            draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
            
            # 한글 텍스트 그리기
            draw.text(position, text, font=font, fill=color)
            
            # OpenCV 이미지로 변환
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ 텍스트 그리기 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 폰트로 대체
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img

    def save_results(self, output_dir='results'):
        """분석 결과를 JSON 파일로 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'emotion_analysis_{timestamp}.json')
            
            # NumPy 타입을 Python 기본 타입으로 변환
            def convert_to_serializable(obj):
                if isinstance(obj, np.float32):
                    return float(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj

            # 최종 통계 계산
            final_stats = {
                '분석 시작 시간': self.results['start_time'],
                '분석 종료 시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '총 프레임 수': int(self.results['frames_processed']),
                '감지된 얼굴 수': int(self.results['total_faces_detected']),
                '평균 감정 점수': float(np.mean(self.results['scores'])) if self.results['scores'] else 0.0,
                '최종 등급': self.grade(float(np.mean(self.results['scores']))) if self.results['scores'] else 'N/A',
                '감정 분포': convert_to_serializable(self.get_emotion_distribution())
            }
            
            # 모든 값을 직렬화 가능한 형태로 변환
            final_stats = convert_to_serializable(final_stats)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
            
            print(f"\n분석 결과가 저장되었습니다: {filename}")
            self.print_summary(final_stats)
            
        except Exception as e:
            print(f"결과 저장 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def get_emotion_distribution(self):
        """감정 분포 계산 (점수 포함)"""
        if not self.results['emotions']:
            return {}
        
        emotion_stats = {}
        for emotion in self.emotion_labels:
            count = self.results['emotions'].count(emotion)
            scores = [float(score) for score in self.results['emotion_scores'][emotion]]
            avg_score = float(np.mean(scores)) if scores else 0.0
            
            emotion_stats[emotion] = {
                '횟수': int(count),
                '비율': f"{(count / len(self.results['emotions']) * 100):.1f}%",
                '평균 점수': f"{avg_score:.1f}",
                '최고 점수': f"{float(max(scores)):.1f}" if scores else "0.0",
                '최저 점수': f"{float(min(scores)):.1f}" if scores else "0.0"
            }
        return emotion_stats
    
    def print_summary(self, stats):
        """분석 결과 요약 출력 (100점 만점 기준)"""
        print("\n=== 감정 분석 결과 요약 (100점 만점) ===")
        print(f"분석 기간: {stats['분석 시작 시간']} ~ {stats['분석 종료 시간']}")
        print(f"총 프레임 수: {stats['총 프레임 수']}")
        print(f"감지된 얼굴 수: {stats['감지된 얼굴 수']}")
        print(f"평균 감정 점수: {stats['평균 감정 점수']:.1f}/100")
        print(f"최종 등급: {stats['최종 등급']}")
        print("\n감정별 상세 통계:")
        print("(기쁨: +30%, 중립: +50%, 부정적 감정: -30% 가중치 적용)")
        print("(최소 점수 60점 보장, 80점 이상 20% 보너스)")
        for emotion, data in stats['감정 분포'].items():
            print(f"\n  {emotion}:")
            print(f"    횟수: {data['횟수']}회 ({data['비율']})")
            print(f"    평균 점수: {data['평균 점수']}/100")
            print(f"    최고 점수: {data['최고 점수']}/100")
            print(f"    최저 점수: {data['최저 점수']}/100")
        print("\n등급 기준 (완화된 기준):")
        print("  A+ (90-100점), A (85-89점), A- (80-84점)")
        print("  B+ (75-79점), B (70-74점), B- (65-69점)")
        print("  C+ (60-64점), C (55-59점), C- (50-54점)")
        print("  D+ (45-49점), D (40-44점), D- (0-39점)")
        print("========================\n")

    def update_recent_emotions(self, emotion_idx):
        self.recent_emotions.append(emotion_idx)
        

    def average_score(self):
        if not self.recent_emotions:
            return 0.0
        return np.mean(self.recent_emotions)

    
def main():
    parser = argparse.ArgumentParser(description='Emotion Analysis Pipeline')
    parser.add_argument('--video', type=str, default=1, help='Video source (default: 1 for webcam)')
    parser.add_argument('--font_path', help='한글 폰트 파일 경로')
    parser.add_argument('--output_dir', default='results', help='결과 저장 디렉토리')
    args = parser.parse_args()

    # 폰트 경로 확인
    font_path = args.font_path if args.font_path else get_font_path()
    if font_path:
        print(f"✓ 한글 폰트 사용: {os.path.basename(font_path)}")
    else:
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

    analyzer = EmotionAnalyzer(font_path=font_path)
    cap = cv2.VideoCapture(int(args.video) if str(args.video).isdigit() else args.video)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # 프레임 처리 간격 설정 (초당 4프레임 = 0.25초 간격)
    frame_interval = 0.25  # 초
    last_process_time = time.time()

    print("감정 분석을 시작합니다. 종료하려면 'q'를 누르세요.")
    print("분석 결과는 자동으로 저장됩니다.")
    print(f"프레임 처리 속도: 초당 {1/frame_interval:.1f}프레임")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            # 프레임 처리 간격 확인
            if current_time - last_process_time < frame_interval:
                # 다음 프레임으로 넘어가기
                cv2.imshow('Emotion Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            last_process_time = current_time
            analyzer.results['frames_processed'] += 1
            faces = analyzer.preprocess(frame)
            faces = analyzer.filter_faces(faces, frame.shape)
            analyzer.results['total_faces_detected'] += len(faces)

            results = []
            if len(faces) == 0:
                h, w = frame.shape[:2]
                probs, scores = analyzer.analyze_emotion(frame)
                emotion_idx = np.argmax(probs)
                label = analyzer.emotion_labels[emotion_idx]
                current_score = analyzer.calculate_score(probs, scores)
                results.append({
                    'rect': (0, 0, w, h),
                    'label': label,
                    'probs': probs,
                    'scores': scores,
                    'score': current_score
                })
            else:
                for (x, y, w, h) in faces:
                    roi = frame[y:y+h, x:x+w]
                    probs, scores = analyzer.analyze_emotion(roi)
                    emotion_idx = np.argmax(probs)
                    label = analyzer.emotion_labels[emotion_idx]
                    current_score = analyzer.calculate_score(probs, scores)
                    results.append({
                        'rect': (x, y, w, h),
                        'label': label,
                        'probs': probs,
                        'scores': scores,
                        'score': current_score
                    })

            for res in results:
                x, y, w, h = res['rect']
                label = res['label']
                probs = res['probs']
                scores = res['scores']
                current_score = res['score']
                
                # 결과 저장
                analyzer.results['emotions'].append(label)
                analyzer.results['emotion_scores'][label].append(current_score)
                analyzer.update_recent_emotions(np.argmax(probs))
                
                # 텍스트 크기 조정
                prob_text = f"{label}: {current_score:.1f}점"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                frame = analyzer.draw_korean_text(frame, prob_text, (x, y - 30), color=(255, 255, 255))

            avg_score = np.mean([r['score'] for r in results]) if results else 0
            grade = analyzer.grade(avg_score)
            
            # 점수 저장
            analyzer.results['scores'].append(avg_score)

            # 텍스트 크기 조정
            info_text = f"평균 점수: {avg_score:.1f}/100 등급: {grade}"
            frame = analyzer.draw_korean_text(frame, info_text, (10, 40), color=(0, 255, 255))

            cv2.imshow('Emotion Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # 분석 결과 저장
        analyzer.save_results(args.output_dir)


if __name__ == '__main__':
    main()
