#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from models import *
from emotion import predict, face_classifier, class_labels
import time
from collections import deque
import argparse
import os

class EmotionAnalyzer:
    def __init__(self, model_path, model_name='emotionnet', image_size=48, gpu=False, font_path=None):
        self.image_size = (image_size, image_size)
        self.device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        
        # 모델 로드 및 최적화
        model_state = torch.load(model_path, map_location=torch.device(self.device))
        self.model = getModel(model_name, silent=True)
        self.model.load_state_dict(model_state['model'])
        self.model.eval()
        
        # CPU 사용 시 추론 최적화
        if self.device == 'cpu':
            torch.set_num_threads(4)
            self.model = torch.jit.script(self.model)
        
        # 감정 점수 계산을 위한 버퍼
        self.emotion_buffer = deque(maxlen=10)
        self.total_scores = []
        self.total_frames = 0
        self.last_update_time = time.time()
        self.update_interval = 0.5
        
        # 한글 폰트 설정
        self.font = None
        if font_path is None:
            # 기본 폰트 경로 시도
            font_path = os.path.join(os.path.dirname(__file__), 'NanumGothic.ttf')
        
        try:
            from PIL import ImageFont, ImageDraw, Image
            if os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, 20)
                print(f"한글 폰트를 로드했습니다: {font_path}")
            else:
                print(f"폰트 파일을 찾을 수 없습니다: {font_path}")
                print("기본 폰트를 사용합니다.")
        except Exception as e:
            print(f"폰트 로드 중 오류 발생: {str(e)}")
            print("기본 폰트를 사용합니다.")
        
        # 변환 파이프라인 최적화
        self.transform = tt.Compose([
            tt.ToPILImage(),
            tt.Grayscale(),
            tt.ToTensor()
        ])
    
    def put_korean_text(self, img, text, position, color=(0, 255, 0), size=20):
        try:
            if self.font is None:
                # 한글 폰트가 없는 경우 기본 폰트 사용
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                return img
            
            # PIL Image로 변환
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 한글 텍스트 그리기
            draw.text(position, text, font=self.font, fill=color)
            
            # OpenCV 이미지로 변환
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"텍스트 그리기 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 이미지 반환
            return img
    
    def calculate_score(self, probs):
        weights = {
            '기쁨': 1.2,
            '중립': 1.0,
            '당황': 0.8,
            '불안': 0.6,
            '상처': 0.4,
            '슬픔': 0.3,
            '분노': 0.2
        }
        
        weighted_score = 0
        total_weight = 0
        for emotion in class_labels:
            if emotion in probs:
                weighted_score += probs[emotion] * weights[emotion]
                total_weight += weights[emotion]
        
        if total_weight > 0:
            return min(100, weighted_score / total_weight * 100)
        return 0
    
    def get_grade(self, score):
        # 등급 계산
        grades = [(95, 'A+'), (90, 'A'), (85, 'A-'), (80, 'B+'), 
                 (75, 'B'), (70, 'B-'), (65, 'C+'), (60, 'C'),
                 (55, 'C-'), (50, 'D+'), (45, 'D')]
        
        for threshold, grade in grades:
            if score >= threshold:
                return grade
        return 'D-'
    
    def get_average_score(self):
        if not self.total_scores:
            return 0, 'N/A'
        avg_score = sum(self.total_scores) / len(self.total_scores)
        return avg_score, self.get_grade(avg_score)
    
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.2, 4, minSize=(30, 30))
        
        if len(faces) == 0:
            return frame, self.get_average_score()[0], self.get_average_score()[1], None
        
        # Filter: ignore faces smaller than 10% or larger than 40% of frame
        h_img, w_img = gray.shape
        valid_faces = [f for f in faces if 0.1 <= (f[2] * f[3]) / (w_img * h_img) <= 0.4]
        if not valid_faces:
            return frame, self.get_average_score()[0], self.get_average_score()[1], None
        face = max(valid_faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, self.image_size, interpolation=cv2.INTER_AREA)
        
        with torch.no_grad():
            tensor = self.transform(roi).unsqueeze(0)
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            probs = {class_labels[i]: float(prob) * 100 
                    for i, prob in enumerate(F.softmax(output, dim=1)[0])}
            current_emotion = max(probs.items(), key=lambda x: x[1])
        
        # 현재 프레임 점수 계산 및 저장
        current_score = self.calculate_score(probs)
        self.total_scores.append(current_score)
        self.total_frames += 1
        
        # 버퍼에 추가
        self.emotion_buffer.append(probs)
        
        # 평균 점수 계산
        avg_score, avg_grade = self.get_average_score()
        
        return frame, avg_score, avg_grade, (current_emotion[0], current_emotion[1], probs)

    def get_emotion_diversity_score(self):
        if not self.emotion_buffer:
            return 0.0
        top_emotions = [max(e.items(), key=lambda x: x[1])[0] for e in self.emotion_buffer]
        diversity = len(set(top_emotions)) / len(class_labels)
        return round(diversity * 100, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model.pth', help='모델 경로')
    parser.add_argument('--model', default='emotionnet', help='네트워크 아키텍처')
    parser.add_argument('--image_size', type=int, default=48, help='입력 이미지 크기')
    parser.add_argument('--gpu', action='store_true', help='GPU 사용')
    parser.add_argument('--camera', type=int, default=1, help='카메라 인덱스')
    parser.add_argument('--font_path', help='한글 폰트 파일 경로')
    args = parser.parse_args()
    
    analyzer = EmotionAnalyzer(args.model_path, args.model, args.image_size, args.gpu, args.font_path)
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"카메라 {args.camera}를 열 수 없습니다.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print(f"카메라 {args.camera}로 웹캠 감정 분석을 시작합니다. 종료하려면 'q'를 누르세요.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        if frame is None or frame.size == 0:
            print("빈 프레임이 감지되었습니다.")
            continue
            
        try:
            frame, score, grade, (current_emotion, current_prob, all_probs) = analyzer.process_frame(frame)
            
            if current_emotion is not None:
                # 배경 박스 추가
                cv2.rectangle(frame, (5, 5), (300, 220), (0, 0, 0), -1)
                
                # 현재 감정 표시
                analyzer.put_korean_text(frame, f"현재 감정: {current_emotion} ({current_prob:.1f}%)", 
                                       (10, 30), (0, 255, 0))
                
                # 모든 감정 확률 표시
                y_pos = 60
                for emotion, prob in all_probs.items():
                    color = (0, 255, 0) if emotion == current_emotion else (200, 200, 200)
                    analyzer.put_korean_text(frame, f"{emotion}: {prob:.1f}%", 
                                           (10, y_pos), color)
                    y_pos += 25
                
                # 구분선
                cv2.line(frame, (5, y_pos), (295, y_pos), (100, 100, 100), 1)
                y_pos += 20
                
                # 전체 평균 점수와 등급 표시
                analyzer.put_korean_text(frame, f"전체 평균 점수: {score:.1f}", 
                                       (10, y_pos + 10), (0, 255, 0))
                analyzer.put_korean_text(frame, f"전체 평균 등급: {grade}", 
                                       (10, y_pos + 40), (0, 255, 0))
                analyzer.put_korean_text(frame, f"분석 프레임 수: {analyzer.total_frames}", 
                                       (10, y_pos + 70), (200, 200, 200))
                analyzer.put_korean_text(frame, f"감정 다양성 지수: {analyzer.get_emotion_diversity_score()}%", 
                                       (10, y_pos + 100), (200, 200, 0))
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"\n=== 프레임 {frame_count} 상태 ===")
                print(f"현재 감정: {current_emotion} ({current_prob:.1f}%)")
                print("전체 감정 확률:")
                for emotion, prob in all_probs.items():
                    print(f"  {emotion}: {prob:.1f}%")
                print(f"전체 평균 점수: {score:.1f}")
                print(f"전체 평균 등급: {grade}")
                print(f"분석 프레임 수: {analyzer.total_frames}")
                print("=====================")
            
            cv2.imshow('Emotion Analysis', frame)
            
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
            continue
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 