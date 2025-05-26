import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from emotion import EmotionNet
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import os

# 모델 로드
device = torch.device("cpu")
model = EmotionNet()

checkpoint = torch.load("model.pth", map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# 감정 클래스
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2,
                     '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}

# 긍정/부정 감정 세트 정의
positive_set = {'기쁨', '당황', '중립'}
negative_set = {'슬픔', '불안', '분노', '상처'}

# 한글 폰트 설정
def get_font_path():
    """시스템에 설치된 한글 폰트 경로 반환"""
    # macOS 기본 한글 폰트
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "/System/Library/Fonts/PingFang.ttc",          # macOS
        "/Library/Fonts/NanumGothic.ttf",              # 나눔고딕
        "/Library/Fonts/MalgunGothic.ttf",             # 맑은 고딕
        "/Library/Fonts/NotoSansCJKkr-Regular.otf"     # Noto Sans
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return path
    
    return None

def set_korean_font():
    """matplotlib에 한글 폰트 설정"""
    font_path = get_font_path()
    if font_path:
        # 폰트 설정
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        return True
    return False

def put_korean_text(img, text, position, font_size=32, color=(255, 255, 255)):
    """한글 텍스트를 이미지에 그리는 함수"""
    # PIL Image로 변환
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 설정
    font_path = get_font_path()
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.loadDefault()
    else:
        font = ImageFont.loadDefault()
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color)
    
    # OpenCV 형식으로 변환
    return np.array(img_pil)

class WebcamEmotionAnalyzer:
    def __init__(self):
        self.results = []
        self.frame_count = 0
        self.start_time = None
        
        # 감정 가중치 (면접 상황에 맞게 조정)
        self.emotion_weights = {
            "기쁨": 80,       # 긍정적 (적절한 미소)
            "중립": 70,       # 중립적 (약간의 긍정점수)
            "당황": 35,       # 부정적 (면접에서 좋지 않음)
            "슬픔": 30,       # 부정적
            "분노": 40,       # 부정적
            "불안": 30,       # 부정적
            "상처": 30        # 부정적
        }

        # 평가 기준 임계값 (면접 상황에 맞게 조정)
        self.thresholds = {
            'neutral_ratio': 0.5,      # 중립 표정 비율 (가장 중요)
            'positive_ratio': 0.3,     # 긍정적 표정 비율 (기쁨)
            'negative_ratio': 0.2,     # 부정적 표정 비율 (최소화)
            'confidence_threshold': 0.6, # 신뢰도 임계값
            'happy_threshold': 0.2      # 기쁨 감정 최소 비율
        }

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        # 폰트 크기 설정
        self.font_sizes = {
            'main': 32,    # 주요 감정 표시
            'sub': 24,     # 확률 표시
            'info': 20     # 기타 정보
        }

    def analyze_webcam(self):
        """웹캠 실시간 감정 분석"""
        print("웹캠 초기화 중...")
        
        # 웹캠 장치 1번으로 직접 지정
        cap = cv2.VideoCapture(1)  # 0 → 1로 변경
        
        if not cap.isOpened():
            print("❌ 웹캠(1번)을 열 수 없습니다.")
            print("다음을 확인해주세요:")
            print("1. 웹캠이 컴퓨터에 제대로 연결되어 있는지")
            print("2. 다른 프로그램에서 웹캠을 사용 중이 아닌지")
            print("3. 웹캠 권한이 허용되어 있는지")
            return None

        # 웹캠 설정 확인
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"웹캠 설정: {width}x{height} @ {fps}fps")
        
        if width == 0 or height == 0:
            print("❌ 웹캠 해상도를 가져올 수 없습니다.")
            cap.release()
            return None

        self.start_time = time.time()
        self.frame_count = 0
        
        print("\n웹캠 분석 시작 (종료하려면 'q'를 누르세요)")
        print("=" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 프레임을 읽을 수 없습니다.")
                    break

                # 프레임 크기 확인
                if frame.size == 0:
                    print("❌ 빈 프레임이 감지되었습니다.")
                    break

                # BGR to RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 이미지 전처리
                image = Image.fromarray(rgb_frame)
                input_tensor = self.transform(image).unsqueeze(0)

                # 감정 예측
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    
                    # 모든 감정의 확률
                    all_probs = {emotion: float(prob) for emotion, prob in zip(class_labels, probs)}
                    
                    # 최고 확률의 감정
                    pred_idx = torch.argmax(probs).item()
                    pred_label = class_labels[pred_idx]
                    confidence = probs[pred_idx].item()

                # 결과 저장
                current_time = time.time() - self.start_time
                self.results.append({
                    'timestamp': current_time,
                    'emotion': pred_label,
                    'confidence': confidence,
                    'all_probs': all_probs
                })
                self.frame_count += 1

                # 화면에 표시할 텍스트 (한글)
                emotion_text = f"감정: {pred_label} ({confidence:.2f})"
                frame = put_korean_text(frame, emotion_text, (10, 30), 
                                      self.font_sizes['main'], (0, 255, 0))
                
                # 모든 감정의 확률 표시 (한글)
                y_pos = 70
                for emotion, prob in all_probs.items():
                    prob_text = f"{emotion}: {prob:.2f}"
                    frame = put_korean_text(frame, prob_text, (10, y_pos),
                                          self.font_sizes['sub'], (255, 255, 255))
                    y_pos += 30

                # 프레임 표시
                cv2.imshow('감정 분석', frame)  # 창 제목도 한글로 변경

                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n사용자가 분석을 종료했습니다.")
                    break

        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if self.frame_count > 0:
                print(f"\n분석 완료: {self.frame_count}개 프레임 분석")
                return self.results
            else:
                print("\n❌ 분석된 프레임이 없습니다.")
                return None

    def calculate_metrics(self):
        """분석 지표 계산"""
        if not self.results:
            return None

        emotions = [r['emotion'] for r in self.results]
        confidences = [r['confidence'] for r in self.results]
        total_samples = len(emotions)

        # 각 감정별 비율 계산
        emotion_ratios = {}
        for emotion in class_labels:
            count = emotions.count(emotion)
            emotion_ratios[emotion] = count / total_samples

        # 기본 비율 계산
        neutral_count = emotions.count("중립")
        happy_count = emotions.count("기쁨")
        negative_count = sum(1 for e in emotions if e not in {"중립", "기쁨"})

        neutral_ratio = neutral_count / total_samples
        happy_ratio = happy_count / total_samples
        negative_ratio = negative_count / total_samples

        # 감정 전이 분석 (면접 상황에 맞게 조정)
        transitions = []
        for i in range(1, len(emotions)):
            prev_emotion = emotions[i-1]
            curr_emotion = emotions[i]
            
            # 자연스러운 전이인지 평가
            prev_weight = self.emotion_weights[prev_emotion]
            curr_weight = self.emotion_weights[curr_emotion]
            
            # 급격한 변화 판단 (30점 이상 차이나면 급격한 변화로 판단)
            if abs(prev_weight - curr_weight) > 30:
                transitions.append('abrupt')
            else:
                transitions.append('natural')

        abrupt_transition_ratio = transitions.count('abrupt') / len(transitions) if transitions else 0
        
        # 안정성 점수 (중립과 기쁨이 얼마나 안정적으로 유지되는지)
        stability_score = 0
        if neutral_ratio >= self.thresholds['neutral_ratio']:
            stability_score += 0.6
        if happy_ratio >= self.thresholds['happy_threshold']:
            stability_score += 0.4

        # 신뢰도 분석
        avg_confidence = np.mean(confidences)
        low_confidence_ratio = sum(1 for c in confidences if c < self.thresholds['confidence_threshold']) / total_samples

        # 부정적 감정 분석
        negative_emotions = [e for e in emotions if e not in {"중립", "기쁨"}]
        negative_ratio = len(negative_emotions) / total_samples

        return {
            'emotion_ratios': emotion_ratios,
            'neutral_ratio': neutral_ratio,
            'happy_ratio': happy_ratio,
            'negative_ratio': negative_ratio,
            'abrupt_transition_ratio': abrupt_transition_ratio,
            'stability_score': stability_score,
            'avg_confidence': avg_confidence,
            'low_confidence_ratio': low_confidence_ratio
        }

    def generate_assessment(self):
        """종합 평가 생성"""
        metrics = self.calculate_metrics()
        if not metrics:
            return {
                'overall_score': 0,
                'grade': 'C',
                'key_findings': ["분석 데이터 부족"],
                'recommendations': []
            }

        assessment = {
            'overall_score': 0,
            'grade': 'C',
            'key_findings': [],
            'recommendations': []
        }

        # 1. 기본 점수 계산 (면접 상황에 맞게 조정)
        base_score = (
            metrics['neutral_ratio'] * 40 +     # 중립 표정 (가장 중요)
            metrics['happy_ratio'] * 30 +       # 기쁨 표정
            metrics['stability_score'] * 20 +   # 안정성
            (1 - metrics['negative_ratio']) * 10  # 부정적 감정 최소화
        )

        # 2. 감점 요소 적용
        penalties = 0
        
        # 급격한 감정 변화에 대한 감점
        if metrics['abrupt_transition_ratio'] > 0.2:
            penalties += 5
            assessment['key_findings'].append("△ 감정 변화가 다소 급격함")
        
        # 부정적 감정이 너무 많은 경우
        if metrics['negative_ratio'] > self.thresholds['negative_ratio']:
            penalties += 10
            assessment['key_findings'].append("⚠ 부정적인 표정이 다소 많음")
        
        # 신뢰도가 낮은 경우
        if metrics['low_confidence_ratio'] > 0.4:
            penalties += 5
            assessment['key_findings'].append("△ 표정 인식이 불안정함")

        # 3. 최종 점수 계산
        final_score = max(0, min(100, base_score - penalties))
        assessment['overall_score'] = round(final_score, 1)

        # 4. 등급 판정
        if final_score >= 75:
            assessment['grade'] = 'A'
            assessment['key_findings'].append("✓ 면접에 적합한 표정 관리")
        elif final_score >= 60:
            assessment['grade'] = 'B'
            assessment['key_findings'].append("○ 대체로 적절한 표정")
        else:
            assessment['grade'] = 'C'
            assessment['key_findings'].append("△ 표정 관리 개선 필요")

        # 5. 구체적인 권장사항 생성
        if metrics['neutral_ratio'] < self.thresholds['neutral_ratio']:
            assessment['recommendations'].append("• 중립적인 표정을 더 유지해보세요")
        if metrics['happy_ratio'] < self.thresholds['happy_threshold']:
            assessment['recommendations'].append("• 적절한 미소를 더 표현해보세요")
        if metrics['negative_ratio'] > self.thresholds['negative_ratio']:
            assessment['recommendations'].append("• 부정적인 표정을 줄여보세요")
        if metrics['abrupt_transition_ratio'] > 0.2:
            assessment['recommendations'].append("• 표정 변화를 더 자연스럽게 해보세요")
        if metrics['low_confidence_ratio'] > 0.4:
            assessment['recommendations'].append("• 표정을 더 명확하게 표현해보세요")

        return assessment

    def create_analysis_chart(self):
        """분석 결과 시각화"""
        if not self.results:
            return

        # 한글 폰트 설정
        if not set_korean_font():
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 차트의 한글이 깨질 수 있습니다.")

        metrics = self.calculate_metrics()
        
        # 1. 감정 분포 파이 차트
        emotions = [r['emotion'] for r in self.results]
        emotion_counts = Counter(emotions)
        
        # 감정을 긍정/중립/부정으로 그룹화
        grouped_data = {
            '긍정': emotion_counts.get('기쁨', 0),
            '중립': emotion_counts.get('중립', 0),
            '부정': sum(emotion_counts.get(e, 0) for e in ['당황', '슬픔', '분노', '불안', '상처'])
        }

        # 차트 스타일 설정
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('면접 표정 분석 결과', fontsize=16, y=0.95)  # pad → y로 변경
        
        # 배경색 설정
        fig.patch.set_facecolor('#f0f0f0')
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('white')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 파이 차트
        colors = ['#4CAF50', '#2196F3', '#F44336']
        wedges, texts, autotexts = ax1.pie(grouped_data.values(), 
                                         labels=grouped_data.keys(), 
                                         autopct='%1.1f%%',
                                         colors=colors, 
                                         startangle=90,
                                         textprops={'fontsize': 12})
        ax1.set_title('표정 분포', fontsize=14, pad=15)

        # 시간별 감정 변화
        timestamps = [r['timestamp'] for r in self.results]
        emotion_values = [self.emotion_weights[r['emotion']] for r in self.results]
        
        # 5-point 이동 평균
        if len(emotion_values) >= 5:
            moving_avg = np.convolve(emotion_values, np.ones(5), 'valid') / 5
            moving_timestamps = timestamps[2:-2]
            
            ax2.plot(timestamps, emotion_values, 'o-', alpha=0.3, label='실제값')
            ax2.plot(moving_timestamps, moving_avg, 'r-', linewidth=2, label='이동평균')
            ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='중립 기준선')
            ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='부정 기준선')
            ax2.set_xlabel('시간 (초)', fontsize=12)
            ax2.set_ylabel('표정 점수', fontsize=12)
            ax2.set_title('시간별 표정 변화', fontsize=14, pad=15)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # 신뢰도 분포
        confidences = [r['confidence'] for r in self.results]
        ax3.hist(confidences, bins=10, color='#9C27B0', alpha=0.7, edgecolor='black')
        ax3.axvline(x=self.thresholds['confidence_threshold'], color='red', linestyle='--',
                   label=f'기준선: {self.thresholds["confidence_threshold"]}')
        ax3.set_xlabel('신뢰도', fontsize=12)
        ax3.set_ylabel('빈도', fontsize=12)
        ax3.set_title('표정 인식 신뢰도', fontsize=14, pad=15)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 주요 지표 비교
        indicators = ['중립 비율', '기쁨 비율', '안정성', '부정 비율']
        current_values = [
            metrics['neutral_ratio'],
            metrics['happy_ratio'],
            metrics['stability_score'],
            metrics['negative_ratio']
        ]
        thresholds = [
            self.thresholds['neutral_ratio'],
            self.thresholds['happy_threshold'],
            0.7,  # 안정성 임계값
            self.thresholds['negative_ratio']
        ]

        x = np.arange(len(indicators))
        width = 0.35
        
        ax4.bar(x - width/2, current_values, width, label='현재', color='#1976D2', alpha=0.8)
        ax4.bar(x + width/2, thresholds, width, label='기준', color='#FFA726', alpha=0.8)
        
        ax4.set_xlabel('지표', fontsize=12)
        ax4.set_ylabel('비율', fontsize=12)
        ax4.set_title('주요 지표 비교', fontsize=14, pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(indicators, fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 차트 여백 조정
        plt.tight_layout()
        
        # 차트 저장
        plt.savefig("webcam_analysis_result.png", dpi=200, bbox_inches='tight')
        plt.show()
        print("📊 분석 결과 차트가 'webcam_analysis_result.png'로 저장되었습니다.")

def main():
    analyzer = WebcamEmotionAnalyzer()
    
    print("🎭 실시간 웹캠 표정 분석")
    print("=" * 50)
    print("• 웹캠이 시작되면 실시간으로 감정이 분석됩니다.")
    print("• 종료하려면 'q' 키를 누르세요.")
    print("• 종료 후 분석 결과가 표시됩니다.")
    
    # 폰트 확인
    font_path = get_font_path()
    if font_path:
        print(f"✓ 한글 폰트 사용: {os.path.basename(font_path)}")
        # matplotlib 폰트 설정
        set_korean_font()
    else:
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    print("=" * 50)
    
    # 웹캠 분석 실행
    results = analyzer.analyze_webcam()
    
    if results is None:
        print("\n❌ 웹캠 분석을 완료할 수 없습니다.")
        return
    
    # 평가 생성
    assessment = analyzer.generate_assessment()
    
    print("\n📋 분석 결과")
    print("=" * 50)
    print(f"💯 종합 점수: {assessment['overall_score']}")
    print(f"📊 등급: {assessment['grade']}")
    
    print("\n🔍 주요 발견사항:")
    for finding in assessment['key_findings']:
        print(f"   {finding}")
    
    if assessment['recommendations']:
        print("\n💡 개선 제안:")
        for rec in assessment['recommendations']:
            print(f"   {rec}")
    
    print("\n📈 분석 차트 생성 중...")
    analyzer.create_analysis_chart()
    
    print("\n✅ 분석 완료!")

if __name__ == "__main__":
    main() 