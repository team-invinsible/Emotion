import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from emotion import EmotionNet
from collections import Counter
from moviepy.editor import VideoFileClip
from PIL import Image
import argparse

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

# 영상 → 프레임 → 감정 추론
def predict_emotions_from_video(video_path):
    clip = VideoFileClip(video_path)
    # results = [] # 결과를 리스트에 저장하는 대신 yield 사용

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    for i, frame in enumerate(clip.iter_frames(fps=10)):  # 초당 3프레임
        image = Image.fromarray(frame)
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = class_labels[pred_idx]
            score = probs[pred_idx].item()

        # results.append((pred_label, score)) # 리스트에 추가하는 대신 yield
        yield (pred_label, score) # 예측 결과를 yield

    # return results # 리스트 반환 대신 generator 반환

# 평가 로직
def evaluate_emotions(results):
    total = len(results)
    emotions = [r[0] for r in results]
    scores = dict((e, []) for e in class_labels)
    for label, score in results:
        scores[label].append(score)

    positive_count = sum([1 for e in emotions if e in positive_set])
    negative_count = sum([1 for e in emotions if e in negative_set])
    neutral_count = emotions.count("중립")
    happy_scores = scores['기쁨']

    transition_count = sum(1 for i in range(1, total) if emotions[i] != emotions[i-1])

    happy_score_avg = np.mean(happy_scores) if happy_scores else 0.0
    neutral_ratio = neutral_count / total
    positive_ratio = positive_count / total
    negative_ratio = negative_count / total

    forced_smile_penalty = -5 if happy_score_avg < 0.5 and negative_count > 0 else 0
    flat_penalty = -5 if neutral_ratio >= 0.9 else 0

    # 감정별 가중치 기반 평가 점수 계산
    emotion_weights = {
        "기쁨": 1.0,
        "당황": 0.8,
        "중립": 0.5,
        "슬픔": -0.8,
        "분노": -1.0,
        "불안": -0.7,
        "상처": -0.6
    }
    emotion_score = sum(emotion_weights.get(e, 0) for e in emotions) / total

    # 감정 다양성 점수
    diversity_score = len(set(emotions)) / len(class_labels)

    # 감정 전이 품질 보너스
    good_transitions = [('불안', '기쁨'), ('슬픔', '중립'), ('중립', '기쁨')]
    transition_bonus = sum(
        2 for i in range(1, total)
        if (emotions[i-1], emotions[i]) in good_transitions
    )

    # 감정 일관성
    emotion_count = Counter(emotions)
    most_common_emotion, freq = emotion_count.most_common(1)[0]
    consistency = freq / total

    # 최종 점수 계산 (개선 버전)
    score = (
        emotion_score * 25 +
        consistency * 5 +
        diversity_score * 10 +
        transition_bonus +
        happy_score_avg * 10 +
        forced_smile_penalty +
        flat_penalty
    )

    # 등급 판정
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    else:
        grade = "C"

    return {
        "점수": round(score, 2),
        "등급": grade,
        "긍정 비율": round(positive_ratio * 100, 1),
        "부정 비율": round(negative_ratio * 100, 1),
        "기쁨 평균 score": round(happy_score_avg, 3),
        "중립 비율": round(neutral_ratio * 100, 1),
        "감정 전이 수": transition_count,
        "forced smile 감점": forced_smile_penalty,
        "flat 감정 감점": flat_penalty
    }

import matplotlib.pyplot as plt

def plot_emotion_timeline(results):
    emotion_colors = {
        '기쁨': 'gold',
        '슬픔': 'blue',
        '중립': 'gray',
        '분노': 'red',
        '불안': 'purple',
        '상처': 'brown',
        '당황': 'green',
        '분노': 'black',
        '상처': 'darkgreen'
    }

    frame_idx = list(range(len(results)))
    emotion_labels = [e[0] for e in results]
    emotion_scores = [e[1] for e in results]
    colors = [emotion_colors.get(label, 'black') for label in emotion_labels]

    plt.figure(figsize=(12, 5))
    plt.scatter(frame_idx, emotion_scores, c=colors, s=100, edgecolors='k', alpha=0.8)
    plt.plot(frame_idx, emotion_scores, linestyle='--', color='lightgray', alpha=0.6)

    for i, label in enumerate(emotion_labels):
        plt.text(frame_idx[i], emotion_scores[i] + 0.02, label, ha='center', fontsize=9)

    plt.ylim(0, 1.05)
    plt.xlabel("Frame Number (sec * 3)")
    plt.ylabel("Emotion score (%)")
    plt.title("Emotion Predict")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("emotion_timeline.png")
    plt.show()
    print("✅ 감정 타임라인 그래프가 'emotion_timeline.png'로 저장되었습니다.")

if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='면접 영상 표정 분석')
    parser.add_argument('video_path', help='분석할 비디오 파일 경로')
    args = parser.parse_args()

    all_results = [] # 전체 결과를 저장할 리스트
    print(f"비디오 파일 분석 중: {args.video_path}")
    print("=" * 50)
    
    try:
        print("비디오 프레임별 감정 예측 결과:")
        for i, (label, score) in enumerate(predict_emotions_from_video(args.video_path)):
            all_results.append((label, score))
            if i % 10 == 0:  # 10프레임마다 진행상황 출력
                print(f"프레임 {i+1}: {label} (Score: {score:.3f})")

        print("\n전체 비디오 평가 결과:")
        report = evaluate_emotions(all_results)

        print("\n📋 분석 결과")
        print("=" * 50)
        for k, v in report.items():
            print(f"{k}: {v}")

        print("\n📈 감정 타임라인 차트 생성 중...")
        plot_emotion_timeline(all_results)
        
        print("\n✅ 분석 완료!")
        
    except FileNotFoundError:
        print(f"❌ 오류: '{args.video_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")