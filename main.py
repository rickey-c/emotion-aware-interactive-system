from fer import FER
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib
import time

# --- 使用交互式 TkAgg 后端（允许实时刷新图表）
matplotlib.use('TkAgg')

# --- 性能配置：每处理 N 帧执行一次情绪检测 ---
PROCESS_EVERY_N_FRAMES = 4

# --- 情绪对应的屏幕颜色（BGR 格式） ---
EMOTION_COLORS = {
    'angry': (0, 0, 255),       # 红色：愤怒
    'disgust': (0, 255, 0),     # 绿色：厌恶
    'fear': (128, 0, 128),      # 紫色：恐惧
    'happy': (0, 255, 255),     # 黄色：高兴
    'sad': (255, 0, 0),         # 蓝色：悲伤
    'surprise': (0, 165, 255),  # 橙色：惊讶
    'neutral': (128, 128, 128), # 灰色：中性
    'None': (0, 0, 0)           # 黑色：未检测到情绪
}

# 初始化情绪检测器（使用 MTCNN 提升人脸检测精度）
detector = FER(mtcnn=True)

# 打开摄像头
cap = cv2.VideoCapture(0)
camera_fps = cap.get(cv2.CAP_PROP_FPS)
if camera_fps == 0:
    camera_fps = 30.0  # 若无法获取帧率，默认30FPS

# 输出视频帧率设置
out_frame_rate = camera_fps / (PROCESS_EVERY_N_FRAMES + 1)
print(f"Camera FPS: {camera_fps:.2f}, Processing 1 in {PROCESS_EVERY_N_FRAMES + 1} frames.")
print(f"Output video FPS set to: {out_frame_rate:.2f}")

# 初始化视频保存对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_video.avi', fourcc, out_frame_rate, (640, 480))

# 初始化实时情绪柱状图
plt.ion()
fig, ax = plt.subplots()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
bars = ax.bar(emotion_labels, [0] * len(emotion_labels), color='lightblue')
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Real-time Emotion Detection')
ax.set_xticks(range(len(emotion_labels)))
ax.set_xticklabels(emotion_labels, rotation=45)

# GIF 记录配置（用于保存情绪变化）
gif_duration = (PROCESS_EVERY_N_FRAMES + 1) / camera_fps
gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=gif_duration)

# 情绪数据累积记录
emotion_statistics = []

# --- 状态变量：记录上一帧的情绪与检测框 ---
frame_count = 0
last_box = None
last_emotion_text = ""
last_emotion_type = 'None'  # 当前情绪类型
# ---------------------------------------------

# --- 创建一个单独窗口用于展示当前情绪颜色 ---
color_window = np.zeros((300, 300, 3), dtype=np.uint8)

def update_chart(detected_emotions, bars, ax, fig):
    """更新实时柱状图"""
    ax.clear()
    ax.bar(emotion_labels, [detected_emotions.get(e, 0) for e in emotion_labels], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Real-time Emotion Detection')
    ax.set_xticks(range(len(emotion_labels)))
    ax.set_xticklabels(emotion_labels, rotation=45)
    fig.canvas.draw()
    fig.canvas.flush_events()


# --- 主循环开始 ---
webcam_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        display_frame = frame.copy()
        frame_count += 1

        # 仅每 N 帧执行一次情绪识别（提升性能）
        if frame_count % (PROCESS_EVERY_N_FRAMES + 1) == 0:
            result = detector.detect_emotions(frame)
            largest_face = None
            max_area = 0

            # 选取最大的人脸作为主情绪
            for face in result:
                box = face["box"]
                x, y, w, h = box
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = face

            if largest_face:
                # --- 提取情绪并标注 ---
                box = largest_face["box"]
                current_emotions = largest_face["emotions"]
                emotion_statistics.append(current_emotions)

                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                emotion_type = max(current_emotions, key=current_emotions.get)
                emotion_score = current_emotions[emotion_type]
                emotion_text = f"{emotion_type}: {emotion_score:.2f}"

                cv2.putText(frame, emotion_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 保存当前情绪结果（用于下一帧显示）
                last_box = box
                last_emotion_text = emotion_text
                last_emotion_type = emotion_type

                # 实时更新图表与GIF
                update_chart(current_emotions, bars, ax, fig)
                fig.canvas.draw()
                canvas = fig.canvas
                width, height = canvas.get_width_height()
                argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
                try:
                    argb = argb.reshape(height, width, 4)
                except ValueError:
                    print(f"[⚠] Reshape error: argb size={argb.size}, expected={height * width * 4}")
                    continue

                rgb = argb[:, :, [1, 2, 3]]
                rgb_resized = cv2.resize(rgb, (640, 480))
                gif_writer.append_data(rgb_resized)

            else:
                # 未检测到人脸，清空状态
                last_box = None
                last_emotion_text = ""
                last_emotion_type = 'None'

            # 保存处理过的视频帧
            out.write(frame)

        # --- 显示阶段（每帧都执行） ---
        # 绘制上次检测的结果
        if last_box:
            x, y, w, h = last_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, last_emotion_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示检测视频窗口
        cv2.imshow('Emotion Detection', display_frame)

        # ✅ 新增：显示当前情绪对应的颜色窗口
        display_color = EMOTION_COLORS.get(last_emotion_type, (0, 0, 0))
        color_window[:] = display_color
        cv2.imshow('Emotion Color', color_window)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    webcam_end_time = time.time()
    print(f"Webcam active time: {webcam_end_time - webcam_start_time:.2f} seconds")

    # 释放资源
    cap.release()
    out.release()
    gif_writer.close()
    cv2.destroyAllWindows()
    plt.close(fig)

    # 生成累计情绪趋势图
    if emotion_statistics:
        emotion_df = pd.DataFrame(emotion_statistics)
        plt.figure(figsize=(10, 10))
        for emotion in emotion_labels:
            plt.plot(emotion_df[emotion].cumsum(), label=emotion)
        plt.title('Cumulative Emotion Statistics Over Time')
        plt.xlabel('Processed Frame')
        plt.ylabel('Cumulative Confidence')
        plt.legend()
        plt.savefig('cumulative_emotions.jpg')
        plt.close()

    print("✅ Emotion detection finished. Files saved:")
    print(" - emotion_video.avi")
    print(" - emotion_chart.gif")
    print(" - cumulative_emotions.jpg")
