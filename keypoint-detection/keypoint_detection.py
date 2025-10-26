import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe姿态检测
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

# 创建姿态检测器
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 水平翻转图像，使其像镜子一样
        frame = cv2.flip(frame, 1)
        
        # 转换颜色空间 BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行姿态检测
        results = pose.process(rgb_frame)
        
        # 在图像上绘制姿态关键点
        if results.pose_landmarks:
            # 绘制姿态关键点和连接线
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # 获取图像尺寸
            height, width = frame.shape[:2]
            
            # 计算边界框
            landmarks = results.pose_landmarks.landmark
            x_coords = [landmark.x * width for landmark in landmarks]
            y_coords = [landmark.y * height for landmark in landmarks]
            
            # 过滤掉置信度低的点
            valid_coords = [(x, y) for x, y, landmark in zip(x_coords, y_coords, landmarks) 
                          if landmark.visibility > 0.5]
            
            if valid_coords:
                x_coords, y_coords = zip(*valid_coords)
                
                # 计算边界框
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # 添加边距
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(width, x_max + margin)
                y_max = min(height, y_max + margin)
                
                # 绘制蓝色边界框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # 添加置信度文本
                confidence = 0.96  # 模拟置信度
                text = f"person {confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 绘制文本背景
                cv2.rectangle(frame, (x_min, y_min - 25), (x_min + text_size[0] + 10, y_min), (255, 0, 0), -1)
                
                # 绘制文本
                cv2.putText(frame, text, (x_min + 5, y_min - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示图像
        cv2.imshow('Human Pose Detection (Press Q to quit)', frame)
        
        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()