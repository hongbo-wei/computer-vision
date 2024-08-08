from ultralytics import YOLO

# load pre-trained YOLOv8n model
model = YOLO('yolov8m-pose.pt')

# Run inference on the webcam
# for video, change source to "video.mp4"
results = model (source=0, show=True, conf=0.3, save=True)