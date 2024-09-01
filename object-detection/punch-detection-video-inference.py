from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API")

rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace().project("boxinghub")
model = project.version("3").model

job_id, signed_url, expire_time = model.predict_video(
    "side-view.mp4",
    fps=2, # adjust the frames per second
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

# make a dictionary to record each class and the number of times it appears
class_counts = {}
# Iteratively print the prediction class
for result in results['boxinghub']:
    if not result["predictions"]:
            result["predictions"] = [{"class": "No object detected", "confidence": 0.0, "x": 0, "y": 0, "width": 0, "height": 0}]
    prediction_class = result['predictions'][0]['class']
    if prediction_class not in class_counts:
        class_counts[prediction_class] = 1
    else:
        class_counts[prediction_class] += 1

# Calculate the total duration of the video
frame_count = len(results['boxinghub'])
fps = 2  # As defined in the predict_video call
video_duration = frame_count / fps / 60  # Convert to minutes

# Print the summary
print(f"In the {video_duration:.2f} mins video, a boxer throws:")
for punch_type, count in class_counts.items():
    print(f"{punch_type}: {count} times")
