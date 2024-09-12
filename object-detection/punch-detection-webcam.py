# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Roboflow API key from the environment variables
roboflow_api_key = os.getenv("ROBOFLOW_API")

# Ensure the API key is available
if not roboflow_api_key:
    raise ValueError("ROBOFLOW_API_KEY is not set in the environment.")

# Set the API key in your environment
os.environ["ROBOFLOW_API_KEY"] = roboflow_api_key

# alternatively, set your API key in your coding environment
# export ROBOFLOW_API_KEY=<your api key>

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="boxinghub/3", # Roboflow model to use
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=render_boxes, # Function to run after each prediction
)
pipeline.start()
pipeline.join()