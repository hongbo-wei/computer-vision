from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw


load_dotenv()  # Load environment variables from .env file
roboflow_api = os.getenv("ROBOFLOW_API")

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api
)

image_file = "object-detection/boxpunch_1.jpg"

# infer on a local image
result = CLIENT.infer(image_file, model_id="boxinghub/2")

# Load the image
image = Image.open(image_file)
draw = ImageDraw.Draw(image)  # Create a drawing object

# Draw boxes around detected objects
for prediction in result["predictions"]:
    x, y = prediction['x'], prediction['y']
    width, height = prediction['width'], prediction['height']
    
    # Calculate the coordinates of the bounding box
    left, top = x - width / 2, y - height / 2
    right, bottom = x + width / 2, y + height / 2
    
    # Draw the bounding box
    draw.rectangle([left, top, right, bottom], outline="green", width=2)
    
    # Optionally, add the class name and confidence above the bounding box
    label = f"{prediction['class']} ({prediction['confidence']:.2f})"
    draw.text((left, top - 10), label, fill="green")

    print(f"Detected: '{prediction['class']}' with confidence: {prediction['confidence']:.2f}")
    
# Save or display the image with boxes
output_image_path = "object-detection/boxing_punch_classification.jpg"
image.save(output_image_path)
image.show()  # This will display the image with the boxes