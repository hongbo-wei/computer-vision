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

my_image = "screw.jpg"

# infer on a local image
result = CLIENT.infer(my_image, model_id="screw-detection-dwylm/1")

# Load the image
image = Image.open(my_image)
draw = ImageDraw.Draw(image)  # Create a drawing object

# Draw boxes around detected objects
for prediction in result["predictions"]:
    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    
    # Calculate the coordinates of the bounding box
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2
    
    # Draw the bounding box
    draw.rectangle([left, top, right, bottom], outline="green", width=2)
    
    # Optionally, add the class name and confidence above the bounding box
    label = f"{prediction['class']} ({prediction['confidence']:.2f})"
    draw.text((left, top - 10), label, fill="green")

    print(f"Detected: '{prediction['class']}' with confidence: {prediction['confidence']:.2f}")
    
# Save or display the image with boxes
output_image_path = "screw_with_boxes.jpg"
image.save(output_image_path)
image.show()  # This will display the image with the boxes