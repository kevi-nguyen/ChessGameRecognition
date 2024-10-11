import io
import cv2
import numpy as np
import pyrealsense2 as rs
import uvicorn
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


def get_frame():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())

    # Rotate the image 90 degrees to the right (clockwise)
    rotated_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)

    return color_image



@app.get("/current_frame")
def current_frame():
    frame = get_frame()
    if frame is None:
        return {"error": "No frame available"}

    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Convert to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return JSONResponse(content={"image_base64": img_base64})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)