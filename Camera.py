import io
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import uvicorn
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt

app = FastAPI()

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_all_streams()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.mjpeg, 30)
pipeline.start(config)


def get_frame():
    frames = pipeline.wait_for_frames(5000)
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    rotated_image = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image



@app.get("/get_snapshot")
def get_snapshot():
    frame = get_frame()
    if frame is None:
        return {"error": "No frame available"}

    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Convert to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64


if __name__ == "__main__":
    frame = get_frame()
    # Convert BGR to RGB for proper color representation
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.imwrite("saved_frame_rgb.jpg", frame)  # Save the RGB frame
    # plt.imshow(rgb_frame)
    # plt.axis('off')  # Hide axes
    # plt.show()
    uvicorn.run(app, host="0.0.0.0", port=8086)