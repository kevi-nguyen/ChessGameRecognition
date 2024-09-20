from typing import Tuple

from pydantic import BaseModel
import cv2
import numpy as np
from ChessboardRecognition import ChessboardRecognition
from ChessboardStateLogic import ChessboardStateLogic
from fastapi import FastAPI, HTTPException
import base64
import gzip
import io
from PIL import Image
import uvicorn

app = FastAPI()

class ImageData(BaseModel):
    frame_str: str

class ImageDataWithPoints(BaseModel):
    frame_str: str
    src_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    dst_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

@app.post("/get_dst_points")
def get_frame_resolution(data: ImageData):
    frame_str = data.frame_str
    frame = decode_base64_image(frame_str)
    return ChessboardRecognition().get_frame_resolution(frame)


@app.post("/get_board_state")
def get_board_state(data: ImageDataWithPoints):
    frame_str = data.frame_str
    src_points = data.src_points
    dst_points = data.dst_points
    frame = decode_base64_image(frame_str)
    frame = ChessboardRecognition().transform_image(frame, src_points, dst_points)
    return ChessboardStateLogic().board_to_string(ChessboardRecognition().get_board_state(frame))


@app.post("/get_src_points")
def find_chessboard_corners_green(data: ImageData):
    frame_str = data.frame_str
    frame = decode_base64_image(frame_str)
    return ChessboardRecognition().find_chessboard_corners_green(frame)


def decode_base64_image(base64_str):
    # Decode the base64 string into bytes
    img_data = base64.b64decode(base64_str)

    # Convert the bytes into a numpy array
    np_arr = np.frombuffer(img_data, np.uint8)

    # Decode the numpy array into an image using OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
    return base64_string

def save_base64_to_file(base64_string: str, file_path: str):
    with open(file_path, "w") as file:
        file.write(base64_string)


if __name__ == "__main__":
    #base64_string = encode_image_to_base64("images/IMG_5388 Large.jpeg")
    #file_path = "output.txt"
    #save_base64_to_file(base64_string, file_path)
    #frame = decode_base64_image(base64_string)
    #cv2.imshow('Chessboard', frame)
    #cv2.waitKey(0)
    #cv2.imshow('Chessboard', frame)
    #ChessboardRecognition().find_chessboard_corners_green(frame)

    uvicorn.run(app, host="0.0.0.0", port=8081)
