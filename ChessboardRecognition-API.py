import ast
import base64
from typing import Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi import Form
from pydantic import BaseModel

from ChessboardRecognition import ChessboardRecognition
from ChessboardStateLogic import ChessboardStateLogic

app = FastAPI()


class ResponseData(BaseModel):
    board_state: str
    src_points: str
    dst_points: str
    orientation: str


@app.post("/get_dst_points")
def get_frame_resolution(frame_str: str = Form(...)):
    frame = decode_base64_image(frame_str)
    return ChessboardRecognition().get_frame_resolution(frame)


@app.post("/get_board_state")
def get_board_state(frame_str: str = Form(...),
                    src_points_str: str = Form(...),
                    dst_points_str: str = Form(...),
                    orientation: str = Form(...)):
    frame = decode_base64_image(frame_str)
    src_points: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ast.literal_eval(
        src_points_str)
    dst_points: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ast.literal_eval(
        dst_points_str)

    frame = ChessboardRecognition().transform_image(frame, src_points, dst_points)
    board_state = ChessboardRecognition().get_board_state(frame)
    board_state = ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation)
    return ChessboardStateLogic().board_to_string(board_state)


@app.post("/get_src_points")
def find_chessboard_corners_green(frame_str: str = Form(...)):
    frame = decode_base64_image(frame_str)
    return ChessboardRecognition().find_chessboard_corners_green(frame)


@app.post("/initialize_game")
def initialize_game(frame_str: str = Form(...)):
    frame = decode_base64_image(frame_str)
    src_points = ChessboardRecognition().find_chessboard_corners_green(frame)
    dst_points = ChessboardRecognition().get_frame_resolution(frame)
    transformed_image = ChessboardRecognition().transform_image(frame, src_points, dst_points)
    board_state = ChessboardRecognition().get_board_state(transformed_image)
    orientation = ChessboardStateLogic().determine_orientation(board_state)
    board_state = ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation)
    return ResponseData(board_state=ChessboardStateLogic().board_to_string(board_state), src_points=str(src_points),
                        dst_points=str(dst_points), orientation=orientation)


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
    # base64_string = encode_image_to_base64("images/IMG_5388 Large.jpeg")
    # file_path = "output.txt"
    # save_base64_to_file(base64_string, file_path)
    # frame = decode_base64_image(base64_string)
    # cv2.imshow('Chessboard', frame)
    # cv2.waitKey(0)
    # cv2.imshow('Chessboard', frame)
    # ChessboardRecognition().find_chessboard_corners_green(frame)

    uvicorn.run(app, host="0.0.0.0", port=8081)
