import ast
import base64
import os
import time
from typing import Tuple

import chess
import cv2
import numpy as np
import serial
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from stockfish import Stockfish

from ChessboardLogic import ChessboardLogic
from ChessboardRecognition import ChessboardRecognition
from ChessboardStateLogic import ChessboardStateLogic

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH') or 'stockfish'
stockfish = Stockfish(path=STOCKFISH_PATH)

app = FastAPI()

# port = '/dev/tty.usbserial-AQ027FTG'
# baudrate = 9600
# ser = serial.Serial(port, baudrate, timeout=1)

button_was_pressed = False
button_is_pressed = False

class ResponseDataCobotMove(BaseModel):
    double_move: bool
    start_i: str
    start_j: str
    end_i: str
    end_j: str
    sec_start_i: str
    sec_start_j: str
    sec_end_i: str
    sec_end_j: str

class ResponseDataInitGame(BaseModel):
    board_state: str
    src_points: str
    dst_points: str
    orientation: str

# @app.get("/wait_for_button_press")
# def wait_for_button_press():
#     """
#     Endpoint to wait for the button to be pressed and then return the status.
#     """
#     start_time = time.time()
#     timeout = 60
#
#     while True:
#         try:
#             if ser.in_waiting > 0:
#                 data = ser.read(ser.in_waiting)
#                 return {"status": "pressed"}
#             # Check for timeout
#             if time.time() - start_time > timeout:
#                 raise HTTPException(status_code=408, detail="Request timed out")
#
#             time.sleep(0.1)
#
#         except serial.SerialException as e:
#             raise HTTPException(status_code=500, detail=f"Serial error: {e}")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/get_best_move")
def get_move(fen: str = Form(...)):
    # Validate the FEN string
    try:
        chess.Board(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    return best_move


@app.post("/update_chessboard")
def move_piece(fen: str = Form(...), move: str = Form(...)):
    return ChessboardLogic().move_piece(fen, move)


@app.post("/check_illegal_move")
def check_illegal_move(fen: str = Form(...), move: str = Form(...)):
    return ChessboardLogic().is_illegal_move(fen, move)


@app.post("/check_game_end")
def is_checkmate(fen: str = Form(...)):
    return ChessboardLogic().is_checkmate(fen)


@app.post("/get_move")
def find_moved_piece(prev_state_string: str = Form(...), curr_state_string: str = Form(...)):
    prev_state = ChessboardStateLogic().string_to_board(prev_state_string)
    curr_state = ChessboardStateLogic().string_to_board(curr_state_string)
    return ChessboardStateLogic().find_moved_piece(prev_state, curr_state)


@app.post("/update_board_state")
def update_board_state(board_state_string: str = Form(...), move: str = Form(...), special: bool = Form(...)):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    return ChessboardStateLogic().board_to_string(ChessboardStateLogic().update_board_state(board_state, move, special))

@app.post("/check_special_move")
def is_special_move(fen: str = Form(...), move: str = Form(...)):
    return ChessboardLogic().is_special_move(fen, move)


@app.post("/get_cobot_move")
def coordinates_to_cobot_move(board_state_string: str = Form(...),
                              move: str = Form(...),
                              special: bool = Form(...),
                              orientation: str = Form(...)):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    result = ChessboardStateLogic().coordinates_to_cobot_move(board_state, move, special, orientation)
    return ResponseDataCobotMove(double_move=result[0], start_i=result[1], start_j=result[2], end_i=result[3], end_j=result[4],
                        sec_start_i=result[5], sec_start_j=result[6], sec_end_i=result[7], sec_end_j=result[8])

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
    return ResponseDataInitGame(board_state=ChessboardStateLogic().board_to_string(board_state), src_points=str(src_points),
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
    uvicorn.run(app, host="0.0.0.0", port=8085)
