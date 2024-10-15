import ast
import base64
from typing import Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi import Form
from pydantic import BaseModel

from ChessboardLogic import ChessboardLogic
from ChessboardRecognition import ChessboardRecognition
from ChessboardStateLogic import ChessboardStateLogic

app = FastAPI()


class ResponseData(BaseModel):
    double_move: bool
    start_i: str
    start_j: str
    end_i: str
    end_j: str
    sec_start_i: str
    sec_start_j: str
    sec_end_i: str
    sec_end_j: str
    capture_blue: str
    capture_red: str


class ResponseDataInit(BaseModel):
    board_state: str
    src_points: str
    dst_points: str
    orientation: str


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
                              orientation: str = Form(...),
                              capture_blue: str = Form(...),
                              capture_red: str = Form(...)):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    capture_blue = tuple(map(int, capture_blue.strip('()').split(',')))
    capture_red = tuple(map(int, capture_red.strip('()').split(',')))
    result = ChessboardStateLogic().coordinates_to_cobot_move(board_state, move, special, orientation, capture_blue,
                                                              capture_red)
    return ResponseData(double_move=result[0], start_i=result[1], start_j=result[2], end_i=result[3], end_j=result[4],
                        sec_start_i=result[5], sec_start_j=result[6], sec_end_i=result[7], sec_end_j=result[8],
                        capture_blue=result[9], capture_red=result[10])


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


@app.post("/initialize_game")
def initialize_game(frame_str: str = Form(...)):
    frame = decode_base64_image(frame_str)
    src_points = ChessboardRecognition().find_chessboard_corners_green(frame)
    dst_points = ChessboardRecognition().get_frame_resolution(frame)
    transformed_image = ChessboardRecognition().transform_image(frame, src_points, dst_points)
    board_state = ChessboardRecognition().get_board_state(transformed_image)
    orientation = ChessboardStateLogic().determine_orientation(board_state)
    board_state = ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation)
    return ResponseDataInit(board_state=ChessboardStateLogic().board_to_string(board_state), src_points=str(src_points),
                            dst_points=str(dst_points), orientation=orientation)


def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)

    np_arr = np.frombuffer(img_data, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
