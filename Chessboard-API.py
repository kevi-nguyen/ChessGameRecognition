from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi import Form
from ChessboardLogic import ChessboardLogic
from ChessboardStateLogic import ChessboardStateLogic
from pydantic import BaseModel

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


@app.post("/get_orientation")
def determine_orientation(initial_board_state_string: str = Form(...)):
    initial_board_state = ChessboardStateLogic().string_to_board(initial_board_state_string)
    return ChessboardStateLogic().determine_orientation(initial_board_state)


@app.post("/get_rotated_board")
def rotate_board_to_bottom(board_state_string: str = Form(...), orientation: str = Form(...)):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    return ChessboardStateLogic().board_to_string(
        ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation))


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
    return ResponseData(double_move=result[0], start_i=result[1], start_j=result[2], end_i=result[3], end_j=result[4],
                        sec_start_i=result[5], sec_start_j=result[6], sec_end_i=result[7], sec_end_j=result[8])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
