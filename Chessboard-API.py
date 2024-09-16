from fastapi import FastAPI, HTTPException
import uvicorn
from ChessboardLogic import ChessboardLogic
from ChessboardStateLogic import ChessboardStateLogic

app = FastAPI()


@app.get("/update_chessboard")
def move_piece(fen: str, move: str):
    return ChessboardLogic().move_piece(fen, move)


@app.get("/check_game_end")
def is_checkmate(fen: str):
    return ChessboardLogic().is_checkmate(fen)


@app.get("/get_move")
def find_moved_piece(prev_state_string: str, curr_state_string: str):
    prev_state = ChessboardStateLogic().string_to_board(prev_state_string)
    curr_state = ChessboardStateLogic().string_to_board(curr_state_string)
    return ChessboardStateLogic().find_moved_piece(prev_state, curr_state)


@app.get("/update_board_state")
def update_board_state(board_state_string: str, move: str, special: str):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    return ChessboardStateLogic().board_to_string(ChessboardStateLogic().update_board_state(board_state, move, special))


@app.get("/get_orientation")
def determine_orientation(self, initial_board_state_string: str):
    initial_board_state = self.string_to_board(initial_board_state_string)
    return ChessboardStateLogic().determine_orientation(initial_board_state)


@app.get("/get_rotated_board")
def rotate_board_to_bottom(self, board_state_string: str, orientation: str):
    board_state = self.string_to_board(board_state_string)
    return self.board_to_string(ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation))


@app.get("/check_special_move")
def is_special_move(self, fen: str, move: str):
    return ChessboardLogic().is_special_move(fen, move)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
