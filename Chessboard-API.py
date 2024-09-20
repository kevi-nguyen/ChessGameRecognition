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
def update_board_state(board_state_string: str, move: str, special: bool):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    return ChessboardStateLogic().board_to_string(ChessboardStateLogic().update_board_state(board_state, move, special))


@app.get("/get_orientation")
def determine_orientation(initial_board_state_string: str):
    initial_board_state = ChessboardStateLogic().string_to_board(initial_board_state_string)
    return ChessboardStateLogic().determine_orientation(initial_board_state)


@app.get("/get_rotated_board")
def rotate_board_to_bottom(board_state_string: str, orientation: str):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    return ChessboardStateLogic().board_to_string(ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation))


@app.get("/check_special_move")
def is_special_move(fen: str, move: str):
    return ChessboardLogic().is_special_move(fen, move)

@app.get("/get_cobot_move")
def coordinates_to_cobot_move(board_state_string: str, move: str, special: bool, orientation: str):
    board_state = ChessboardStateLogic().string_to_board(board_state_string)
    move = ChessboardStateLogic().chess_to_cell_notation(move)
    start_pos, end_pos = move
    start_row, start_col = start_pos
    end_row, end_col = end_pos

    piece = board_state[start_row][start_col]

    if special:
        # Handle castling
        if piece == 'blue' and start_pos == (7, 4) and end_pos in [(7, 6), (7, 2)]:
            # Kingside castling
            if end_pos == (7, 6):
                move1 = positions_to_string((7, 4), (7, 6), orientation)
                move2 = positions_to_string((7, 7), (7, 5), orientation)
                return move1 + move2
            # Queenside castling
            elif end_pos == (7, 2):
                move1 = positions_to_string((7, 4), (7, 2), orientation)
                move2 = positions_to_string((7, 0), (7, 3), orientation)
                return move1 + move2
        elif piece == 'red' and start_pos == (0, 4) and end_pos in [(0, 6), (0, 2)]:
            # Kingside castling
            if end_pos == (0, 6):
                move1 = positions_to_string((0, 4), (0, 6), orientation)
                move2 = positions_to_string((0, 7), (0, 5), orientation)
                return move1 + move2
            # Queenside castling
            elif end_pos == (0, 2):
                move1 = positions_to_string((0, 4), (0, 2), orientation)
                move2 = positions_to_string((0, 0), (0, 3), orientation)
                return move1 + move2
        # Handle en passant
        if piece == 'blue' and start_row == 3 and end_row == 2 and board_state[end_row][
            end_col] is None and start_col != end_col:
            move1 = positions_to_string((start_row, start_col), (end_row, end_col), orientation)
            move2 = positions_to_string((start_row, end_col), (9, 9), orientation)
            return move1 + move2
        elif piece == 'red' and start_row == 4 and end_row == 5 and board_state[end_row][
            end_col] is None and start_col != end_col:
            move1 = positions_to_string((start_row, start_col), (end_row, end_col), orientation)
            move2 = positions_to_string((start_row, end_col), (9, 9), orientation)
            return move1 + move2
    else:
        move1 = 'XXXX'

        if board_state[end_row][end_col] is not None:
            move1 = positions_to_string((end_row, end_col), (9, 9), orientation)

        move2 = positions_to_string((start_row, start_col), (end_row, end_col), orientation)

        return move1 + move2


def positions_to_string(start, end, orientation):
    """
    Converts the start and end positions into a string after transforming based on the board orientation.

    Parameters:
    - start: A tuple (start_i, start_j) representing the start position.
    - end: A tuple (end_i, end_j) representing the end position.
    - orientation: The current orientation of the board ('bottom', 'left', 'top', 'right').

    Returns:
    - A string in the format 'start_i start_j end_i end_j'.
    """
    # Transform coordinates based on the orientation
    start_transformed = ChessboardStateLogic().transform_coordinates(start[0], start[1], orientation)
    end_transformed = ChessboardStateLogic().transform_coordinates(end[0], end[1], orientation)

    start_i, start_j = start_transformed
    end_i, end_j = end_transformed

    # Convert the integers to strings and concatenate them
    return f"{start_i}{start_j}{end_i}{end_j}"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
