from fastapi import FastAPI, HTTPException
import uvicorn
from ChessboardLogic import ChessboardLogic
from ChessboardStateLogic import ChessboardStateLogic

app = FastAPI()

class ChessboardAPI:

    @app.get("/update_chessboard")
    def move_piece(self, fen, move):
        return ChessboardLogic().move_piece(fen, move)

    @app.get("/check_game_end")
    def is_checkmate(self, fen):
        return ChessboardLogic().is_checkmate(fen)

    @app.get("/get_move")
    def find_moved_piece(self, prev_state_string, curr_state_string):
        prev_state = self.string_to_board(prev_state_string)
        curr_state = self.string_to_board(curr_state_string)
        return ChessboardStateLogic().find_moved_piece(prev_state, curr_state)

    @app.get("/update_board_state")
    def update_board_state(self, board_state_string, move, special):
        board_state = self.string_to_board(board_state_string)
        return self.board_to_string(ChessboardStateLogic().update_board_state(board_state, move, special))

    @app.get("/get_orientation")
    def determine_orientation(self, initial_board_state_string):
        initial_board_state = self.string_to_board(initial_board_state_string)
        return ChessboardStateLogic().determine_orientation(initial_board_state)

    @app.get("/get_rotated_board")
    def rotate_board_to_bottom(self, board_state_string, orientation):
        board_state = self.string_to_board(board_state_string)
        return self.board_to_string(ChessboardStateLogic().rotate_board_to_bottom(board_state, orientation))

    @app.get("/check_special_move")
    def is_special_move(self, fen, move):
        return ChessboardLogic().is_special_move(fen, move)


    def board_to_string(self, board_state):
        """
        Converts the board state (2D list) to a string format for CPEE.
        - None -> '0'
        - 'blue' -> '1'
        - 'red' -> '2'

        Parameters:
        - board_state: 2D list representing the current state of the board.

        Returns:
        - A string representation of the board.
        """
        return ''.join(
            [''.join([str(0) if cell is None else str(1) if cell == 'blue' else str(2) for cell in row]) for row in
             board_state])

    def string_to_board(self, board_string):
        """
        Converts a string representation back to a board state (2D list).

        Parameters:
        - board_string: String representation of the board state.

        Returns:
        - 2D list representing the current state of the board.
        """
        board_size = 8
        return [[None if c == '0' else 'blue' if c == '1' else 'red' for c in board_string[i:i + board_size]] for i in
                range(0, len(board_string), board_size)]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)