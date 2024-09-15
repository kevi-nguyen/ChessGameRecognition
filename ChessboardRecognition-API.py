from ChessboardRecognition import ChessboardRecognition
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()


class ChessboardRecognitionAPI:

    @app.get("/get_dst_points")
    def get_frame_resolution(self, frame):
        return ChessboardRecognition().get_frame_resolution(frame)

    @app.get("/get_board_state")
    def get_board_state(self, image, src_points, dst_points):
        image = ChessboardRecognition().transform_image(image, src_points, dst_points)
        return self.board_to_string(ChessboardRecognition().get_board_state(image))

    @app.get("/get_src_points")
    def find_chessboard_corners_green(self, frame):
        return ChessboardRecognition().find_chessboard_corners_green(frame)

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
