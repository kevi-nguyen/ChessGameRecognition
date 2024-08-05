from ChessboardRecognition import ChessboardRecognition
from Chessboard import Chessboard
import cv2
import requests

if __name__ == '__main__':
    # Create a ChessboardRecognition object
    chessboard_recognition = ChessboardRecognition('images/IMG_4764.jpeg')

    # Transform the image
    transformed_image = chessboard_recognition.transform_image()

    # Get the initial board state
    initial_board_state = chessboard_recognition.get_board_state(transformed_image)

    # Create a Chessboard object
    chessboard = Chessboard()

    for row in initial_board_state:
        print(row)

    print(chessboard.board)

    # Get the FEN string of the current board state
    fen_string = chessboard.fen()
    print(f"FEN: {fen_string}")

    # Wait for a user signal to find the moved piece
    while True:
        # Check if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Display the transformed image to keep the window active
            cv2.imshow('Transformed Image', transformed_image)

            # Get the current board state
            current_board_state = chessboard_recognition.get_board_state(transformed_image)

            # Find the moved piece
            move, piece = chessboard_recognition.find_moved_piece(initial_board_state)

            # If a move was found, make the move on the chessboard
            if move is not None:

                # Check players move
                chessboard.move_piece(move)
                chessboard.display_move(move, piece)

                # Update the initial board state
                initial_board_state = current_board_state

                # Send the FEN string to the microservice and get the best move
                response = requests.get('http://localhost:8080/get_move', params={'fen': fen_string})
                if response.status_code == 200:
                    best_move = response.json().get('move')
                    print(f"Best move: {best_move}")

                    # Cobots move
                    chessboard.move_piece(best_move)
                    chessboard.display_move(best_move, piece)

                else:
                    print(f"Failed to get the best move: {response.status_code}")

                for row in initial_board_state:
                    print(row)

                print(chessboard.board)

            else:
                print("No move found")