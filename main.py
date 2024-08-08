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

    # Set a fixed starting board state
    starting_board_state = [row[:] for row in initial_board_state]

    # Flag to indicate if the game has just started
    is_game_start = True

    # Create a Chessboard object
    chessboard = Chessboard()

    for row in initial_board_state:
        print(row)

    print(chessboard.board)

    # Wait for a user signal to find the moved piece
    while True:
        # Check if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Get the current board state
            current_board_state = chessboard_recognition.get_board_state(transformed_image)

            #current_board_state = simulate_move(current_board_state, ((6, 4), (4, 4)))

            # Find the moved piece
            move, piece = chessboard_recognition.find_moved_piece(initial_board_state, current_board_state)

            # If no move was found
            if move is None:
                # Check if it is the start of the game
                if is_game_start and starting_board_state == current_board_state:
                    print("No move found, starting as white.")

                    # Get the FEN string of the current board state
                    fen_string = chessboard.fen()
                    print(f"FEN: {fen_string}")

                    # Send the initial FEN string to the microservice and get the best move
                    response = requests.get('http://localhost:8080/get_move', params={'fen': fen_string})
                    if response.status_code == 200:
                        best_move = response.json().get('move')
                        print(f"Best move: {best_move}")

                        # Robot's move
                        chessboard.move_piece(best_move)
                        #chessboard.display_move(best_move, piece)

                        # Update the initial board state
                        initial_board_state = chessboard_recognition.update_board_state(initial_board_state, chessboard_recognition.chess_to_cell_notation(best_move))

                        # Set the game start flag to False
                        is_game_start = False

                    else:
                        print(f"Failed to get the best move: {response.status_code}")
                else:
                    print("No move found")
            else:
                # If a move was found, make the move on the chessboard
                chessboard.move_piece(move)
                #chessboard.display_move(move, piece)

                # Update the initial board state
                initial_board_state = current_board_state

                # Get the FEN string of the current board state
                fen_string = chessboard.fen()
                print(f"FEN: {fen_string}")

                # Send the FEN string to the microservice and get the best move
                response = requests.get('http://localhost:8080/get_move', params={'fen': fen_string})
                if response.status_code == 200:
                    best_move = response.json().get('move')
                    print(f"Best move: {best_move}")

                    # Robot's move
                    chessboard.move_piece(best_move)
                    #chessboard.display_move(best_move, piece)

                    # Update the initial board state
                    initial_board_state = chessboard_recognition.update_board_state(initial_board_state, chessboard_recognition.chess_to_cell_notation(best_move))

                else:
                    print(f"Failed to get the best move: {response.status_code}")

            for row in initial_board_state:
                print(row)

            print(chessboard.board)