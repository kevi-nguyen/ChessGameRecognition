import copy

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

            # Get user input for the move
            user_input = input("Enter your move (e.g., 'd2d4') or 'no' to skip: ").strip()

            if user_input.lower() == 'no':
                print("No move made.")
                continue

            # Create a copy of the initial board state for the current move
            current_board_state = copy.deepcopy(initial_board_state)

            special = chessboard.is_special_move(user_input)

            chessboard_recognition.update_board_state(current_board_state,
                                                      chessboard_recognition.chess_to_cell_notation(user_input), special)

            for row in current_board_state:
                print(row)

            for row in initial_board_state:
                print(row)

            # Find the moved piece
            move, piece = chessboard_recognition.find_moved_piece(initial_board_state, current_board_state)
            print(f"Move: {move}, Piece: {piece}")

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
                        special = chessboard.is_special_move(best_move)
                        chessboard.move_piece(best_move)
                        #chessboard.display_move(best_move, piece)

                        # Update the initial board state
                        initial_board_state = chessboard_recognition.update_board_state(initial_board_state, chessboard_recognition.chess_to_cell_notation(best_move), special)

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

                # Create a deep copy of the current board state to update the initial board state
                initial_board_state = copy.deepcopy(current_board_state)

                # Get the FEN string of the current board state
                fen_string = chessboard.fen()
                print(f"FEN: {fen_string}")

                # Send the FEN string to the microservice and get the best move
                response = requests.get('http://localhost:8080/get_move', params={'fen': fen_string})
                if response.status_code == 200:
                    best_move = response.json().get('move')
                    print(f"Best move: {best_move}")

                    # Robot's move
                    special = chessboard.is_special_move(best_move)
                    chessboard.move_piece(best_move)
                    #chessboard.display_move(best_move, piece)

                    # Update the initial board state
                    initial_board_state = chessboard_recognition.update_board_state(initial_board_state, chessboard_recognition.chess_to_cell_notation(best_move), special)

                else:
                    print(f"Failed to get the best move: {response.status_code}")

            for row in initial_board_state:
                print(row)

            print(chessboard.board)