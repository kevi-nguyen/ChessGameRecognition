import copy
from ChessboardRecognition import ChessboardRecognition
from Chessboard import Chessboard
import cv2
import requests
from VideoCapture import VideoCapture
from ChessboardStateLogic import ChessboardStateLogic

if __name__ == '__main__':
    # Create a ChessboardRecognition object
    chessboard_recognition = ChessboardRecognition()

    # Create a ChessboardStateLogic object
    chessboard_state_logic = ChessboardStateLogic()

    # Create a Chessboard object
    chessboard = Chessboard()

    # Create a VideoCapture object
    cap = VideoCapture()

    # Get board position
    frame = cap.get_snapshot()
    height, width, _ = frame.shape
    cap.save_snapshot('initial_snapshot.png')
    src_points = chessboard_recognition.find_chessboard_corners(frame)
    dst_points = chessboard_recognition.get_frame_resolution(frame)

    # Wait for user to set up the chess pieces and press 's' to signal readiness
    print("Set up the chess pieces and press 's' to signal readiness.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Get the initial board state
    frame = cap.get_snapshot()
    cap.save_snapshot('current_snapshot.png')
    transformed_image = chessboard_recognition.transform_image(frame, src_points, dst_points)
    initial_board_state = chessboard_recognition.get_board_state(transformed_image, height, width)

    # Determine the orientation of the board
    orientation = chessboard_state_logic.determine_orientation(initial_board_state)

    # Rotate the board to have the blue pieces at the bottom
    initial_board_state = chessboard_state_logic.rotate_board_to_bottom(initial_board_state, orientation)

    # Set a fixed starting board state
    starting_board_state = [row[:] for row in initial_board_state]

    # Flag to indicate if the game has just started
    is_game_start = True

    for row in initial_board_state:
        print(row)

    print(chessboard.board)

    # Wait for a user signal to find the moved piece
    while True:
        # Check if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Create a copy of the initial board state for the current move
            # current_board_state = copy.deepcopy(initial_board_state)

            # Get user input for the move
            # user_input = input("Enter your move (e.g., 'd2d4') or 'no' to skip: ").strip()

            # Create a copy of the initial board state for the current move
            # current_board_state = copy.deepcopy(initial_board_state)

            #    special = chessboard.is_special_move(user_input)

            #   chessboard_recognition.update_board_state(current_board_state,
            #                                            chessboard_recognition.chess_to_cell_notation(user_input),
            #                                           special)

            frame = cap.get_snapshot()
            cap.save_snapshot('current_snapshot.png')
            transformed_image = chessboard_recognition.transform_image(frame, src_points, dst_points)
            current_board_state = chessboard_recognition.get_board_state(transformed_image, height, width)

            # Rotate the board to have the blue pieces at the bottom
            current_board_state = chessboard_state_logic.rotate_board_to_bottom(current_board_state, orientation)

            # Find the moved piece
            move, piece = chessboard_state_logic.find_moved_piece(initial_board_state, current_board_state)
            print(f"Move: {move}, Piece: {piece}")

            # If no move was found
            if move is None:
                # Check if it is the start of the game
                if is_game_start and starting_board_state == current_board_state:
                    print("No move found, starting as white.")

                    # Get the FEN string of the current board state
                    fen_string = chessboard.fen()

                    # Send the initial FEN string to the microservice and get the best move
                    response = requests.get('http://localhost:8080/get_move', params={'fen': fen_string})
                    if response.status_code == 200:
                        best_move = response.json().get('move')
                        print(f"Best move: {best_move}")

                        # Robot's move
                        special = chessboard.is_special_move(best_move)
                        chessboard.move_piece(best_move)
                        # chessboard.display_move(best_move, piece)

                        # Update the initial board state
                        initial_board_state = chessboard_state_logic.update_board_state(initial_board_state,
                                                                                        chessboard_state_logic.chess_to_cell_notation(
                                                                                            best_move), special)

                        # Set the game start flag to False
                        is_game_start = False

                    else:
                        print(f"Failed to get the best move: {response.status_code}")
                else:
                    print("No move found")
            else:
                # If a move was found, make the move on the chessboard
                if not chessboard.move_piece(move):
                    print("Illegal move. Please try again.")
                    continue
                # chessboard.display_move(move, piece)

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
                    # chessboard.display_move(best_move, piece)

                    # Update the initial board state
                    initial_board_state = chessboard_state_logic.update_board_state(initial_board_state,
                                                                                    chessboard_state_logic.chess_to_cell_notation(
                                                                                        best_move), special)

                else:
                    print(f"Failed to get the best move: {response.status_code}")

            for row in initial_board_state:
                print(row)

            print(chessboard.board)
