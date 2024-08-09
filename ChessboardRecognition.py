import cv2
import numpy as np


class ChessboardRecognition:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def cell_to_chess_notation(self, cell):
        # Convert the cell coordinates to chess notation
        return chr(cell[1] + ord('a')) + str(8 - cell[0])

    def chess_to_cell_notation(self, move_str):
        """
        Converts a move string to a tuple.

        Parameters:
        - move_str: String representing the move, e.g., 'e2e4'.

        Returns:
        - Tuple representing the move in the format ((start_row, start_col), (end_row, end_col)).
        """
        start_col = ord(move_str[0]) - ord('a')
        start_row = 8 - int(move_str[1])
        end_col = ord(move_str[2]) - ord('a')
        end_row = 8 - int(move_str[3])
        return ((start_row, start_col), (end_row, end_col))

    def transform_image(self):
        # Define the pixel coordinates of the four corners of the chessboard in the distorted image
        src_points = np.float32([[375, 760], [2530, 726], [325, 2957], [2625, 2936]])

        # Define the pixel coordinates of the four corners of the chessboard in the corrected image
        dst_points = np.float32([[0, 0], [3024, 0], [0, 4032], [3024, 4032]])

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation to the image
        corrected_image = cv2.warpPerspective(self.image, matrix, (3024, 4032))
        return corrected_image

    def get_board_state(self, image):
        # Apply a Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert the blurred image to the HSV color space
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Apply histogram equalization to the V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Define a wider color range for red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Define color ranges for blue
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Combine the masks for red and blue
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image, image, mask=combined_mask)

        # Calculate the size of each cell in pixels
        cell_size = (3024 // 8, 4032 // 8)

        # Define an empty 2D array to hold the board state
        board_state = [[None for _ in range(8)] for _ in range(8)]

        # Iterate over the cells of the chessboard
        for i in range(8):
            for j in range(8):
                # Calculate the coordinates of the cell
                x1 = j * cell_size[0]
                y1 = i * cell_size[1]
                x2 = x1 + cell_size[0]
                y2 = y1 + cell_size[1]

                # Draw lines for each cell on the image with the combined mask
                cv2.line(res, (x1, y1), (x2, y1), (0, 255, 0), 2)  # Top line
                cv2.line(res, (x1, y1), (x1, y2), (0, 255, 0), 2)  # Left line
                cv2.line(res, (x2, y1), (x2, y2), (0, 255, 0), 2)  # Right line
                cv2.line(res, (x1, y2), (x2, y2), (0, 255, 0), 2)  # Bottom line

                # Get the color of the piece in the cell
                if cv2.countNonZero(blue_mask[y1:y2, x1:x2]) > 50:
                    board_state[i][j] = 'blue'
                elif cv2.countNonZero(red_mask[y1:y2, x1:x2]) > 50:
                    board_state[i][j] = 'red'

        # Display the image with cell lines
        cv2.imshow('Image with cell lines', res)

        return board_state

    def find_moved_piece(self, prev_state, curr_state):
        """
        Finds the moved piece by comparing the previous and current board states.

        Parameters:
        - prev_state: 2D list representing the previous state of the board.
        - curr_state: 2D list representing the current state of the board.

        Returns:
        - Tuple containing the move in UCI format and the piece that was moved.
        """

        # Initialize the start and end positions
        start = end = second_start = None

        # Variables for the colours of the start and end positions to handle en passant
        start_colour = second_start_colour = end_colour = None

        # Iterate over the cells of the chessboard to find the start position
        for i in range(len(prev_state)):
            for j in range(len(prev_state[i])):
                # If the cell contains a piece in the old state but is empty in the new state, it is the start position
                if prev_state[i][j] is not None and (
                        curr_state[i][j] is None or curr_state[i][j] != prev_state[i][j]):
                    if start is None:
                        start = self.cell_to_chess_notation((i, j))
                        start_colour = 'red' if prev_state[i][j] == 'red' else 'blue'
                    else:
                        second_start = self.cell_to_chess_notation((i, j))
                        second_start_colour = 'red' if prev_state[i][j] == 'red' else 'blue'
                    break
            if start is not None:
                break

        # Iterate over the cells of the chessboard to find the end position
        for i in range(len(curr_state)):
            for j in range(len(curr_state[i])):
                # If the cell contains a piece in the new state but is empty or contains a different piece in the old state, it is the end position
                if curr_state[i][j] is not None and (
                        prev_state[i][j] is None or curr_state[i][j] != prev_state[i][j]):
                    end = self.cell_to_chess_notation((i, j))
                    end_colour = 'red' if curr_state[i][j] == 'red' else 'blue'
                    break
            if end is not None:
                break

        # Handle special cases
        if start and end:
            # Castling
            if second_start:
                if prev_state[7][4] == 'red' and prev_state[7][7] == 'red' and curr_state[7][4] is None and \
                        curr_state[7][7] is None:
                    if curr_state[7][6] == 'red' and curr_state[7][5] == 'red':
                        return 'e1g1', 'red'  # Kingside castling
                    elif curr_state[7][2] == 'red' and curr_state[7][3] == 'red':
                        return 'e1c1', 'red'  # Queenside castling
                if prev_state[0][4] == 'blue' and prev_state[0][7] == 'blue' and curr_state[0][4] is None and \
                        curr_state[0][7] is None:
                    if curr_state[0][6] == 'blue' and curr_state[0][5] == 'blue':
                        return 'e8g8', 'blue'  # Kingside castling
                    elif curr_state[0][2] == 'blue' and curr_state[0][3] == 'blue':
                        return 'e8c8', 'blue'  # Queenside castling

                # En passant
                if start_colour == end_colour:
                    correct_start = start
                else:
                    correct_start = second_start

                return correct_start + end, prev_state[ord(correct_start[0]) - ord('a')][8 - int(correct_start[1])]

            return start + end, prev_state[ord(start[0]) - ord('a')][8 - int(start[1])]

        return None, None

    def update_board_state(self, board_state, move):
        """
        Updates the board state manually after a move.

        Parameters:
        - board_state: 2D list representing the current state of the board.
        - move: Tuple representing the move in the format ((start_row, start_col), (end_row, end_col)).

        Returns:
        - Updated board state after the move.
        """
        start_pos, end_pos = move
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        # Get the piece at the start position
        piece = board_state[start_row][start_col]

        # Move the piece to the end position
        board_state[end_row][end_col] = piece
        board_state[start_row][start_col] = None

        return board_state

    def get_board_state_from_stream(self):
        while True:
            # Read a frame from the video stream
            ret, frame = self.video_stream.read()

            # If the frame was read successfully, process it
            if ret:
                # Transform the image
                transformed_image = self.transform_image(frame)

                # Get the board state
                board_state = self.get_board_state(transformed_image)

                # Output the board state
                for row in board_state:
                    print(row)

                # Display the transformed image
                cv2.imshow('Transformed Image', transformed_image)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    def release_stream(self):
        # When everything done, release the video capture object
        self.video_stream.release()

        # Closes all the frames
        cv2.destroyAllWindows()
