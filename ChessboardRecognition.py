import cv2
import numpy as np


class ChessboardRecognition:

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

    def detect_chessboard_harris(self, image):

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to float32
        gray = np.float32(gray)

        # Apply the Harris corner detection
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate the result to mark the corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, marking the corners in red
        image[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Display the result
        cv2.imshow('Harris Corners', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_chessboard_corners(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(blurred, (7, 7), None)

        if ret:
            # Refine corner locations
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Extract the coordinates of the four outer corners
            corners = corners.reshape(-1, 2)
            top_left = corners[0]
            top_right = corners[6]
            bottom_left = corners[-7]
            bottom_right = corners[-1]
            print(
                f"Top left: {top_left}, Top right: {top_right}, Bottom left: {bottom_left}, Bottom right: {bottom_right}")

            temp = self.find_extended_chessboard_corners([tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)])
            #img = cv2.drawChessboardCorners(frame, (7, 7), corners, ret)
            #cv2.imwrite('chessboard_with_corners.png', img)
            #cv2.imshow('Chessboard', img)
            #cv2.waitKey(0)
            return temp
        else:
            raise ValueError("Could not find a chessboard with the specified pattern")

    def find_extended_chessboard_corners(self, corners):
        # Extract the coordinates of the four corners
        top_left, top_right, bottom_left, bottom_right = corners

        # Calculate the size of one square
        square_size_x = (top_right[0] - top_left[0]) / 6
        square_size_y = (bottom_left[1] - top_left[1]) / 6

        # Calculate the new corners one square further
        extended_top_left = (top_left[0] - square_size_x, top_left[1] - square_size_y)
        extended_top_right = (top_right[0] + square_size_x, top_right[1] - square_size_y)
        extended_bottom_left = (bottom_left[0] - square_size_x, bottom_left[1] + square_size_y)
        extended_bottom_right = (bottom_right[0] + square_size_x, bottom_right[1] + square_size_y)

        print(
            f"Top left: {extended_top_left}, Top right: {extended_top_right}, Bottom left: {extended_bottom_left}, Bottom right: {extended_bottom_right}")

        return [extended_top_left, extended_top_right, extended_bottom_left, extended_bottom_right]

    def find_chessboard_corners_harris(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Convert the grayscale image to float32
        gray = np.float32(blurred)

        # Apply the Harris corner detection
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate the result to mark the corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, marking the corners
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Find the coordinates of the corners
        corners = np.argwhere(dst > 0.01 * dst.max())
        corners = sorted(corners, key=lambda x: (x[1], x[0]))

        if len(corners) >= 4:
            # Extract the coordinates of the four outer corners
            top_left = tuple(corners[0])
            top_right = tuple(corners[7])
            bottom_left = tuple(corners[-8])
            bottom_right = tuple(corners[-1])
            print(
                f"Top left: {top_left}, Top right: {top_right}, Bottom left: {bottom_left}, Bottom right: {bottom_right}")

            # Draw the corners on the frame for visualization
            for corner in [top_left, top_right, bottom_left, bottom_right]:
                cv2.circle(frame, corner, 5, (0, 255, 0), -1)

            # Display the frame with corners
            cv2.imshow('Chessboard Corners', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return [top_left, top_right, bottom_left, bottom_right]
        else:
            raise ValueError("Could not find a chessboard with the specified pattern")

    def find_chessboard_corners_shi_tomasi(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(blurred, maxCorners=81, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)

        if len(corners) >= 4:
            # Extract the coordinates of the four outer corners
            corners = sorted(corners, key=lambda x: (x[0][1], x[0][0]))
            top_left = tuple(corners[0][0])
            top_right = tuple(corners[7][0])
            bottom_left = tuple(corners[-8][0])
            bottom_right = tuple(corners[-1][0])
            print(
                f"Top left: {top_left}, Top right: {top_right}, Bottom left: {bottom_left}, Bottom right: {bottom_right}")

            # Draw the corners on the frame for visualization
            for corner in [top_left, top_right, bottom_left, bottom_right]:
                cv2.circle(frame, corner, 5, (0, 255, 0), -1)

            # Display the frame with corners
            cv2.imshow('Chessboard Corners', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return [top_left, top_right, bottom_left, bottom_right]
        else:
            raise ValueError("Could not find a chessboard with the specified pattern")

    def transform_image(self, image, src_points, dst_points):
        """
        Transforms the image using the given source and destination points.

        Parameters:
        - image: The input image to be transformed.
        - src_points: A list of four points (x, y) in the source image.
        - dst_points: A list of four points (x, y) in the destination image.

        Returns:
        - The transformed image.
        """
        # Convert points to numpy arrays
        src_points = np.array(src_points, dtype='float32')
        dst_points = np.array(dst_points, dtype='float32')

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Get the size of the destination image
        width = int(max(dst_points[:, 0]) - min(dst_points[:, 0]))
        height = int(max(dst_points[:, 1]) - min(dst_points[:, 1]))

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(image, matrix, (width, height))

        return transformed_image

    def get_frame_resolution(self, frame):
        # Get the resolution of the frame
        height, width, _ = frame.shape
        # Define the destination points based on the resolution
        dst_points = [
            (0, 0),  # Top-left corner
            (width, 0),  # Top-right corner
            (0, height),  # Bottom-left corner
            (width, height)  # Bottom-right corner
        ]
        print(f"Destination points: {dst_points}")
        return dst_points

    def get_board_state(self, image, width, height):
        # Apply a Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert the blurred image to the HSV color space
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Apply histogram equalization to the V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 150])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Define color ranges for blue
        lower_blue = np.array([110, 150, 150])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Combine the masks for red and blue
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image, image, mask=combined_mask)

        # Calculate the size of each cell in pixels
        cell_size = (height // 8, width // 8)

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

                threshold = 1000

                # Get the color of the piece in the cell
                if cv2.countNonZero(blue_mask[y1:y2, x1:x2]) > threshold:
                    board_state[i][j] = 'blue'
                elif cv2.countNonZero(red_mask[y1:y2, x1:x2]) > threshold:
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
                if prev_state[i][j] is not None and curr_state[i][j] is None:
                    if start is None:
                        start = self.cell_to_chess_notation((i, j))
                        start_colour = 'red' if prev_state[i][j] == 'red' else 'blue'
                    else:
                        second_start = self.cell_to_chess_notation((i, j))
                        second_start_colour = 'red' if prev_state[i][j] == 'red' else 'blue'
            if start is not None and second_start is not None:
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
            if start and end:
                # Castling
                if second_start:
                    # Check for blue castling (blue is on the 7th rank)
                    if prev_state[7][4] == 'blue' and prev_state[7][7] == 'blue' and curr_state[7][4] is None and \
                            curr_state[7][7] is None and curr_state[7][6] == 'blue' and curr_state[7][5] == 'blue':
                        return 'e1g1', 'blue'  # Kingside castling
                    elif prev_state[7][4] == 'blue' and prev_state[7][0] == 'blue' and curr_state[7][4] is None and \
                            curr_state[7][0] is None and curr_state[7][2] == 'blue' and curr_state[7][3] == 'blue':
                        return 'e1c1', 'blue'  # Queenside castling

                    # Check for red castling (red is on the 0th rank)
                    if prev_state[0][4] == 'red' and prev_state[0][7] == 'red' and curr_state[0][4] is None and \
                            curr_state[0][7] is None and curr_state[0][6] == 'red' and curr_state[0][5] == 'red':
                        return 'e8g8', 'red'  # Kingside castling
                    elif prev_state[0][4] == 'red' and prev_state[0][0] == 'red' and curr_state[0][4] is None and \
                            curr_state[0][0] is None and curr_state[0][2] == 'red' and curr_state[0][3] == 'red':
                        return 'e8c8', 'red'  # Queenside castling

                # En passant
                if start_colour == end_colour:
                    correct_start = start
                else:
                    correct_start = second_start

                return correct_start + end, prev_state[ord(correct_start[0]) - ord('a')][8 - int(correct_start[1])]

            return start + end, prev_state[ord(start[0]) - ord('a')][8 - int(start[1])]

        return None, None

    def update_board_state(self, board_state, move, special):
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

        if special:
            # Handle castling
            if piece == 'blue' and start_pos == (7, 4) and end_pos in [(7, 6), (7, 2)]:
                # Kingside castling
                if end_pos == (7, 6):
                    board_state[7][4] = None
                    board_state[7][6] = 'blue'
                    board_state[7][7] = None
                    board_state[7][5] = 'blue'
                # Queenside castling
                elif end_pos == (7, 2):
                    board_state[7][4] = None
                    board_state[7][2] = 'blue'
                    board_state[7][0] = None
                    board_state[7][3] = 'blue'
            elif piece == 'red' and start_pos == (0, 4) and end_pos in [(0, 6), (0, 2)]:
                # Kingside castling
                if end_pos == (0, 6):
                    board_state[0][4] = None
                    board_state[0][6] = 'red'
                    board_state[0][7] = None
                    board_state[0][5] = 'red'
                # Queenside castling
                elif end_pos == (0, 2):
                    board_state[0][4] = None
                    board_state[0][2] = 'red'
                    board_state[0][0] = None
                    board_state[0][3] = 'red'

            # Handle en passant
            if piece == 'blue' and start_row == 3 and end_row == 2 and board_state[end_row][
                end_col] is None and start_col != end_col:
                board_state[start_row][start_col] = None
                board_state[end_row][end_col] = 'blue'
                board_state[start_row][end_col] = None
            elif piece == 'red' and start_row == 4 and end_row == 5 and board_state[end_row][
                end_col] is None and start_col != end_col:
                board_state[start_row][start_col] = None
                board_state[end_row][end_col] = 'red'
                board_state[start_row][end_col] = None
        else:
            # Normal move
            board_state[end_row][end_col] = piece
            board_state[start_row][start_col] = None

        return board_state


