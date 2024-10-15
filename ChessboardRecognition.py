import cv2
import numpy as np

from ColorDetector import ColorDetector
from Retinex import Retinex


class ChessboardRecognition:

    def transform_image(self, image, src_points, dst_points):
        src_points = np.array(src_points, dtype='float32')
        dst_points = np.array(dst_points, dtype='float32')

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        width = int(max(dst_points[:, 0]) - min(dst_points[:, 0]))
        height = int(max(dst_points[:, 1]) - min(dst_points[:, 1]))

        transformed_image = cv2.warpPerspective(image, matrix, (width, height))

        return transformed_image

    def get_frame_resolution(self, frame):
        height, width, _ = frame.shape

        dst_points = [
            (0, 0),
            (width, 0),
            (0, height),
            (width, height)
        ]
        return dst_points

    def get_board_state(self, image):
        width, height, _ = image.shape

        msrcp = Retinex().msrcp(image)

        # Process the image to get masks for blue and red colors
        blue_mask = ColorDetector().process_image(msrcp, 'blue')
        red_mask = ColorDetector().process_image(msrcp, 'red')

        # Combine the masks for red and blue
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Bitwise-AND mask and original MSRCP image
        res = cv2.bitwise_and(msrcp, msrcp, mask=combined_mask)

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

                threshold = 500

                # Get the color of the piece in the cell
                if cv2.countNonZero(blue_mask[y1:y2, x1:x2]) > threshold:
                    board_state[i][j] = 'blue'
                elif cv2.countNonZero(red_mask[y1:y2, x1:x2]) > threshold:
                    board_state[i][j] = 'red'

        return board_state

    def preprocess_image_for_green(self, image):
        msrcp = Retinex().msrcp(image)

        green_mask = ColorDetector().process_image(msrcp, 'green')

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the green mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, green_mask, msrcp

    def find_chessboard_corners_green(self, frame):
        contours, green_mask, msrcp = self.preprocess_image_for_green(frame)

        green_corners = []

        # Filter contours to find the four green markers
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    green_corners.append((cX, cY))

        if len(green_corners) < 4:
            raise ValueError("Could not find at least 4 green markers.")

        # Sort corners by x-coordinate to separate left and right
        green_corners = sorted(green_corners, key=lambda x: x[0])

        # Separate left and right corners
        left_corners = green_corners[:2]
        right_corners = green_corners[2:]

        # Sort left corners by y-coordinate to identify top-left and bottom-left
        left_corners = sorted(left_corners, key=lambda x: x[1])
        top_left = left_corners[0]
        bottom_left = left_corners[1]

        # Sort right corners by y-coordinate to identify top-right and bottom-right
        right_corners = sorted(right_corners, key=lambda x: x[1])
        top_right = right_corners[0]
        bottom_right = right_corners[1]

        for point in [top_left, top_right, bottom_left, bottom_right]:
            cv2.circle(frame, point, 10, (0, 255, 0), -1)

        return [top_left, top_right, bottom_left, bottom_right]
