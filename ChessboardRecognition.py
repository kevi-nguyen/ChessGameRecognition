import cv2
import numpy as np
from Retinex import Retinex
from ColorDetector import ColorDetector


class ChessboardRecognition:

    def preprocess_image(self, image):
        cv2.imshow('Original Image', image)

        msrcp = Retinex().msrcp(image)
        cv2.imshow('MSRCP Image', msrcp)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(msrcp, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Gray Image', gray)

        # Apply adaptive histogram equalization to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        cv2.imshow('Enhanced Gray Image', enhanced_gray)

        # gamma = 0.5
        # adjusted = np.array(255 * (enhanced_gray / 255) ** gamma, dtype='uint8')

        # cv2.imshow('Adjusted Image', adjusted)

        thresholded = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 15, 10)

        cv2.imshow('Threshold Image', thresholded)

        kernel = np.ones((3, 3), np.uint8)
        cleaned_thresh = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Cleaned Threshold Image', cleaned_thresh)

        cv2.waitKey(0)

        return enhanced_gray

    def find_chessboard_corners(self, frame):

        processed_image = self.preprocess_image(frame)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(processed_image, (7, 7), None)

        if ret:
            # Refine corner locations
            corners = cv2.cornerSubPix(processed_image, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Extract the coordinates of the four outer corners
            corners = corners.reshape(-1, 2)
            top_left = corners[0]
            top_right = corners[6]
            bottom_left = corners[-7]
            bottom_right = corners[-1]
            print(
                f"Top left: {top_left}, Top right: {top_right}, Bottom left: {bottom_left}, Bottom right: {bottom_right}")

            temp = self.find_extended_chessboard_corners(
                [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)])
            img = cv2.drawChessboardCorners(frame, (7, 7), corners, ret)
            cv2.imwrite('chessboard_with_corners.png', img)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(0)
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
        """
        Determine the board state and detect colors dynamically based on HSV ranges.

        Parameters:
        - image: Input image.
        - width: Width of the image.
        - height: Height of the image.

        Returns:
        - board_state: 2D list of board state with detected colors.
        """
        cv2.imshow('Original Image', image)

        msrcp = Retinex().msrcp(image)
        cv2.imshow('MSRCP Image', msrcp)

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

                threshold = 1000

                # Get the color of the piece in the cell
                if cv2.countNonZero(blue_mask[y1:y2, x1:x2]) > threshold:
                    board_state[i][j] = 'blue'
                elif cv2.countNonZero(red_mask[y1:y2, x1:x2]) > threshold:
                    board_state[i][j] = 'red'

        # Display the image with cell lines
        cv2.imshow('Image with cell lines', res)

        return board_state

    def preprocess_image_for_green(self, image):
        # Apply MSRCP for better color and contrast
        msrcp = Retinex().msrcp(image)
        cv2.imshow('MSRCP Image', msrcp)

        green_mask = ColorDetector().process_image(msrcp, 'green')

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the green mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, green_mask, msrcp

    def find_chessboard_corners_green(self, frame):
        # Preprocess the image to detect green markers
        contours, green_mask, msrcp = self.preprocess_image_for_green(frame)

        print(f"Number of contours: {len(contours)}")

        cv2.imshow('Green Mask', green_mask)
        cv2.imshow('MSRCP Image', msrcp)

        cv2.waitKey(0)

        # Initialize variables for storing the detected corner coordinates
        green_corners = []

        # Filter contours to find the four green markers
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Filter out small contours
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

        print(f"Top left: {top_left}, Top right: {top_right}, Bottom left: {bottom_left}, Bottom right: {bottom_right}")

        for point in [top_left, top_right, bottom_left, bottom_right]:
            cv2.circle(frame, point, 10, (0, 255, 0), -1)

        cv2.imshow('Detected Corners', frame)
        cv2.waitKey(0)

        return [top_left, top_right, bottom_left, bottom_right]
