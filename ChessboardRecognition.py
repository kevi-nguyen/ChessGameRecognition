import cv2
import numpy as np


class ChessboardRecognition:

    def find_chessboard_corners(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Gray Image', gray)

        # Apply adaptive histogram equalization to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        cv2.imshow('Enhanced Gray Image', enhanced_gray)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)

        cv2.imshow('Blurred Image', blurred)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(enhanced_gray, (7, 7), None)

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

        cv2.imshow('Original Image', image)

        # Apply the Gray World Assumption to normalize the colors
        normalized_image = self.apply_gray_world_assumption(image)

        cv2.imshow('Normalized Image', normalized_image)

        # Apply a Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        cv2.imshow('Blurred Image', blurred_image)

        # Convert the blurred image to the HSV color space
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        cv2.imshow('HSV Image', hsv)

        # Apply histogram equalization to the V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        cv2.imshow('Equalized V Channel', hsv)

        lower_red1 = np.array([0, 105, 105])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 105, 105])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Define color ranges for blue
        lower_blue = np.array([110, 100, 100])
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

    def apply_gray_world_assumption(self, image):
        # Ensure the image is valid
        if image is None:
            raise ValueError("Invalid image provided")

        # Convert the image to the LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into its channels
        l, a, b = cv2.split(lab)

        # Calculate the average values of the A and B channels
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        print(f"Average A: {avg_a}, Average B: {avg_b}")

        # Adjust the A and B channels to make the average color gray
        a = a - ((avg_a - 128) * (l / 255.0) * 1.1)
        b = b - ((avg_b - 128) * (l / 255.0) * 1.1)

        # Clip the values to ensure they are within valid range
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

        # Merge the adjusted channels back into a LAB image
        lab = cv2.merge([l, a, b])

        # Convert the LAB image back to the BGR color space
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

