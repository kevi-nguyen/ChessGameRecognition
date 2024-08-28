import cv2
import numpy as np
from Retinex import Retinex

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

        #gamma = 0.5
        #adjusted = np.array(255 * (enhanced_gray / 255) ** gamma, dtype='uint8')

        #cv2.imshow('Adjusted Image', adjusted)

        thresholded = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 15,10)

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

    def get_board_state2(self, image, width, height):

        cv2.imshow('Original Image', image)

        msrcp = Retinex().msrcp(image)
        cv2.imshow('MSRCP Image', msrcp)

        # Convert the MSRCP to the HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cv2.imshow('HSV Image', hsv)

        # Apply histogram equalization to the V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        cv2.imshow('Equalized V Channel', hsv)
        cv2.waitKey(0)

        lower_red1 = np.array([0, 105, 105])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 105, 105])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Define color ranges for blue
        lower_blue = np.array([100, 130, 130])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Combine the masks for red and blue
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Bitwise-AND mask and original image
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

    def compute_histogram(self, image_hsv):
        """
        Compute histograms for each HSV channel.

        Parameters:
        - image_hsv: HSV image array.

        Returns:
        - h_hist: Hue histogram.
        - s_hist: Saturation histogram.
        - v_hist: Value histogram.
        """
        h_hist = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])
        return h_hist, s_hist, v_hist

    def determine_hsv_ranges(self, h_hist, s_hist, v_hist, color_name='blue'):
        """
        Determine HSV ranges for a specific color based on histograms.

        Parameters:
        - h_hist: Hue histogram.
        - s_hist: Saturation histogram.
        - v_hist: Value histogram.
        - color_name: Name of the color to detect ('blue' or 'red').

        Returns:
        - lower_bound: Lower HSV bounds for the color.
        - upper_bound: Upper HSV bounds for the color.
        """
        # Find prominent ranges based on histogram thresholds
        hue_thresh = 0.1 * h_hist.max()
        sat_thresh = 0.1 * s_hist.max()
        val_thresh = 0.1 * v_hist.max()

        hue_range = np.where(h_hist > hue_thresh)[0]
        saturation_range = np.where(s_hist > sat_thresh)[0]
        value_range = np.where(v_hist > val_thresh)[0]

        # Ensure there is at least one value in the range
        if hue_range.size == 0:
            hue_range = np.array([0, 179])
        if saturation_range.size == 0:
            saturation_range = np.array([0, 255])
        if value_range.size == 0:
            value_range = np.array([0, 255])

        # Define more precise ranges based on detected color
        if color_name == 'blue':
            # Blue hue generally falls between 100 to 140
            lower_hue = max(hue_range[0], 100)
            upper_hue = min(140, 140)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0], 100)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0], 100)  # Avoid very low value
            upper_value = min(255, 255)
        elif color_name == 'red':
            # Red hue generally falls between 0 to 10 or 160 to 180
            lower_hue = max(hue_range[0], 0)
            upper_hue = min(hue_range[-1], 10)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0], 100)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0], 100)  # Avoid very low value
            upper_value = min(255, 255)
        elif color_name == 'yellow':
            # Yellow hue generally falls between 20 to 30
            lower_hue = max(hue_range[0], 20)
            upper_hue = min(hue_range[-1], 30)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0], 100)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0], 100)  # Avoid very low value
            upper_value = min(255, 255)
        elif color_name == 'green':
            # Green hue generally falls between 35 to 85
            lower_hue = max(hue_range[0], 35)
            upper_hue = min(hue_range[-1], 85)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0], 100)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0], 100)  # Avoid very low value
            upper_value = min(255, 255)

        return (lower_hue, lower_saturation, lower_value), (upper_hue, upper_saturation, upper_value)

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

        # Convert the MSRCP to the HSV color space
        hsv = cv2.cvtColor(msrcp, cv2.COLOR_BGR2HSV)

        # Compute histograms for HSV channels
        h_hist, s_hist, v_hist = self.compute_histogram(hsv)

        # Determine HSV ranges for blue and red dynamically
        lower_red, upper_red = self.determine_hsv_ranges(h_hist, s_hist, v_hist, color_name='red')
        print(f"Lower Red: {lower_red}, Upper Red: {upper_red}")
        lower_blue, upper_blue = self.determine_hsv_ranges(h_hist, s_hist, v_hist, color_name='blue')
        print(f"Lower Blue: {lower_blue}, Upper Blue: {upper_blue}")

        # Apply histogram equalization to the V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        cv2.imshow('Equalized V Channel', hsv)
        cv2.waitKey(0)

        # Threshold the HSV image to get only red colors
        lower_red1 = np.array(lower_red)
        upper_red1 = np.array(upper_red)
        lower_red2 = np.array([160, lower_red[1], lower_red[2]])
        upper_red2 = np.array([180, upper_red[1], upper_red[2]])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Define color ranges for blue
        lower_blue = np.array(lower_blue)
        upper_blue = np.array(upper_blue)

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

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

    def preprocess_image_for_yellow(self, image):
        # Apply MSRCP for better color and contrast
        msrcp = Retinex().msrcp(image)
        cv2.imshow('MSRCP Image', msrcp)

        # Convert the MSRCP to the HSV color space
        hsv = cv2.cvtColor(msrcp, cv2.COLOR_BGR2HSV)

        # Compute histograms for HSV channels
        h_hist, s_hist, v_hist = self.compute_histogram(hsv)

        # Determine HSV ranges for yellow dynamically
        lower_yellow, upper_yellow = self.determine_hsv_ranges(h_hist, s_hist, v_hist, color_name='yellow')
        print(f"Lower Yellow: {lower_yellow}, Upper Yellow: {upper_yellow}")

        # Threshold the HSV image to get only yellow colors
        lower_yellow = np.array(lower_yellow)
        upper_yellow = np.array(upper_yellow)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the yellow mask
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, yellow_mask, msrcp

    def find_chessboard_corners_yellow(self, frame):
        # Preprocess the image to detect yellow markers
        contours, yellow_mask, msrcp = self.preprocess_image_for_yellow(frame)

        print(f"Number of contours: {len(contours)}")

        cv2.imshow('Yellow Mask', yellow_mask)
        cv2.imshow('MSRCP Image', msrcp)

        cv2.waitKey(0)

        # Initialize variables for storing the detected corner coordinates
        yellow_corners = []

        # Filter contours to find the four yellow markers
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out small contours
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    yellow_corners.append((cX, cY))

        if len(yellow_corners) < 4:
            raise ValueError("Could not find at least 4 yellow markers.")

        # Sort corners by x-coordinate to separate left and right
        yellow_corners = sorted(yellow_corners, key=lambda x: x[0])

        # Separate left and right corners
        left_corners = yellow_corners[:2]
        right_corners = yellow_corners[2:]

        # Sort left corners by y-coordinate to identify top-left and bottom-left
        left_corners = sorted(left_corners, key=lambda x: x[1])
        top_left = left_corners[0]
        bottom_left = left_corners[1]

        # Sort right corners by y-coordinate to identify top-right and bottom-right
        right_corners = sorted(right_corners, key=lambda x: x[1])
        top_right = right_corners[0]
        bottom_right = right_corners[1]

        print(f"Top left: {top_left}, Top right: {top_right}, Bottom left: {bottom_left}, Bottom right: {bottom_right}")

        return [top_left, top_right, bottom_left, bottom_right]
