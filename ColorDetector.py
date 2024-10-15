import cv2
import numpy as np


class ColorDetector:
    def __init__(self):
        # Predefined HSV hue ranges for different colors
        self.color_ranges = {
            'blue': (100, 140),
            'red': (0, 10, 160, 180),
            'yellow': (20, 30),
            'green': (35, 85)
        }

    def compute_histogram(self, image_hsv, color_name):
        """
        Compute histograms for each HSV channel focusing on the color's predefined hue range.

        Parameters:
        - image_hsv: HSV image array.
        - color_name: Name of the color to focus on ('blue', 'red', 'yellow', 'green').

        Returns:
        - h_hist: Hue histogram.
        - s_hist: Saturation histogram.
        - v_hist: Value histogram.
        """
        hue_range = self.color_ranges[color_name]
        if color_name == 'red':
            mask1 = cv2.inRange(image_hsv, (hue_range[0], 50, 50), (hue_range[1], 255, 255))
            mask2 = cv2.inRange(image_hsv, (hue_range[2], 50, 50), (hue_range[3], 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(image_hsv, (hue_range[0], 50, 50), (hue_range[1], 255, 255))

        masked_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

        h_hist = cv2.calcHist([masked_hsv], [0], mask, [180], [0, 180])
        s_hist = cv2.calcHist([masked_hsv], [1], mask, [256], [0, 256])
        v_hist = cv2.calcHist([masked_hsv], [2], mask, [256], [0, 256])

        return h_hist, s_hist, v_hist

    def determine_hsv_ranges(self, h_hist, s_hist, v_hist, color_name):
        """
        Dynamically determine HSV ranges for a specific color based on histograms.

        Parameters:
        - h_hist: Hue histogram.
        - s_hist: Saturation histogram.
        - v_hist: Value histogram.
        - color_name: Name of the color to detect ('blue', 'red', 'yellow', 'green').

        Returns:
        - lower_bound: Lower HSV bounds for the color.
        - upper_bound: Upper HSV bounds for the color.
        """
        # Calculate thresholds based on histogram mean and standard deviation
        hue_thresh = 0.1 * h_hist.max()
        sat_thresh = 0.1 * s_hist.max()
        val_thresh = 0.1 * v_hist.max()

        hue_range = np.where(h_hist > hue_thresh)[0]
        saturation_range = np.where(s_hist > sat_thresh)[0]
        value_range = np.where(v_hist > val_thresh)[0]

        # print(
        #     f"Histogram thresholds for {color_name}: hue_thresh={hue_thresh}, sat_thresh={sat_thresh}, val_thresh={val_thresh}")
        # print(f"Detected hue range: {hue_range}")
        # print(f"Detected saturation range: {saturation_range}")
        # print(f"Detected value range: {value_range}")

        # Ensure there is at least one value in each range
        if hue_range.size == 0:
            hue_range = np.array([0, 179])
        if saturation_range.size == 0:
            saturation_range = np.array([0, 255])
        if value_range.size == 0:
            value_range = np.array([0, 255])

        # Set the lower and upper bounds based on detected ranges
        lower_hue = hue_range[0]
        upper_hue = hue_range[-1]
        lower_saturation = saturation_range[0]
        upper_saturation = saturation_range[-1]
        lower_value = value_range[0]
        upper_value = value_range[-1]

        # Adjust bounds based on predefined color ranges
        if color_name == 'blue':
            # Blue hue generally falls between 100 to 140
            lower_hue = max(hue_range[0] - 10, 100)
            upper_hue = min(140, 140)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0] - 200, 70)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0] - 200, 70)  # Avoid very low value
            upper_value = min(255, 255)
        elif color_name == 'red':
            # Red hue generally falls between 0 to 10 or 160 to 180
            lower_hue = max(hue_range[0] - 5, 0)
            upper_hue = min(hue_range[-1], 10)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0] - 100, 70)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0] - 100, 70)  # Avoid very low value
            upper_value = min(255, 255)
        elif color_name == 'yellow':
            # Yellow hue generally falls between 20 to 30
            lower_hue = max(hue_range[0] - 10, 20)
            upper_hue = min(hue_range[-1], 30)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0], 100)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0] - 200, 100)  # Avoid very low value
            upper_value = min(255, 255)
        elif color_name == 'green':
            # Green hue generally falls between 35 to 85
            lower_hue = max(hue_range[0] - 20, 20)
            upper_hue = min(hue_range[-1], 85)
            # Ensure saturation and value are not too broad
            lower_saturation = max(saturation_range[0], 90)  # Avoid very low saturation
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0], 90)  # Avoid very low value
            upper_value = min(255, 255)

        # Ensure that lower bound is less than upper bound
        if lower_hue > upper_hue:
            lower_hue, upper_hue = upper_hue, lower_hue

        #print(f"{color_name.capitalize()} - Lower HSV: ({lower_hue}, {lower_saturation}, {lower_value}), Upper HSV: ({upper_hue}, {upper_saturation}, {upper_value})")

        return (lower_hue, lower_saturation, lower_value), (upper_hue, upper_saturation, upper_value)

    def process_image(self, image, color_name):
        """
        Process the image to detect a specific color.

        Parameters:
        - image: Input BGR image.
        - color_name: Name of the color to detect ('blue', 'red', 'yellow', 'green').

        Returns:
        - mask: Binary mask of the detected color.
        """
        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply histogram equalization to the V channel
        image_hsv[:, :, 2] = cv2.equalizeHist(image_hsv[:, :, 2])

        # Compute histograms for HSV channels
        h_hist, s_hist, v_hist = self.compute_histogram(image_hsv, color_name)

        # Determine HSV ranges for the specified color
        lower_bound, upper_bound = self.determine_hsv_ranges(h_hist, s_hist, v_hist, color_name)

        if color_name == 'red':
            # Red hue generally falls between 0 to 10 and 160 to 180
            lower_red1 = np.array([0, lower_bound[1], lower_bound[2]])
            upper_red1 = np.array([10, upper_bound[1], upper_bound[2]])
            lower_red2 = np.array([160, lower_bound[1], lower_bound[2]])
            upper_red2 = np.array([180, upper_bound[1], upper_bound[2]])

            # Create masks for each red range
            mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # For other colors, create a single mask
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
            mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

        return mask
