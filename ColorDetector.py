import cv2
import numpy as np


class ColorDetector:
    def __init__(self):
        self.color_ranges = {
            'blue': (100, 140),
            'red': (0, 10, 160, 180),
            'yellow': (20, 30),
            'green': (35, 85)
        }

    def compute_histogram(self, image_hsv, color_name):
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
        # Calculate thresholds based on histogram mean and standard deviation
        hue_thresh = 0.1 * h_hist.max()
        sat_thresh = 0.1 * s_hist.max()
        val_thresh = 0.1 * v_hist.max()

        hue_range = np.where(h_hist > hue_thresh)[0]
        saturation_range = np.where(s_hist > sat_thresh)[0]
        value_range = np.where(v_hist > val_thresh)[0]

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
            lower_hue = max(hue_range[0] - 10, 100)
            upper_hue = min(140, 140)
            lower_saturation = max(saturation_range[0] - 200, 70)
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0] - 200, 70)
            upper_value = min(255, 255)
        elif color_name == 'red':
            lower_hue = max(hue_range[0] - 5, 0)
            upper_hue = min(hue_range[-1], 10)
            lower_saturation = max(saturation_range[0] - 100, 70)
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0] - 100, 70)
            upper_value = min(255, 255)
        elif color_name == 'yellow':
            lower_hue = max(hue_range[0] - 10, 20)
            upper_hue = min(hue_range[-1], 30)
            lower_saturation = max(saturation_range[0], 100)
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0] - 200, 100)
            upper_value = min(255, 255)
        elif color_name == 'green':
            lower_hue = max(hue_range[0] - 20, 20)
            upper_hue = min(hue_range[-1], 85)
            lower_saturation = max(saturation_range[0], 90)
            upper_saturation = min(255, 255)
            lower_value = max(value_range[0], 90)
            upper_value = min(255, 255)

        if lower_hue > upper_hue:
            lower_hue, upper_hue = upper_hue, lower_hue

        return (lower_hue, lower_saturation, lower_value), (upper_hue, upper_saturation, upper_value)

    def process_image(self, image, color_name):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image_hsv[:, :, 2] = cv2.equalizeHist(image_hsv[:, :, 2])

        h_hist, s_hist, v_hist = self.compute_histogram(image_hsv, color_name)

        lower_bound, upper_bound = self.determine_hsv_ranges(h_hist, s_hist, v_hist, color_name)

        # Red need two masks
        if color_name == 'red':
            lower_red1 = np.array([0, lower_bound[1], lower_bound[2]])
            upper_red1 = np.array([10, upper_bound[1], upper_bound[2]])
            lower_red2 = np.array([160, lower_bound[1], lower_bound[2]])
            upper_red2 = np.array([180, upper_bound[1], upper_bound[2]])

            mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
            mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

        return mask
