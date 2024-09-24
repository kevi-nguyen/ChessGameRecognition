import cv2


class VideoCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Error: Could not open video stream with camera index {camera_index}.")

    def get_snapshot(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read frame.")
        return frame

    def save_snapshot(self, filename):
        frame = self.get_snapshot()
        cv2.imwrite(filename, frame)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
