# Chess Game Recognition

This project implements a chess game recognition system using computer vision. The system captures a snapshot of the chessboard, processes the image to recognize the positions of chess pieces based on color, and then transform the coordinates, allowing a robot to interact with the chessboard. The game logic is integrated with Stockfish to determine the best possible moves, and the system can and should communicate with external services.

## Features

- **Color-Based Chessboard Recognition**: Identifies chess pieces based on color (blue for white pieces and red for black pieces) and determines their positions on the board.
- **Automatic Move Detection**: Tracks and identifies chess moves by comparing previous board states to the current one.
- **Stockfish Integration**: Utilizes Stockfish to compute the best move for the robot after each player move.
- **Robot Control**: Translates chess moves into robot coordinates to make physical movements on the chessboard.
- **Base64 Encoded Image Processing**: The recognition system expects a chessboard image encoded in Base64 format and will decode it and further operate on it.
- **FastAPI Integration**: Provides a handful RESTful API interfaces in a microservice architecture for managing chess games and processing snapshots.

## Requirements

- Python 3.x
- OpenCV for image processing.
- FastAPI for web services.
- Stockfish for chess AI.
- Numpy for matrix operations.
- ... see below

The project dependencies can be installed using `pip`:

```
opencv-python
opencv-python-headless
python-chess
stockfish
fastapi
uvicorn
numpy
requests
chess
pydantic
python-multipart
pyserial
pyrealsense2-macosx
anyio
h11
jinja2
setuptools
wheel
certifi
urllib3
```
Additional Requirements
- Chess Pieces: The pieces should be colored blue and red to represent white and black pieces, respectively.
- Chessboard Corners: Use green markers for the corners of the chessboard. These are mandatory to recognize the chessboard accurately.
- Image Orientation: The images must be adjusted so that the defined “bottom” orientation aligns with the bottom side of the image. The “bottom” refers to the side where the player stands in opposition to the robot. Then you will able to set up the pieces side the way you like - the recognition system will automatically rotate the board to "bottom".

Prerequites for a full chess game with this repository
- Button: A physical button is used to signal the robot its turn.
- Robot Arm: A robotic arm is used to interact with the chess pieces.
- Process Engine: An orchestrating process engine is required to manage the interactions between the provided services and the robot.
- Intel RealSense Camera: Ensure that an Intel RealSense camera is used for capturing the chessboard and pieces.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kevi-nguyen/ChessGameRecognition.git
   cd ChessGameRecognition
   ```

2. **Install Dependencies**:
   Activate your virtual environment and install the required dependencies from the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```
3. **Adjust Paths and Ports**:
   Before running the Chess Game Recognition system, you may need to adjust certain file paths and port configurations for the correct operation of key components like **Stockfish** and the **button integration**.


   **Stockfish Path Configuration**:
      The project relies on Stockfish for chess move analysis, and it needs to know where the Stockfish executable is located. By default, the code will look for `stockfish` (or `stockfish.exe` on Windows). However, this assumes either:
        - The executable is in the same directory as the script.
        - Or it is in the system's `PATH` environment variable.
  
      If Stockfish is installed in a custom directory, you can either:
      1. Place `stockfish.exe` in the same folder as your script.
      2. Set the correct path explicitly in the code, like so:

      ```python
      STOCKFISH_PATH = os.getenv('STOCKFISH_PATH') or 'C:/path/to/stockfish.exe'
      stockfish = Stockfish(path=STOCKFISH_PATH)
      ```

      For Linux systems, update the path similarly with the appropriate executable path, such as:

      ```python
      STOCKFISH_PATH = os.getenv('STOCKFISH_PATH') or '/usr/local/bin/stockfish'
      ```
   **Button Integration (Serial Port Adjustment)**:
      If you're using a button connected via a serial port, the `port` variable in the code needs to match the serial port available on your computer.
      The current port is configured for a macOS system:
      ```python
      port = '/dev/tty.usbserial-AQ027FTG'
      ```
      On a Windows system, you will need to change this to the appropriate **COM port** (e.g., `COM3`), and on Linux, it might look like `/dev/ttyUSB0`. Adjust the code as follows:

      - For **Windows**:
      ```python
      port = 'COM3'  # Replace with the correct COM port number
      ```

      - For **Linux**:
      ```python
      port = '/dev/ttyUSB0'  # Replace with the correct USB serial port
      ```

      Ensure that the correct serial driver is installed and that the device is properly connected.

5. **Run the Services**:
   There will be four different services, which can be started independently:
   - Chessboard-API
   - ChessboardRecognition-API
   - Button-API
   - Stockfish-API
   - Camera
   - (Monotlith-API)
   
   To start the chess game recognition services, use the following command:
   ```bash
   python <specific service>.py
   ```
   This will start the FastAPI service on different URLs for example `http://0.0.0.0:8080`, and it will expose endpoints for interacting with the chessboard, recognizing moves, and sending them to the robot.

## Overview of the Game Structure

The chess game recognition system works through the following steps:

1. **Snapshot Capture**: A snapshot of the chessboard is taken using a camera and encoded in Base64 format.
2. **Corner Detection**: The image is processed to detect the green-marked corners of the chessboard. This establishes the coordinates of the chessboard in the image.
3. **Chessboard Mapping**: Once the corners are detected, the board is divided into 64 squares. Each square can be analyzed to detect whether a blue or red chess piece is present (blue for white pieces, red for black pieces).
4. **Board State Management**: The initial board state is tracked, and the system continuously updates the board state by comparing the initial state to the current state after each player's move. Concurrently, a virtual chessboard is tracked with the actual pieces. In that way we always know which piece is on which square.
5. **Move Detection**: When the player makes a move, the system compares the previous board state with the current one to detect which piece was moved. After the board state and the virtual chessboard are updated accordingly, the current FEN string is sent to Stockfish to get the best response.
6. **Robot Move Execution**: After Stockfish computes the best move, we are transforming the chess move into robot coordinates and sent these to the robot to execute the move.
7. **Cycle Repeats**: The player moves again and presses a button to signal the end of their move, a new snapshot is taken, and the process starts again.

## Color Recognition

### Image Processing Pipeline

The system employs an advanced image processing pipeline to ensure accurate detection of blue and red chess pieces, even under varying lighting conditions. The process is divided into several stages:

1. **Multi-Scale Retinex with Color Preservation**:
   The image is pre-processed using a technique called Multi-Scale Retinex with Color Preservation (MSRCP) to maintain color constancy. This reduces the effect of lighting variations on color detection. Gaussian blur, histogram equalization, and other operations are applied to enhance image quality.
   
   Example of converting to HSV color space and applying histogram equalization:
   ```python
   # Convert the image to HSV color space
   image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   # Apply histogram equalization to the V channel
   image_hsv[:, :, 2] = cv2.equalizeHist(image_hsv[:, :, 2])
   ```

2. **Masking and Color Segmentation**:
   For detecting specific colors (blue and red in this case), the system computes histograms for each HSV channel, focusing on the predefined hue range for each color. The histogram data is then used to create a mask that isolates the regions of the image containing the target color.

   Example:
   ```python
   # Compute histograms for HSV channels
   h_hist, s_hist, v_hist = self.compute_histogram(image_hsv, color_name)

   # Determine HSV ranges for the specified color
   lower_bound, upper_bound = self.determine_hsv_ranges(h_hist, s_hist, v_hist, color_name)

   # Create a mask based on the HSV color range
   mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
   ```

3. **Morphological Operations**:
   After masking, the system applies morphological operations (like erosion and dilation) to clean up the mask, removing noise and enhancing the regions that represent chess pieces.
   
   ```python
   # Apply morphological operations to clean the mask
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   ```

4. **Analyzing Chess Piece Colors**:
   Once the masks are generated, the system analyzes the image to locate blue and red pieces based on the dynamically predefined HSV ranges. This allows the system to determine the position of each piece on the board.

### Conclusion

With this approach, the system ensures accurate detection of chess pieces in various lighting conditions and maintains precise tracking of the chess game.

By following the detailed steps and integrating color recognition with Stockfish and robotic control, the chess game recognition system provides a seamless interface for human-robot chess interaction.

---

MIT License

Copyright (c) 2024 Kevin Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
