## Project Journey and Challenges

I considered using a machine learning model for chessboard detection, but the complexity and downsides—like training data requirements and unpredictable performance—led me to explore alternative approaches. Ultimately, I opted for a colour recognition approach, as there were already red and blue chess pieces available. 

The colour recognition approach doesn’t difference between individual pieces and only recognizes the two sides (blue and red). So, I must parallelly keep track of the game state with a virtual chessboard to always know the pieces position. That means that only full games from the start are possible.

I needed to implement chess logic, such as handling special moves (like castling or en passant), and updating the red/blue board state after a move. The main challenge here was to transform moves in chess notation to i and j coordinates for my board and of course to map these coordinates to corresponding special moves.

Another hurdle was the possibility that the player sets up the game differently (for example red and blue switched). I assumed that blue has to be the white side and modified my program so that it is able to detect the boards orientation by searching for the blue king position on the board and rotating it to default “bottom” afterwards. This happens at initialisation of the chessboard.

At first, I opted for hard-coded coordinates to detect the corners of the chessboard, but this solution was too fragile, especially when small camera movements or lighting changes affected the results. I concluded that I have to detect the corners in my program to ensure a robust chessboard detection.

I tried using the method cv2.findChessboardCorners() to detect the chessboard. But because it is triggered by white and black squares and only considers the inner corners of the chessboard, I was only able to detect the inner 6x6 chessboard and its corners. So, I calculated the actual corners of the 8x8 chessboard by using the inner corners and the average square size.

Another problem was that this method was really sensitive to lighting changes. So, after trying to make it more robust with grey scale and blur techniques, I decided that this approach is not suitable for my approach.

In my search to achieve the highest possible colour constancy I also tried using the MRSCP (Multi Scale Retinex with Colour Preservation) algorithm with further adjustments. These led to satisfying results, so I continued using this pipeline for my next steps.

Finally, my solution was to use green markers at the chessboard corners to reliably detect the corners coordinates.

To detect the pieces and the corners I needed detect colour. So, I started experimenting with different colour masks. The problem was that the red and blue HSV ranges changed with different lighting conditions (although using my colour constancy pipeline). Therefore, my hard coded ranges were inefficient. I decided to dynamically adjust HSV ranges based on the existing blue and red colours frame. This way the program was always automatically adjusting its HSV ranges to only capture the exact blue and red HSV values of the chess pieces.

Another layer of complexity was integrating the cobot to grab different pieces accurately. I had to measure the exact coordinates and distance so that the robot was able to grab all pieces with one unique pick-up motion.

Finally, a significant technical hurdle was installing the Intel RealSense SDK and building it from source (for MacOS), which was necessary to enable camera interaction for the chessboard detection.

