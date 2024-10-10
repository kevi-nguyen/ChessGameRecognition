class ChessboardStateLogic:

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

    def find_moved_piece(self, prev_state, curr_state):
        """
        Finds the moved piece by comparing the previous and current board states.

        Parameters:
        - prev_state: 2D list representing the previous state of the board.
        - curr_state: 2D list representing the current state of the board.

        Returns:
        - Tuple containing the move in UCI format
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
                        return 'e1g1'  # Kingside castling
                    elif prev_state[7][4] == 'blue' and prev_state[7][0] == 'blue' and curr_state[7][4] is None and \
                            curr_state[7][0] is None and curr_state[7][2] == 'blue' and curr_state[7][3] == 'blue':
                        return 'e1c1'  # Queenside castling

                    # Check for red castling (red is on the 0th rank)
                    if prev_state[0][4] == 'red' and prev_state[0][7] == 'red' and curr_state[0][4] is None and \
                            curr_state[0][7] is None and curr_state[0][6] == 'red' and curr_state[0][5] == 'red':
                        return 'e8g8'  # Kingside castling
                    elif prev_state[0][4] == 'red' and prev_state[0][0] == 'red' and curr_state[0][4] is None and \
                            curr_state[0][0] is None and curr_state[0][2] == 'red' and curr_state[0][3] == 'red':
                        return 'e8c8'  # Queenside castling

                # En passant
                if start_colour == end_colour:
                    correct_start = start
                else:
                    correct_start = second_start

                if correct_start and end:
                    return correct_start + end
                else:
                    return None

            return start + end

        return None

    def update_board_state(self, board_state, move, special):
        """
        Updates the board state manually after a move.

        Parameters:
        - board_state: 2D list representing the current state of the board.
        - move: Tuple representing the move in the format ((start_row, start_col), (end_row, end_col)).

        Returns:
        - Updated board state after the move.
        """
        move = self.chess_to_cell_notation(move)
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

    def determine_orientation(self, initial_board_state):
        """
        Determine the orientation of the board based on the position of the blue king.
        Returns one of 'bottom', 'top', 'left', 'right'.
        """
        # Check for the Blue (White) king in the key positions
        if 'blue' in str(initial_board_state[7][4]):
            return 'bottom'
        elif 'blue' in str(initial_board_state[0][3]):
            return 'top'
        elif 'blue' in str(initial_board_state[4][0]):
            return 'left'
        elif 'blue' in str(initial_board_state[3][7]):
            return 'right'

        # Default to bottom if the king is not in any of these key positions
        return 'bottom'

    def rotate_board_to_bottom(self, board_state, orientation):
        """
        Rotates the board state to the 'bottom' orientation.

        Parameters:
        - board_state: 2D list representing the current state of the board.
        - orientation: String indicating the current orientation of the board ('bottom', 'top', 'left', 'right').

        Returns:
        - Rotated board state with the 'bottom' orientation.
        """

        # Number of 90-degree clockwise rotations needed to get to 'bottom' orientation
        rotations = {'bottom': 0, 'right': 1, 'top': 2, 'left': 3}

        # Get the number of 90-degree rotations needed
        num_rotations = rotations[orientation]

        # Rotate the board the specified number of times
        for _ in range(num_rotations):
            # Rotate the board 90 degrees clockwise
            board_state = [list(row) for row in zip(*board_state[::-1])]

        return board_state

    def transform_coordinates(self, i, j, orientation):
        """
        Transforms the board coordinates (i, j) based on the current board orientation.

        Parameters:
        - i: Row index (0 to 7).
        - j: Column index (0 to 7).
        - orientation: The current orientation of the board ('bottom', 'left', 'top', 'right').

        Returns:
        - Transformed (i_new, j_new) based on the orientation.
        """
        if orientation == 'bottom':  # Default orientation
            return i, j
        elif orientation == 'left':  # 90° clockwise
            return j, 7 - i
        elif orientation == 'top':  # 180°
            return 7 - i, 7 - j
        elif orientation == 'right':  # 90° counterclockwise
            return 7 - j, i
        else:
            raise ValueError("Invalid orientation. Must be 'bottom', 'left', 'top', or 'right'.")

    def coordinates_to_cobot_move(self, board_state, move, special, orientation):
        move = self.chess_to_cell_notation(move)
        start_pos, end_pos = move
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        double_move = False

        piece = board_state[start_row][start_col]

        if special:
            double_move = True
            # Handle castling
            if piece == 'blue' and start_pos == (7, 4) and end_pos in [(7, 6), (7, 2)]:
                # Kingside castling
                if end_pos == (7, 6):
                    start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((7, 4), (7, 6), orientation)
                    start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((7, 7), (7, 5), orientation)
                    return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2),
                            str(start_j2), str(end_i2), str(end_j2)]
                # Queenside castling
                elif end_pos == (7, 2):
                    start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((7, 4), (7, 2), orientation)
                    start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((7, 0), (7, 3), orientation)
                    return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2),
                            str(start_j2), str(end_i2), str(end_j2)]
            elif piece == 'red' and start_pos == (0, 4) and end_pos in [(0, 6), (0, 2)]:
                # Kingside castling
                if end_pos == (0, 6):
                    start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((0, 4), (0, 6), orientation)
                    start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((0, 7), (0, 5), orientation)
                    return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2),
                            str(start_j2), str(end_i2), str(end_j2)]
                # Queenside castling
                elif end_pos == (0, 2):
                    start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((0, 4), (0, 2), orientation)
                    start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((0, 0), (0, 3), orientation)
                    return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2),
                            str(start_j2), str(end_i2), str(end_j2)]
            # Handle en passant
            if piece == 'blue' and start_row == 3 and end_row == 2 and board_state[end_row][
                end_col] is None and start_col != end_col:
                start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((start_row, start_col),
                                                                              (end_row, end_col), orientation)
                start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((start_row, end_col), (10, 0), orientation)
                return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2),
                        str(start_j2), str(end_i2), str(end_j2)]
            elif piece == 'red' and start_row == 4 and end_row == 5 and board_state[end_row][
                end_col] is None and start_col != end_col:
                start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((start_row, start_col),
                                                                              (end_row, end_col), orientation)
                start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((start_row, end_col), (10, 0), orientation)
                return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2),
                        str(start_j2), str(end_i2), str(end_j2)]
        else:
            start_i1, start_j1, end_i1, end_j1 = (10, 10, 10, 10)

            if board_state[end_row][end_col] is not None:
                double_move = True
                start_i1, start_j1, end_i1, end_j1 = self.positions_to_string((end_row, end_col), (10, 0), orientation)

            start_i2, start_j2, end_i2, end_j2 = self.positions_to_string((start_row, start_col), (end_row, end_col),
                                                                          orientation)

            return [double_move, str(start_i1), str(start_j1), str(end_i1), str(end_j1), str(start_i2), str(start_j2),
                    str(end_i2), str(end_j2)]

    def positions_to_string(self, start, end, orientation):
        """
        Converts the start and end positions into tuples after transforming based on the board orientation.

        Parameters:
        - start: A tuple (start_i, start_j) representing the start position.
        - end: A tuple (end_i, end_j) representing the end position.
        - orientation: The current orientation of the board ('bottom', 'left', 'top', 'right').

        Returns:
        - Two tuples representing the transformed start and end positions.
        """
        # Transform coordinates based on the orientation
        start_transformed = self.transform_coordinates(start[0], start[1], orientation)
        end_transformed = self.transform_coordinates(end[0], end[1], orientation)

        return start_transformed[0], start_transformed[1], end_transformed[0], end_transformed[1]

    def board_to_string(self, board_state):
        """
        Converts the board state (2D list) to a string format for CPEE.
        - None -> '0'
        - 'blue' -> '1'
        - 'red' -> '2'

        Parameters:
        - board_state: 2D list representing the current state of the board.

        Returns:
        - A string representation of the board.
        """
        return ''.join(
            [''.join([str(0) if cell is None else str(1) if cell == 'blue' else str(2) for cell in row]) for row in
             board_state])

    def string_to_board(self, board_string):
        """
        Converts a string representation back to a board state (2D list).

        Parameters:
        - board_string: String representation of the board state.

        Returns:
        - 2D list representing the current state of the board.
        """
        board_size = 8
        return [[None if c == '0' else 'blue' if c == '1' else 'red' for c in board_string[i:i + board_size]] for i in
                range(0, len(board_string), board_size)]
