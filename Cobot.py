from ChessboardStateLogic import ChessboardStateLogic


class Cobot:

    def __init__(self):
        self.out_i = 9
        self.out_j = 9

    def coordinates_to_cobot_move(self, board_state, move, special, orientation):
        start_pos, end_pos = move
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        piece = board_state[start_row][start_col]

        if special:
            # Handle castling
            if piece == 'blue' and start_pos == (7, 4) and end_pos in [(7, 6), (7, 2)]:
                # Kingside castling
                if end_pos == (7, 6):
                    self.move_piece(ChessboardStateLogic().transform_coordinates(7, 4, orientation),
                                    ChessboardStateLogic().transform_coordinates(7, 6, orientation))
                    self.move_piece(ChessboardStateLogic().transform_coordinates(7, 7, orientation),
                                    ChessboardStateLogic().transform_coordinates(7, 5, orientation))
                # Queenside castling
                elif end_pos == (7, 2):
                    self.move_piece(ChessboardStateLogic().transform_coordinates(7, 4, orientation),
                                    ChessboardStateLogic().transform_coordinates(7, 2, orientation))
                    self.move_piece(ChessboardStateLogic().transform_coordinates(7, 0, orientation),
                                    ChessboardStateLogic().transform_coordinates(7, 3, orientation))
            elif piece == 'red' and start_pos == (0, 4) and end_pos in [(0, 6), (0, 2)]:
                # Kingside castling
                if end_pos == (0, 6):
                    self.move_piece(ChessboardStateLogic().transform_coordinates(0, 4, orientation),
                                    ChessboardStateLogic().transform_coordinates(0, 6, orientation))
                    self.move_piece(ChessboardStateLogic().transform_coordinates(0, 7, orientation),
                                    ChessboardStateLogic().transform_coordinates(0, 5, orientation))
                # Queenside castling
                elif end_pos == (0, 2):
                    self.move_piece(ChessboardStateLogic().transform_coordinates(0, 4, orientation),
                                    ChessboardStateLogic().transform_coordinates(0, 2, orientation))
                    self.move_piece(ChessboardStateLogic().transform_coordinates(0, 0, orientation),
                                    ChessboardStateLogic().transform_coordinates(0, 3, orientation))

            # Handle en passant
            if piece == 'blue' and start_row == 3 and end_row == 2 and board_state[end_row][
                end_col] is None and start_col != end_col:
                self.move_piece(ChessboardStateLogic().transform_coordinates(start_row, start_col, orientation),
                                ChessboardStateLogic().transform_coordinates(end_row, end_col, orientation))
                self.move_piece(ChessboardStateLogic().transform_coordinates(start_row, end_col, orientation),
                                (self.out_i, self.out_j))
            elif piece == 'red' and start_row == 4 and end_row == 5 and board_state[end_row][
                end_col] is None and start_col != end_col:
                self.move_piece(ChessboardStateLogic().transform_coordinates(start_row, start_col, orientation),
                                ChessboardStateLogic().transform_coordinates(end_row, end_col, orientation))
                self.move_piece(ChessboardStateLogic().transform_coordinates(start_row, end_col, orientation),
                                (self.out_i, self.out_j))
        else:

            if board_state[end_row][end_col] is not None:
                self.move_piece(ChessboardStateLogic().transform_coordinates(end_row, end_col, orientation),
                                (self.out_i, self.out_j))

            self.move_piece(ChessboardStateLogic().transform_coordinates(start_row, start_col, orientation),
                            ChessboardStateLogic().transform_coordinates(end_row, end_col, orientation))

    def move_piece(self, start_pos, end_pos):
