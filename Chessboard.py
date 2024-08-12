import chess


class Chessboard:
    def __init__(self):
        # Initialize the board in the starting position
        self.board = chess.Board()

    def move_piece(self, move):
        # Validate the move
        if chess.Move.from_uci(move) in self.board.legal_moves:
            # Make the move
            self.board.push(chess.Move.from_uci(move))
            return True
        else:
            return False

    def is_checkmate(self):
        # Check if the current board state is checkmate
        return self.board.is_checkmate()

    def fen(self):
        # Return the FEN string of the current board state
        return self.board.fen()

    def display_move(self, move, piece):
        # Castling moves
        if move in ["e1g1", "e1c1", "e8g8", "e8c8"]:
            if move == "e1g1" or move == "e8g8":
                print("Castling: Kingside")
            elif move == "e1c1" or move == "e8c8":
                print("Castling: Queenside")
        # En passant moves
        elif piece.lower() == 'p' and move[1] != move[3] and (move[2] == '5' or move[2] == '6'):
            print("En Passant")
        else:
            print(f"Move: {move}")

    def is_special_move(self, move):
        """
        Check if the move is a special move (castling or en passant).

        Parameters:
        - move: String representing the move in UCI format, e.g., 'e1g1'.

        Returns:
        - Boolean indicating if the move is a special move.
        """
        uci_move = chess.Move.from_uci(move)
        if uci_move in self.board.legal_moves:
            if self.board.is_castling(uci_move):
                return True
            if self.board.is_en_passant(uci_move):
                return True
        return False
