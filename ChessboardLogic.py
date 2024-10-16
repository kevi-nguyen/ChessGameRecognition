import chess


class ChessboardLogic:

    def move_piece(self, fen, move):
        board = chess.Board(fen)
        if chess.Move.from_uci(move) in board.legal_moves:
            board.push(chess.Move.from_uci(move))
            return board.fen()
        else:
            return False

    def is_checkmate(self, fen):
        board = chess.Board(fen)
        # Check if the current board state is checkmate
        return board.is_checkmate()

    def is_special_move(self, fen, move):
        board = chess.Board(fen)
        uci_move = chess.Move.from_uci(move)
        if uci_move in board.legal_moves:
            if board.is_castling(uci_move):
                return True
            if board.is_en_passant(uci_move):
                return True
        return False

    def is_illegal_move(self, fen, move):
        board = chess.Board(fen)
        if chess.Move.from_uci(move) in board.legal_moves:
            return False
        else:
            return True
