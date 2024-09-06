import os
import chess
from stockfish import Stockfish
from fastapi import FastAPI, HTTPException
import uvicorn

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH') or 'stockfish'
stockfish = Stockfish(path=STOCKFISH_PATH)

app = FastAPI()


@app.get("/get_move")
def get_move(fen: str):
    # Validate the FEN string
    try:
        chess.Board(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    return {"move": best_move}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)