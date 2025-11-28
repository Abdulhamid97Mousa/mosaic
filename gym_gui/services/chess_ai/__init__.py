"""Chess AI services for Human vs Agent gameplay.

This package provides AI engines that can play chess against human players:
- StockfishService: Wrapper for the Stockfish chess engine
"""

from gym_gui.services.chess_ai.stockfish_service import StockfishService

__all__ = ["StockfishService"]
