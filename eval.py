import gym
import numpy as np
import time
import random

from xml.etree import ElementTree as ET
import importlib.util
import requests
import argparse
import torch
import random
import sys
import importlib
import env

def rot90(pattern, board_size):
    """Rotate the pattern 90 degrees clockwise."""
    return [(board_size - 1 - y, x) for x, y in pattern]

def rot180(pattern, board_size):
    """Rotate the pattern 180 degrees."""
    return [(board_size - 1 - x, board_size - 1 - y) for x, y in pattern]

def rot270(pattern, board_size):
    """Rotate the pattern 270 degrees clockwise."""
    return [(y, board_size - 1 - x) for x, y in pattern]

def flip_h(pattern, board_size):
    """Horizontal flip of the pattern."""
    return [(x, board_size - 1 - y) for x, y in pattern]

def flip_v(pattern, board_size):
    """Vertical flip of the pattern."""
    return [(board_size - 1 - x, y) for x, y in pattern]

def flip_d1(pattern, board_size):
    """Diagonal flip (top-left to bottom-right)."""
    return [(y, x) for x, y in pattern]

def flip_d2(pattern, board_size):
    """Diagonal flip (top-right to bottom-left)."""
    return [(board_size - 1 - y, board_size - 1 - x) for x, y in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        self.symmetry_to_pattern_idx = []
        for idx, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            for sym in syms:
                self.symmetry_patterns.append(sym)
                self.symmetry_to_pattern_idx.append(idx)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        symmetries = [
            pattern,  # Original pattern
            rot90(pattern, self.board_size),
            rot180(pattern, self.board_size),
            rot270(pattern, self.board_size),
            flip_h(pattern, self.board_size),
            flip_v(pattern, self.board_size),
            flip_d1(pattern, self.board_size),
            flip_d2(pattern, self.board_size)
        ]
        # Remove duplicates while preserving order
        unique_symmetries = []
        seen = set()
        for sym in symmetries:
            sym_tuple = tuple(sorted(sym))
            if sym_tuple not in seen:
                seen.add(sym_tuple)
                unique_symmetries.append(sym)

        return unique_symmetries

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            pattern_idx = self.symmetry_to_pattern_idx[i]
            total_value += self.weights[pattern_idx][feature]
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        delta /= len(self.symmetry_patterns)
        total_value = 0
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            pattern_idx = self.symmetry_to_pattern_idx[i]
            self.weights[pattern_idx][feature] += alpha * delta
            total_value += self.weights[pattern_idx][feature]
        return total_value

if __name__ == "__main__":
    env.eval_score()  
    



    