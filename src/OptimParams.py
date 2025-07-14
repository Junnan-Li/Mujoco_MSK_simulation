from dataclasses import dataclass


@dataclass
class OptimParams:
    learning_rate: float = 0.01
    max_iter: int = 1000
    tol: float = 1e-4
