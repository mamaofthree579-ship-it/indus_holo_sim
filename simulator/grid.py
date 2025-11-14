import numpy as np

def create_grid(grid_size=256):
    X = np.linspace(0, 1, grid_size)
    Y = np.linspace(0, 1, grid_size)
    XX, YY = np.meshgrid(X, Y)
    return XX, YY
