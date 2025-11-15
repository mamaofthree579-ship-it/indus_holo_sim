import numpy as np

def create_grid(grid_size=200):
    """
    Small convenience wrapper: returns x,y meshgrid in normalized coords 0..1
    """
    X = np.linspace(0.0, 1.0, grid_size)
    Y = np.linspace(0.0, 1.0, grid_size)
    XX, YY = np.meshgrid(X, Y)
    return XX, YY
