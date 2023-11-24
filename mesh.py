import numpy as np

x = np.arange(0, 5)
y = np.arange(0, 5)
z = np.arange(0, 5)

coord = np.meshgrid(np.stack(x, y), z, indexing="ij")