import numpy as np
import matplotlib.pyplot as plt

a = np.array([4, 3, 2, 4, 4, 1, 1, 1, 3, 3, 1, 3, 3, 3, 1, 1])
from scipy import stats
print(stats.mode(a,axis=None))
