import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        list_e = [np.exp(x-max(z)) for x in z]
        sum_ez = sum(list_e)
        prob = [np.round(x/sum_ez,4) for x in list_e]
        return prob
