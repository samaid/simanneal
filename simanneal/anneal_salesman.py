from simanneal import VectorAnnealer

try:
    import mkl_random as random
except ImportError:
    import numpy.random as random


class TravellingSalesmanProblem(VectorAnnealer):

    """Test annealer with a travelling salesman problem.
    """

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, distance_matrix):
        super().__init__(state)
        self.distance_matrix = distance_matrix
        self.__fill_random_buffer()

    def __fill_random_buffer(self):
        self.__random_buffer = random.randint(0, len(self.state) - 1, size=self.batch_size)
        self.__random_buffer_counter = 0

    def random(self):
        k = self.__random_buffer_counter
        if k >= self.batch_size:
            self.__fill_random_buffer()
            k = 0
        rn = self.__random_buffer[k]
        self.__random_buffer_counter += 1
        return rn

    def move(self):
        def _dm(i, j):
            si = self.state[i]
            sj = self.state[j]
            d = self.distance_matrix[si][sj]
            return d

        """Swaps two cities in the route."""
        a = self.random()
        b = self.random()

        # Permutation positions - we want i < j for the following code under `else` statement to work correctly
        i = min(a, b)
        j = max(a, b)

        # Calculate energy delta for this candidate
        # Apply different formulas for energy delta for neighbor and distant swap candidates
        if j - i > 1:
            # Distant permutation
            dE = _dm(i - 1, j) + _dm(j, i + 1) + _dm(j - 1, i) + _dm(i, j + 1) - \
                 _dm(i - 1, i) - _dm(i, i + 1) - _dm(j - 1, j) - _dm(j, j + 1)
        else:
            # Neighbors permutation
            dE = _dm(i - 1, j) + _dm(i, j + 1) - \
                 _dm(i - 1, i) - _dm(j, j + 1)

        self.state[a], self.state[b] = self.state[b], self.state[a]
        return (a, b), dE

    def undo_move(self, state_change):
        a, b = state_change
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """Calculates the length of the route."""
        e = 0
        for i in range(len(self.state)):
            e += self.distance_matrix[self.state[i-1]][self.state[i]]
        return e
