from simanneal import Annealer
import abc
import time

try:
    import mkl_random as random
except ImportError:
    import numpy.random as random


class VectorAnnealer(Annealer):

    batch_size = 8096

    @abc.abstractmethod
    def undo_move(self, state_change):
        pass

    def anneal_batch(self, T, Tfactor, step, batch_size):
        trials = accepts = improves = 0

        x = random.exponential(size=batch_size)

        E = self.E

        for k in range(batch_size):
            step += 1

            E = self.E
            state_change, dE = self.move()

            if dE is None:
                dE = self.energy() - E

            if abs(dE) <= self.tolerance:
                # Avoid low energy noise impacting auto-tuning of schedule
                dE = 0.0

            E += dE

            trials += 1

            if dE < 0.0 or x[k] > dE/T:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1

                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            else:
                # Reject new state and roll back to old one
                self.undo_move(state_change)
                E = self.E
            self.E = E
            T *= Tfactor

        return T, E, trials, accepts, improves

    def anneal(self, callback=None, **kwargs):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        self.E = self.energy()
        self.best_energy = self.E
        self.best_state = self.copy_state(self.state)

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')

        # Note initial state
        Tfactor = (self.Tmin/self.Tmax) ** (1.0/self.steps)
        T = self.Tmax
        E = self.E

        self.update(step, T, E, None, None)

        self.batch_size = min(self.batch_size, self.steps)
        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            trials = accepts = improves = 0
            dt = 0
            if not self.pause:
                t1 = time.time()
                T, E, trials, accepts, improves = self.anneal_batch(T, Tfactor, step, self.batch_size)
                dt = time.time()-t1

                step += self.batch_size
                self.update(step, T, E, accepts / trials, improves / trials)
            if callback is not None:
                callback(self, step, trials, accepts, improves, dt)

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy
