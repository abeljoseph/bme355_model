import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Model:
    """
    Model of a shank foot undergoing functional electrical stimulus (FES); a Nonlinear State-Space Model
    """

    def __init__(self):
        # Constants mentioned in simulation plan
        return

    def get_torque_grav(self, x):
        """
        :param x: x param
        """
        # Use simulation plan equation
        return

    def get_torque_acc(self, x):
        """
        :param x: x param
        """
        # Use simulation plan equation
        return

    def get_torque_ela(self, x):
        """
        :param x: x param
        """
        # Use simulation plan equation
        return

    def get_force_fl(self, x):
        """
        :param x: x param
        """
        # Use simulation plan equation
        return
    
    def get_force_fv(self, x):
        """
        :param x: x param
        """
        # Use simulation plan equation
        return

    def get_length_mt(self, x):
        """
        :param x: x param
        """
        # Use simulation plan equation
        return

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [X1, X2, X3]
        :return: time derivatives of state variables
        """
        return

    def simulate(self, total_time):
        """
        :param total_time: seconds to simulate
        :return: time, state (times at which the state is estimated, state vector at each time)
        """
        # Decide an initial condition, this might change due to all the variables present.
        x0 = [0, 0, 0]
        t_span = (0, total_time)
        dt = 0.01

        sol = solve_ivp(self.get_derivative, t_span, x0, max_step=dt)
        return sol.t, sol.y.T


def plot_graphs(model, time, states):
    # plt.title('States of Circulation versus Time')
    # plt.plot(time, states[:, 0], 'k', label='Ventricular Pressure')
    # plt.plot(time, states[:, 1], 'g', label='Atrial Pressure')
    # plt.plot(time, states[:, 2], 'r', label='Arterial Pressure')
    # plt.plot(time, aortic_pressure, 'c', label='Aortic Pressure')
    # plt.ylabel('Pressure (mmHg)')
    # plt.xlabel('Time (s)')
    # plt.legend(loc='upper left')
    # plt.show()
    return


if __name__ == '__main__':
    # Initiate Model
    model = Model()
    
    # Simulate Model
    t, x = model.simulate(5)
    plot_graphs(model, t, x)