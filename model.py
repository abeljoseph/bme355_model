import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Model:
    """
    Model of a shank foot undergoing functional electrical stimulus (FES); a Nonlinear State-Space Model
    """

    def __init__(self):
        # Constants mentioned in simulation plan
        self.Tact = 0.01
        self.Tdeact = 0.04
        self.J = 0.0197
        self.d = 3.7
        self.B = 0.82
        self.cF = 11.45
        self.mF = 1.0275
        self.av = 1.33
        self.fv1 = 0.18
        self.fv2 = 0.023
        self.vmax = -0.9
        self.fmax = 600
        self.w = 0.56
        self.lt = 22.3
        self.lmt = 32.1
        self.a1 = 2.1
        self.a2 = -0.08
        self.a3 = -7.97
        self.a4 = 0.19
        self.a5 = -1.79
        self.a = [self.a1, self.a2, self.a3, self.a4, self.a5]
        self.u = u
        self.ax = ax
        self.az = az
        self.ashank = ashank
        self.ashank1 = ashank1
        self.x_ext = [self.ax, self.az, self.ashank, self.ashank1]
        self.fact = fact
        self.af = af 
        self.af1 = af1
        self.x = [self.fact, self.af, self.af1]
        self.g = 9.81

    def get_torque_grav(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Gravity torque of the foot around the ankle
        """
        Tgrav = -self.mF*self.cF*self.g*np.cos(self.x[1])
        return Tgrav

    def get_torque_acc(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Torque induced by the movement of the ankle
        """
        Tacc = self.mF*self.cF*(self.x_ext[0]*np.sin(self.x[1]) - self.x_ext[1]*np.cos[self.x[1]])
        return Tacc

    def get_torque_ela(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Passive elastic torque around the ankle due to passive muscles and tissues
        """
        Tela = np.exp(self.a1 + self.a2*self.x[1]) - np.exp(self.a3 + self.a4*self.x[1]) + self.a5
        return Tela

    def get_force_fl(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        # Use simulation plan equation
        pass
    
    def get_force_fv(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        # Use simulation plan equation
        return

    def get_length_mt(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        # Use simulation plan equation
        pass

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: time derivatives of state variables
        """
        pass

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
    pass


if __name__ == '__main__':
    # Initiate Model
    model = Model()
    
    # Simulate Model
    t, x = model.simulate(10)
    plot_graphs(model, t, x)
