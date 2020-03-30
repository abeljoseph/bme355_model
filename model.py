import numpy as np
from scipy.integrate import solve_ivp
from math import exp
import matplotlib.pyplot as plt


class Model:
    """
    Model of a shank foot undergoing functional electrical stimulus (FES); a Nonlinear State-Space Model
    """

    def __init__(self):
        # Constants mentioned in simulation plan
        self.T_act = 0.01
        self.T_deact = 0.04
        self.J = 0.0197
        self.d = 3.7
        self.B = 0.82
        self.c_F = 11.45
        self.m_F = 1.0275
        self.a_v = 1.33
        self.f_v1 = 0.18
        self.f_v2 = 0.023
        self.v_max = -0.9
        self.f_max = 600
        self.w = 0.56
        self.l_t = 22.3
        self.l_mt0 = 32.1
        self.l_ce = self.l_mt0 - self.l_t
        self.l_ce_opt = None  # TODO: Define
        self.l_foot = 0.26 #m
        self.a1 = 2.1
        self.a2 = -0.08
        self.a3 = -7.97
        self.a4 = 0.19
        self.a5 = -1.79
        self.a = [self.a1, self.a2, self.a3, self.a4, self.a5]
        self.g = 9.81

        # TODO: Define these variables
        self.u = u
        self.ax = ax
        self.az = az
        self.a_shank = a_shank
        self.a_shank1 = a_shank1
        self.x_ext = [self.ax, self.az, self.a_shank, self.a_shank1]
        self.f_act = f_act

        # TODO: Determine whether to remove these
        self.a_f = a_f
        self.a_f1 = a_f1
        self.x = [self.f_act, self.a_f, self.a_f1]

    def get_torque_grav(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Gravity torque of the foot around the ankle
        """
        return -self.m_F*self.c_F*self.g*np.cos(x[1])

    def get_torque_acc(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Torque induced by the movement of the ankle
        """
        return self.m_F*self.c_F*(self.x_ext[0]*np.sin(x[1]) - self.x_ext[1]*np.cos(x[1]))

    def get_torque_ela(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Passive elastic torque around the ankle due to passive muscles and tissues
        """
        return np.exp(self.a1 + self.a2 * x[1]) - np.exp(self.a3 + self.a4 * x[1]) + self.a5

    def get_force_fl(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        w, l_ce, l_ce_opt = self.w, self.l_ce, self.l_ce_opt

        # TODO: Remove error handling after l_ce_opt is implemented
        try: return exp(-(-(l_ce - l_ce_opt)/w * l_ce_opt)**2) / (self.x_ext[2] - x[1])
        except TypeError: print("Ensure that model.l_ce_opt is defined.")
    
    def get_force_fv(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        v_ce = self.d * (self.x_ext[3] - x[3])  # contraction speed
        v_max, a_v, f_v1, f_v2 = self.v_max, self.a_v, self.f_v1, self.f_v2
        if v_ce < 0:
            return ((1 - v_ce/v_max) / (1 + v_ce/(v_max*f_v1))) / (self.x_ext[3] - x[2])
        return ((1 + a_v*(v_ce/f_v2)) / (1 + v_ce/f_v2)) / (self.x_ext[3] - x[2])

    def get_force_m(self, x):
        f_fl, f_fv = self.get_force_fl(x), self.get_force_fv(x)
        return x[0] * self.f_max * f_fl * (self.x_ext[2] - x[1]) * f_fv * (self.x_ext[3] - x[2])

    def get_length_mt(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        return self.l_mt0 + (self.x_ext[2] - x[1])*self.d

    def get_toe_height(self, ankle_height, ankle_angle):
        return ankle_height - l_foot*np.sin(ankle_angle)

    def get_derivative(self, t, x, u):
        """
        :param t: time
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: time derivatives of state variables
        """
        return [(u - x[0]) * (u/self.T_act - (1 - u)/self.T_deact),
                x[2],
                (1/self.J) * self.get_force_m(x[0])*self.d + self.get_torque_grav(x) + self.get_torque_acc(x) + self.get_torque_ela(x)]

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
