import numpy as np
from scipy.integrate import solve_ivp
from math import exp
import matplotlib.pyplot as plt


class Model:
    """
    Model of a shank foot undergoing functional electrical stimulus (FES); a Nonlinear State-Space Model
    """

    def __init__(self, u_profile=np.zeros(351)):
        # Constants mentioned in simulation plan
        self.T_act = 0.01
        self.T_deact = 0.04
        self.J = 0.0197
        self.d = 0.037
        self.B = 0.82
        self.c_F = 0.1145
        self.m_F = 1.0275
        self.a_v = 1.33
        self.f_v1 = 0.18
        self.f_v2 = 0.023
        self.v_max = -0.9
        self.f_max = 600
        self.w = 0.56
        self.l_t = 0.223
        self.l_mt0 = 0.321
        self.l_ce_opt = self.l_mt0 - self.l_t #TODO: Verify this value
        self.l_foot = 0.26 #m
        self.a1 = 2.1
        self.a2 = -0.08
        self.a3 = -7.97
        self.a4 = 0.19
        self.a5 = -1.79
        self.a = [self.a1, self.a2, self.a3, self.a4, self.a5]
        self.g = 9.81

        # External States
        ax = []
        with open('data_files/x_acc_interpolated.csv') as f:
            for line in f:
                ax.append([float(x) for x in list(str(line).strip().split(','))])
        self.ax = ax

        az = []
        with open('data_files/z_acc_interpolated.csv') as f:
            for line in f:
                az.append([float(x) for x in list(str(line).strip().split(','))])
        self.az = az

        a_shank = []
        with open('data_files/shank_angle_interpolated.csv') as f:
            for line in f:
                a_shank.append([float(x) for x in list(str(line).strip().split(','))])
        self.a_shank = a_shank
        
        a_shank1 = []
        with open('data_files/shank_velocity_interpolated.csv') as f:
            for line in f:
                a_shank1.append([float(x) for x in list(str(line).strip().split(','))])
        self.a_shank1 = a_shank1
        
        self.x_external = [self.ax, self.az, self.a_shank, self.a_shank1]

        self.u_profile = u_profile

        self.total_sim_time = 0.35

    def set_u_profile(self, u_profile):
        self.u_profile = u_profile
    
    def get_torque_grav(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Gravity torque of the foot around the ankle
        """
        return -self.m_F*self.c_F*self.g*np.cos(np.deg2rad(x[1]))

    def get_torque_acc(self, x, x_ext):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Torque induced by the movement of the ankle
        """
        return self.m_F*self.c_F*(x_ext[0]*np.sin(np.deg2rad(x[1])) - x_ext[1]*np.cos(np.deg2rad(x[1])))

    def get_torque_ela(self, x):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: Passive elastic torque around the ankle due to passive muscles and tissues
        """
        return np.exp(self.a1 + self.a2 * x[1]) - np.exp(self.a3 + self.a4 * x[1]) + self.a5

    def get_force_fl(self, x, x_ext):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        w, l_ce, l_ce_opt = self.w, self.get_length_ce(x, x_ext), self.l_ce_opt
        return exp((-((l_ce - l_ce_opt) / (w * l_ce_opt))**2))
    
    def get_force_fv(self, x, x_ext):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        v_ce = self.d * (x_ext[3] - x[2])  # contraction speed
        v_max, a_v, f_v1, f_v2 = self.v_max, self.a_v, self.f_v1, self.f_v2
        if v_ce < 0:
            return (1 - v_ce/v_max) / (1 + v_ce/(v_max*f_v1))
        return (1 + a_v*(v_ce/f_v2)) / (1 + v_ce/f_v2)

    def get_force_m(self, x, x_ext):
        f_fl, f_fv = self.get_force_fl(x, x_ext), self.get_force_fv(x, x_ext)
        return x[0] * self.f_max * f_fl * f_fv

    def get_length_mt(self, x, x_ext):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        return self.l_mt0 + (x_ext[2] - x[1])*self.d

    def get_length_ce(self, x, x_ext):
        """
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return:
        """
        return self.get_length_mt(x, x_ext) - self.l_t

    def get_toe_height(self, ankle_angle):
        ankle_data = []
        with open('data_files/ankle_height_interpolated.csv') as f:
            for line in f:
                ankle_data.append([float(x) for x in list(str(line).strip().split(','))])
        ankle_height = [i[1] for i in ankle_data]

        return ankle_height[:len(ankle_angle)] - self.l_foot*np.sin(np.deg2rad(ankle_angle))

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [activation level; foot's absolute orientation wrt horizontal axis; foot's absolute rotational velocity]
        :return: time derivatives of state variables
        """
        u = self.u_profile[int(t*1000)]
        x_ext = [i[int(t*1000)][1] for i in self.x_external]

        return [(u - x[0]) * (u/self.T_act - (1 - u)/self.T_deact),
                x[2],
                (1/self.J) * (self.get_force_m(x, x_ext)*self.d + self.get_torque_grav(x) + self.get_torque_acc(x, x_ext) + self.get_torque_ela(x) + self.B*(x_ext[3]-x[2]))]

    def simulate(self):
        """
        :param total_time: seconds to simulate
        :return: time, state (times at which the state is estimated, state vector at each time)
        """
        # Initial State based off paper
        f_act = 0
        a_f = 0
        a_f1 = -15
        y0 = [f_act, a_f, a_f1]

        t_span = (0, self.total_sim_time)
        dt = 0.001

        sol = solve_ivp(self.get_derivative, t_span, y0, max_step=dt)
        return sol.t, sol.y.T


def plot_graphs(model, time, states):
    # State Graphs
    plt.figure()
    plt.title('States of Foot versus Time')
    plt.subplot(3, 1, 1)
    plt.plot(time, states[:, 0], 'k')
    plt.ylabel('Activation Level')
    plt.subplot(3, 1, 2)
    plt.plot(time, states[:, 1], 'g')
    plt.ylabel('Absolute Orientation wrt Horizontal Axis (deg)')
    plt.subplot(3, 1, 3)
    plt.plot(time, states[:, 2], 'r')
    plt.xlabel('Time(s)')
    plt.ylabel('Absolute Rotational Velocity (deg/s)')
    plt.tight_layout()

    # Toe Height Graph
    toe_height = model.get_toe_height(states[:, 1])
    plt.figure()
    plt.title('Height of Toe versus Time')
    plt.plot(time, toe_height, 'r')
    plt.xlabel('Time(s)')
    plt.ylabel('Toe Height (m)')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # Initiate Model
    model = Model()    
    # Simulate Model
    u_profile_1 = [0 for i in range(351)]
    # u_profile_1 = np.concatenate(([0 for i in range(176)], [0.6 for i in range(175)]))
    model.set_u_profile(u_profile_1)
    t, x = model.simulate()
    plot_graphs(model, t, x)
