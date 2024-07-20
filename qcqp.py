import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

def dynamics(x, u):
    theta = x[2]
    v, omega = u
    Px_dot = v * np.cos(theta)
    Py_dot = v * np.sin(theta)
    theta_dot = omega
    return np.array([Px_dot, Py_dot, theta_dot])

# Lyapunov function
def lyapunov(x, x_d):
    return np.array([[pow(x[0]-x_d[0], 2) + pow(x[1]-x_d[1], 2) + pow(x[2]-x_d[2], 2)]])
    #return np.array([[pow(x[0]-x_d[0], 2) + pow(x[1]-x_d[1], 2) ]])
# Derivative of Lyapunov function
def delta_vxgx(x, x_d):
    dx = x - x_d
    delta_vx = np.array([[2 * dx[0], 2 * dx[1], 2 * dx[2]]])
    gx = np.array([[np.cos(x[2]), 0], [np.sin(x[2]), 0], [0, 1]])
    return delta_vx @ gx

# Define state vector and control input vector
x = np.array([0.0, 0.0, 0.0])  # state vector [x, y, theta]
u = np.array([0.0, 0.0])       # control input vector [v, omega]

# Target state vector
x_d = np.array([0.2, 0.2, np.pi/4])

# CLF-QP parameter
alpha = 0.1
# Simulation parameter
dt = 0.01
T = 200
num_steps = int(T / dt)
states = [x]

for _ in range(num_steps):
    V = lyapunov(x, x_d)
    V_dot = delta_vxgx(x, x_d)

    # CLF-QP config
    H = np.eye(2)
    f = np.zeros(2)
    A = V_dot
    b = -alpha * V

    # Solve QP problem
    u = solve_qp(H, f, A, b, solver='cvxopt')
    print(u)
    # Update state vector
    x_dot = dynamics(x, u)
    x = x + x_dot * dt
    states.append(x)

# Visualization
states = np.array(states)
plt.plot(states[:, 0], states[:, 1], label='Trajectory')
plt.scatter(x_d[0], x_d[1], color='red', label='Target')

# Add yaw angle arrow representation for every 10 steps
# for i in range(0, len(states), 100):
#     plt.quiver(states[i, 0], states[i, 1], np.cos(states[i, 2]), np.sin(states[i, 2]), angles='xy', scale_units='xy', scale=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Car Trajectory')
plt.grid(True)
plt.show()
