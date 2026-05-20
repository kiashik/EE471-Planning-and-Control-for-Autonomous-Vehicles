"""
EE 471 - Prelab 2
Ashik Islam 

Assumptions: These function belongs to QCarEKF class.
        wheelbase parameter exists: self.L
"""
import numpy as np

# ====================== Part A4 ===============================================
def f(self, X, u, dt):
    '''
    Kinematic Bicycle Model:
        - X = [x, y, psi].T
        - u[0] = v (speed in [m/s])
        - u[1] = delta (steering angle in [rad])
        - dt: time step [s]
    Returns: 
        X(K+1) or propagated state vector for next time step in the same format 
        as X.

    '''
    v, delta = u[0], u[1]
    
    x_next = X[0,0] + v * dt * np.cos(X[2,0])
    y_next = X[1,0] + v * dt * np.sin(X[2,0])
    psi_next = X[2,0] + v / self.L * dt * np.tan(delta) 

    X_next = np.array([[x_next], 
                        [y_next], 
                        [psi_next]])   # col vector

    return X_next
# ====================== end A4 ===============================================

# ====================== Part A5 ===============================================
def Jf(self, X, u, dt):
    '''
    Jacobian of motion model with respect to the state.
    Parameters:
        - X = [x, y, psi].T
        - u[0] = v (speed in [m/s])
        - u[1] = delta (steering angle in [rad])
        - dt: time step [s]
    Returns: 
        3x3 NumPy array motion model Jacobian
    '''
    v, delta = u[0], u[1]

    jf0 = np.array([1, 0, -1 * v * dt * np.sin(X[2,0])])
    jf1 = np.array([0, 1, v * dt * np.cos(X[2,0])])
    jf2 = np.array([0, 0, 1])

    Jf = np.vstack((jf0, jf1, jf2))

    return Jf
# ====================== end A5 ===============================================

# ====================== Part B3 ===============================================
def prediction(self, dt, u):
    '''
    Description:
        updates the estimator object in place, so after propagating the state and 
        covariance, it should simply return with no additional output.
        After propagating the state, wrap the heading angle to a principal interval.
    '''
    Fx = self.Jf(self.xHat, u, dt)  # Motion Jacobian wrt states

    self.xHat = self.f(self.xHat, u, dt) # state prediction using non-linear f
    self.P = Fx @ self.P @ Fx.T + self.Q # covariance prediction

    self.xHat[2, 0] = wrap_to_pi(self.xHat[2, 0])  

# ====================== end B3 ===============================================

# ====================== Part C4 ===============================================
# method in GyroKF class
def prediction(self, dt, u):
    '''
    Gyroscope heading/bias prediction
    Parameters:
        dt: time step [s]
        u: most recent gyroscope measurement

    '''

    Ad = np.eye(2) + self.A * dt    # discrete state-space equations
    Bd = self.B * dt

    self.xHat = Ad @ self.xHat + Bd @ u     # state prediction using non-linear f
    self.P = Ad @ self.P @ Ad.T + self.Q # covariance prediction


    self.xHat[0, 0] = wrap_to_pi(self.xHat[0, 0])   # wrap heading angle


# ====================== end C4 ===============================================


