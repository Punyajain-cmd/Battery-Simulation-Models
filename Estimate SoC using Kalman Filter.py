import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 1.0           # Time step (s)
t_total = 3600     # 1 hour
Q = 2.5            # Capacity (Ah)
Voc0 = 4.2         # Nominal OCV (V)
Rint = 0.05        # Internal resistance (Ohm)

# Arrays
t = np.arange(0, t_total, dt)
I_true = np.random.normal(2.0, 0.1, len(t))   # True current
V_meas = Voc0 - I_true*Rint + np.random.normal(0, 0.01, len(t))  # Measured voltage (with noise)

# Initialize Kalman filter
SoC_est = np.zeros_like(t)
P = 1.0
R = 0.01   # Measurement noise
Q_kf = 0.0001   # Process noise
SoC_est[0] = 1.0

for k in range(1, len(t)):
    # Prediction
    SoC_pred = SoC_est[k-1] - (I_true[k-1]*dt) / (Q*3600)
    P = P + Q_kf

    # Update
    K = P / (P + R)
    SoC_est[k] = SoC_pred + K * ( (V_meas[k] - (Voc0 - (1-SoC_pred)*0.5)) - 0 )
    P = (1 - K) * P

# Plot
plt.plot(t/60, SoC_est)
plt.title('State of Charge Estimation with Kalman Filter')
plt.xlabel('Time (minutes)')
plt.ylabel('Estimated SoC')
plt.grid()
plt.show()
