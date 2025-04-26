from QuantLib import *
import numpy as np
import matplotlib.pyplot as plt
import utils
# % matplotlib inline  # Removed as it is a Jupyter Notebook magic command


sigma = 0.015
a = 0.1
timestep = 360
length = 30 #years
forward_rate = 0.05
day_count = Actual360()
todays_date = Date(26, 4, 2025)


Settings.instance().evaluation_date = todays_date

spot_curve = FlatForward(todays_date,
                         QuoteHandle(SimpleQuote(forward_rate)),
                         day_count)

spot_curve_handle = YieldTermStructureHandle(spot_curve)

hw_process = HullWhiteProcess(spot_curve_handle, a, sigma)

rng = GaussianRandomSequenceGenerator(
    UniformRandomSequenceGenerator(timestep, UniformRandomGenerator()))
seq = GaussianPathGenerator(hw_process, length, timestep, rng, False)

def generate_paths(num_paths, timestep):
    arr = np.zeros((num_paths, timestep+1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr

num_paths = 10
time, paths = generate_paths(num_paths, timestep)
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(num_paths):
    ax.plot(time, paths[i, :], lw=0.8, alpha=0.6)
ax.set_title("Hull-White Short Rate Simulation")
plt.savefig("hull_white_simulation.png", dpi=300, bbox_inches='tight')

plt.show()

# Plotting the variance of the short rates over time:
num_paths = 1000
time, paths = generate_paths(num_paths, timestep)
vol = [np.var(paths[:, i]) for i in range(timestep+1)]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, vol, "-.", lw=3, alpha=0.6)
ax.plot(time,sigma*sigma/(2*a)*(1.0-np.exp(-2.0*a*np.array(time))), "-", lw=2, alpha=0.5)
ax.set_title("Variance of Short Rates", size=14)
plt.savefig("hull_white_variance.png", dpi=300, bbox_inches='tight')
plt.show()

# Plotting the average of the short rates over time:

def alpha(forward, sigma, a,t):
    return forward + 0.5* np.power(sigma/a*(1.0 - np.exp(-a*t)), 2)
    

avg = [np.mean(paths[:, i]) for i in range(timestep+1)]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, avg, "-.", lw=3, alpha=0.6)
ax.plot(time, alpha(forward_rate, sigma, a, np.array(time)), "-", lw=2, alpha=0.5)
ax.set_title("Average of Short Rates", size=14)
plt.savefig("hull_white_average.png", dpi=300, bbox_inches='tight')
plt.show()
