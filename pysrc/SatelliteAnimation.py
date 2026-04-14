import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t = np.linspace(0, 10, 100)  # Time array for one full orbit
y = np.sin(t)  # Simulated distance between satellites (oscillating)

fig, axis = plt.subplots()

axis.set_xlim([min(t), max(t)])
axis.set_ylim([-2,2])
animated_plot, = axis.plot([], [] )


def update_data(frame):
    animated_plot.set_data(t[:frame], y[:frame])

    
    return animated_plot,
    

animation = FuncAnimation(fig=fig, func=update_data, frames=len(t),interval=25)

plt.show()
