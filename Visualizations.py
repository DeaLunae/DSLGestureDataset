import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="magma")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 14
nmax=60
x = np.linspace(-4,4,N+1)
x, y = np.meshgrid(x, x)
zarray = np.zeros((N+1, N+1, nmax))

f = lambda x,y,sig : 1/np.sqrt(sig)*np.exp(-(x**2+y**2)/sig**2)

for i in range(nmax):
    zarray[:,:,i] = f(x,y,1.5+np.sin(i*2*np.pi/nmax))

plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0,1.5)
anim = animation.FuncAnimation(fig, update_plot, nmax, fargs=(zarray, plot),interval=1000)
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'],progress_callback=lambda i, n: print(f'\rSaving frame {i} of {n}', end=''))
plt.show()