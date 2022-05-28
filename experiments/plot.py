import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd

mpl.use('Qt5Agg')

dt = pd.read_csv('epoch_S1_S2_target_acc.csv', names=['epo', 's1', 's2', 'tar', 'acc'])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = dt.s1
y = dt.s2
z = dt.epo

X=np.arange(1,19+0.1,1,dtype=int)
Y=np.arange(1,19+0.1,1,dtype=int)
X,Y=np.meshgrid(X,Y)
Z = []
for i in range(19):
    tmp=[]
    for j in range(19):
        tmp.append(dt['epo'][i*19+j])
    Z.append(tmp)
#print(Z)
Z=np.array(Z)

surf = ax.plot_surface(X, Y, Z, cmap='plasma',
                       linewidth=0, antialiased=False)
ax.invert_zaxis()


plt.show()
#plt.savefig("epoch_S1_target_acc.png")
