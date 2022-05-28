import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.use('Qt5Agg')

dt = pd.read_csv('epoch_S1_target_acc.csv', names=['epo','s1','tar','acc'])

#gd = dt.groupby(['S1', 'S2'])['Acc'].count()

print(dt)

dt[['s1', 'epo']].plot(grid=True)

plt.title('Relation between number of neurons and classification efficiency')
plt.ylabel('Amount of epochs needed for correct classification')
plt.xlabel('Amount of S1 neurons')
plt.xticks(range(1,20))
plt.yticks(range(0,135,10))
plt.savefig('S1_acc.png', dpi=500)

plt.show()
