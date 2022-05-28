import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.use('Qt5Agg')

filter_fun = lambda x: x['Acc'] == 100

dt = pd.read_csv('S1_S2_acc.tsv', sep='\t', names=['S1','S2','Acc'])[filter_fun]

gd = dt.groupby(['S1', 'S2'])['Acc'].count()

print(gd)

#gd.plot(grid=True)

#plt.title('Relation between number of neurons and classification accuracy')
#plt.xlabel('Numbers of S1 and S2 neurons')
#plt.ylabel('Amount of 100% classifications')
#plt.xticks(range(1,20))
#plt.yticks(range(900,1000,5))
#plt.savefig('Test.png', dpi=500)

#plt.show()
