import matplotlib.pyplot as plt
from matplotlib import mpl
from matplotlib.ticker import LinearLocator
import numpy as np
from data import __dataImport as dataImport
import pandas as pd

mpl.use('Qt5Agg')

data = dataImport('epoch_S1_target_acc.csv')




plt.show()
#plt.savefig("epoch_S1_target_acc.png")
