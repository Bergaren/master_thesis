import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import numpy as np

c = sns.color_palette("tab10")
#c = ['#DAE8FC','#D5E8D4', '#FFE6CC', '#FFF2CC', '#F8CECC', '#E1D5E7']

def production_pie():
   data = {
      'Hydro'     : 64.6,
      'Nuclear'   : 64.3,
      'Wind'      : 19.9,
      'Solar'     : 0.5,
      'Other'     : 15.1
   }
   pie, ax = plt.subplots()
   labels = data.keys()
   plt.pie(x=data.values(), autopct="%.1f%%", explode=[0.05]*len(data), labels=labels, pctdistance=0.5, colors=c, textprops={'fontsize': 11})

def loss_plot():   
   # evenly sampled time at 200ms intervals
   e = np.arange(-2.5, 2.51, 0.1)
   # red dashes, blue squares and green triangles
   plt.plot(e, np.absolute(e), c=c[0], label="MAE")
   plt.plot(e, e**2, c=c[1], label="MSE")
   plt.plot(e, [Huber(x) for x in e], c=c[2], label="Huber")
   
   plt.xlabel('Error')
   plt.ylabel('Loss')

def Huber(x):
   delta = 1
   if np.abs(x) > delta:
      return delta*(np.abs(x) - 0.5*delta)
   else:
      return 0.5*(x**2)


loss_plot()
plt.legend()
plt.show()