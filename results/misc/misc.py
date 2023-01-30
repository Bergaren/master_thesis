import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
   
def price_distribution_plot():
   dl_results = pd.read_csv('../test_MLP.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0)
   print(dl_results.describe())
   fig, axs = plt.subplots(3, 2, tight_layout=True)
   axs[0,0].set_title('Prediction')
   axs[0,1].set_title('True')
   axs[0,0].hist(dl_results.iloc[:,0], color=c[0], bins=50)
   axs[0,0].set_xlabel('Price spread')
   axs[0,0].set_ylabel('Count')
   axs[0,1].hist(dl_results.iloc[:,1], color=c[1], bins=50)
   axs[0,1].set_xlabel('Price spread')
   axs[0,1].set_ylabel('Count')
   #plt.hist(dl_results.iloc[:,0], color=c[0], bins=20, label='Prediction')
   axs[1,0].boxplot(dl_results.iloc[:,0],showfliers=False, vert=False)
   axs[1,0].set_xlabel('Price spread')
   axs[1,1].boxplot(dl_results.iloc[:,1], showfliers=False, vert=False)
   axs[1,1].set_xlabel('Price spread')

def cap_dist_plot():
   cap  = pd.read_csv('../../data/elspot_dayahead/cap.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)

   fig = plt.figure(constrained_layout=True)
   gs = GridSpec(3, 2, figure=fig)

   ax1 = fig.add_subplot(gs[0,0])
   ax2 = fig.add_subplot(gs[0,1])
   ax3 = fig.add_subplot(gs[1,0])
   ax4 = fig.add_subplot(gs[1,1])
   ax5 = fig.add_subplot(gs[2,0])

   ax1.hist(cap.iloc[:,0], bins=50, label='Cap SE2 > SE3', color=c[0])
   ax1.set_xlabel('MW')
   ax1.set_ylabel('Count')
   ax1.set_title('Cap SE2 > SE3')
   ax2.hist(cap.iloc[:,1], bins=50, label='Cap SE4 > SE3', color=c[1])
   ax2.set_xlabel('MW')
   ax2.set_ylabel('Count')
   ax2.set_title('Cap SE4 > SE3')
   ax3.hist(cap.iloc[:,2], bins=50, label='Cap NO1 > SE3', color=c[2])
   ax3.set_xlabel('MW')
   ax3.set_ylabel('Count')
   ax3.set_title('Cap NO1 > SE3')
   ax4.hist(cap.iloc[:,3], bins=50, label='Cap DK1 > SE3', color=c[3])
   ax4.set_xlabel('MW')
   ax4.set_ylabel('Count')
   ax4.set_title('Cap DK1 > SE3')
   ax5.hist(cap.iloc[:,4], bins=50, label='Cap FI > SE3', color=c[4])
   ax5.set_xlabel('MW')
   ax5.set_ylabel('Count')
   ax5.set_title('Cap FI > SE3')

cap_dist_plot()
plt.show()