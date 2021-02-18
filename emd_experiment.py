import numpy as np
from PyEMD import EMD, Visualisation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vmdpy import VMD

def standard_score(df):
   for column in ['Dayahead SE3', 'Dayahead SE4', 'Intraday SE3', 'Intraday SE4']:
      std = df[column].std()
      mean = df[column].mean()

      df['{} (s)'.format(column)] = (df[column] - mean) / std
   return df

df = pd.read_csv('data/price_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
df = standard_score(df)

alpha = 2000       # moderate bandwidth constraint  
tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  


#. Run actual VMD code  
us, u_hat, omega = VMD(df['Dayahead SE3 (s)'].to_numpy()[:24], alpha, tau, K, DC, init, tol)
colors = sns.color_palette("tab10")

fig1, axs = plt.subplots(len(us)+1, 2)
axs[0,0].plot(df['Dayahead SE3'][:24])
axs[0,0].title.set_text('Price')
for n, u in enumerate(us):
   axs[n+1,0].plot(u, c=colors[n+1])
   axs[n+1,0].title.set_text('VMD {}'.format(n+1))


""" emd = EMD()
emd.emd(df['Dayahead SE3 (s)'].to_numpy()[:100*24], max_imf=3)
imfs, res = emd.get_imfs_and_residue()


N = len(imfs)
colors = sns.color_palette("tab10")

axs[0,1].plot(df['Dayahead SE3'][:100*24])
axs[0,1].title.set_text('Price')

for n, imf in enumerate(imfs):
   axs[n+1,1].plot(imf, c=colors[n+1])
   axs[n+1,1].title.set_text('IMF {}'.format(n+1)) """

plt.show()
