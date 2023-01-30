import numpy as np
from PyEMD import EMD, CEEMDAN, Visualisation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vmdpy import VMD
import time

'''
   In price and load forecasting mode decomposition was reported to enhance the performance.
   However, all papers that was read had incorporated a lookahead bias in the process. 
   When this was taken into account superior performance with mode decomposition could not be found.
'''

days = 5


def standard_score(df):
    for column in ['Dayahead SE3', 'Dayahead SE4', 'Intraday SE3', 'Intraday SE4']:
        std = df[column].std()
        mean = df[column].mean()

        df['{} (s)'.format(column)] = (df[column] - mean) / std
    return df


df = pd.read_csv('data/price_data.csv', encoding="ISO-8859-1", sep=',',
                 decimal='.', index_col='Delivery', parse_dates=['Delivery'])
colors = sns.color_palette("tab10")

""" emd = EMD()
emd.emd(df['Dayahead SE3'].to_numpy()[:days*24], max_imf=5)
imfs, res = emd.get_imfs_and_residue()
fig1, axs = plt.subplots(len(imfs)+1, 2)

axs[0,0].plot(df['Dayahead SE3'].iloc[:days*24])
axs[0,1].title.set_text('Price')

for n, imf in enumerate(imfs):
   axs[n+1,0].plot(imf, c=colors[n+1])
   axs[n+1,0].title.set_text('IMF {}'.format(n+1)) """

alpha = 2000       # moderate bandwidth constraint
tau = 0.           # noise-tolerance (no strict fidelity enforcement)
K = 2             # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-5

# . Run actual VMD code
""" us, u_hat, omega = VMD(df['Dayahead SE3'].iloc[:days*24].to_numpy(), alpha, tau, K, DC, init, tol)

axs[0,0].plot(df['Dayahead SE3'].iloc[:days*24])
axs[0,0].title.set_text('Price')
for n, u in enumerate(us):
   axs[n+1,0].plot(u, c=colors[n+1])
   axs[n+1,0].title.set_text('VMD {}'.format(n+1)) """

ceemdan = CEEMDAN()
ceemdan.ceemdan(df['Dayahead SE3'].to_numpy()[:days*24], max_imf=5)
imfs, res = ceemdan.get_imfs_and_residue()

fig1, axs = plt.subplots(len(imfs)+1, 1)

axs[0].plot(df['Dayahead SE3'].iloc[:days*24])
axs[0].title.set_text('Price')

for n, imf in enumerate(imfs):
    axs[n+1].plot(imf, c=colors[n+1])
    axs[n+1].set_ylabel('IMF {}'.format(n+1))

plt.show()
