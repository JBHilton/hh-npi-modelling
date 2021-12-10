''' This creates a scatter plot of input vs output values of beta_ext.'''

from os import mkdir
from os.path import isdir
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.pyplot import yscale
from matplotlib.cm import get_cmap
from numpy import exp, log, where

if isdir('plots/between_hh_fitting') is False:
    mkdir('plots/between_hh_fitting')

with open('outputs/between_hh_fitting/results.pkl', 'rb') as f:
    beta_in, beta_out = load(f)

fig, ax = subplots(1, 1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
ax.plot(
        beta_in,
        beta_out,
        '.',
        label='Data',
        alpha=alpha)
ax.plot(
        [beta_in.min(), beta_in.max()],
        [beta_in.min(), beta_in.max()],
        label='y=x',
        alpha=alpha)

ax.set_xlabel('Input value')
ax.set_ylabel('Estimated value')
ax.legend(ncol=1, loc='upper left')
ax.set_box_aspect(1)
fig.savefig('plots/between_hh_fitting/scatter.png', bbox_inches='tight', dpi=300)
