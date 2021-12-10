'''This plots the UK-like model with a single set of parameters for 100 days
'''
from os import mkdir
from os.path import isdir
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.pyplot import yscale
from matplotlib.cm import get_cmap
from numpy import exp, log, where

if isdir('plots/uk') is False:
    mkdir('plots/uk')

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

with open('outputs/uk/results.pkl', 'rb') as f:
    H, time_series = load(f)
with open('outputs/uk/fitted_model_input.pkl', 'rb') as f:
    model_input = load(f)

t = time_series['time']
t_15_loc = where(t > 15)[0][0]
t_30_loc = where(t > 30)[0][0]
total_cases = (time_series['E'] +
               time_series['P'] +
               time_series['I']).sum(1) / model_input.ave_hh_size
exp_curve = total_cases[t_15_loc] * exp(growth_rate * (t[:t_30_loc+1]-t[t_15_loc]))

lgd=['Fitted simulation results', 'Exponential growth curve']

fig, ax = subplots(1, 1, sharex=True)

cmap = get_cmap('tab20')
alpha = 0.5
ax.plot(
        t[:t_30_loc+1],
        total_cases[:t_30_loc+1],
        label=lgd[0],
        alpha=alpha)
yscale('log')
ax.plot(
        t[:t_30_loc+1],
        exp_curve,
        label=lgd[1],
        alpha=alpha)
yscale('log')
ax.set_ylabel('Time in days')
ax.set_ylabel('Prevalence')
ax.set_ylim(0.1 * total_cases[0], total_cases[0] * exp(growth_rate * t[t_30_loc]))
ax.legend(ncol=1, loc='upper left')
ax.set_box_aspect(1)
fig.savefig('plots/uk/exp_growth.png', bbox_inches='tight', dpi=300)
