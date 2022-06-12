'''This plots the UK-like model with a single set of parameters for 100 days
'''
from os import mkdir
from os.path import isdir
from pickle import load
from matplotlib.pyplot import subplots
from matplotlib.pyplot import yscale
from matplotlib.cm import get_cmap
from numpy import arange, array, exp, log, where
from model.preprocessing import calculate_sitp
from model.specs import TRANCHE2_SITP

if isdir('plots/uk') is False:
    mkdir('plots/uk')

DOUBLING_TIME = 3
growth_rate = log(2) / DOUBLING_TIME

with open('outputs/uk/fitted_model_input.pkl', 'rb') as f:
    model_input = load(f)

lgd=['Fitted simulation results', 'Exponential growth curve']

def make_exp_growth_plot(time_series, lab, ax, ax_lab, lab_pos):
        t = time_series['time']
        t_7_loc = where(t > 7)[0][0]
        t_30_loc = where(t > 30)[0][0]
        total_cases = (time_series['E'] +
                       time_series['P'] +
                       time_series['I']).sum(1) / model_input.ave_hh_size
        exp_growth_time = arange(-t[t_7_loc], t[t_30_loc]-t[t_7_loc], 2.5)
        exp_curve = total_cases[t_7_loc] * exp(growth_rate * exp_growth_time)

        cmap = get_cmap('tab20')
        alpha = 1
        ax.plot(
                t[:t_30_loc+1],
                total_cases[:t_30_loc+1],
                'k-',
                label=lgd[0],
                alpha=alpha)
        yscale('log')
        ax.plot(
                exp_growth_time+t[t_7_loc],
                exp_curve,
                'k.',
                label=lgd[1],
                alpha=alpha)
        yscale('log')
        ax.set_ylim(0.1 * total_cases[0], 1.0)
        ax.set_box_aspect(1)
        ax.text(lab_pos[0], lab_pos[1], ax_lab,
                    fontsize=12,
                    verticalalignment='top',
                    fontfamily='serif',
                    bbox=dict(facecolor='1', edgecolor='none', pad=3.0))

        return None

imm_label = ['0', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1']
fig, ((ax_0, ax_1),
      (ax_2, ax_3),
      (ax_4, ax_5)) = subplots(3, 2, sharex=True, sharey=True, figsize=(6,8))
axes = [ax_0, ax_1, ax_2, ax_3, ax_4, ax_5]
ax_label = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
label_pos = [[-15, 1], [-10, 1]]

for i in range(6):
        with open('outputs/uk/imm'+imm_label[i]+'_traj.pkl', 'rb') as f:
            H, ts = load(f)
        make_exp_growth_plot(ts, imm_label[i], axes[i], ax_label[i], label_pos[i%2])

ax_4.set_xlabel('Time in days')
ax_5.set_xlabel('Time in days')
ax_0.set_ylabel('Prevalence')
ax_2.set_ylabel('Prevalence')
ax_4.set_ylabel('Prevalence')
ax_1.legend(bbox_to_anchor=(1.05, 1), ncol=1, loc='upper left')
fig.savefig('plots/uk/exp_growth_by_traj.png', bbox_inches='tight', dpi=300)
