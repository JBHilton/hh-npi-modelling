'''This plots the results of the parameter sweep for the long-term bubbling
example.
'''
from os import mkdir
from os.path import isdir
from pickle import load
from math import ceil, floor
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn import heatmap

if isdir('plots/long_term_bubbles') is False:
    mkdir('plots/long_term_bubbles')

with open('outputs/long_term_bubbles/results.pkl','rb') as f:
    (growth_rate,
     peaks,
     R_end,
     hh_prop,
     attack_ratio,
     bubble_prob_range,
     external_mix_range) = load(f)

r_min = 0.01 * floor(100 * growth_rate.min())
r_max = 0.01 * floor(100 * growth_rate.max())
rtick = r_min + 0.2 * (r_max - r_min) * arange(6)
peak_min = floor(peaks.min())
peak_max = 5 * ceil(peaks.max() / 5)
peaktick = peak_min + 0.2 * (peak_max - peak_min) * arange(6)
R_end_min = floor(R_end.min())
R_end_max = 10 * ceil(R_end.max() / 10)
R_endtick = R_end_min + 0.2 * (R_end_max - R_end_min) * arange(6)
hh_prop_min = floor(hh_prop.min())
hh_prop_max = 5 * ceil(hh_prop.max() / 5)
hh_proptick = hh_prop_min + 0.2 * (hh_prop_max - hh_prop_min) * arange(6)
attack_ratio_min = attack_ratio.min()
attack_ratio_max = attack_ratio.max()


fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, constrained_layout=True)

axim=ax1.imshow(growth_rate,
                origin='lower',
                extent=(0,1,0,1),
                vmin=r_min,
                vmax=r_max)
ax1.set_ylabel('% uptake of support bubbles')
ax1.text(-0.5, 1.1, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Growth rate",
                cax=cax,
                ticks=rtick)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

axim=ax2.imshow(peaks,
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax2.text(-0.3, 1.1, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Peak % prevalence",
                cax=cax,
                ticks=peaktick)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

axim=ax3.imshow(R_end,
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax3.set_ylabel('% uptake of support bubbles')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax3.text(-0.5, 1.1, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Cumulative % prevalence",
                cax=cax,
                ticks=R_endtick)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

axim=ax4.imshow(hh_prop,
                origin='lower',
                extent=(0,1,0,1),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax4.set_xlabel('% reduction in\n between-household\n transmission')
ax4.text(-0.3, 1.1, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="% of households infected",
                cax=cax,
                ticks=hh_proptick)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

fig.savefig('plots/long_term_bubbles/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ((ax1, ax2)) = subplots(1, 2)
axim=ax1.imshow(peaks,
                origin='lower',
                extent=(0,1,0,1),
                vmin=peak_min,
                vmax=peak_max)
ax1.set_xlabel('Proportional reduction in\n between-household\n transmission')
ax1.set_ylabel('Proportional uptake of support\n bubbles among\n elligible households')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("top", size="5%", pad=0.05)
cbar = colorbar(axim, label="Peak % prevalence", orientation='horizontal', cax=cax)
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

axim=ax2.imshow(R_end,
                origin='lower',
                extent=(0,1,0,1),
                vmin=R_end_min,
                vmax=R_end_max)
ax2.set_xlabel('Proportional reduction in\n between-household\n transmission')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("top", size="5%", pad=0.05)
cbar = colorbar(axim, label="Cumulative % prevalence", orientation='horizontal', cax=cax)
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

fig.savefig('plots/long_term_bubbles/poster_plot.png',
            bbox_inches='tight',
            dpi=300)
close()
