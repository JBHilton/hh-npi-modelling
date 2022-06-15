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
hh_prop_max = 100
hh_proptick = hh_prop_min + 0.2 * (hh_prop_max - hh_prop_min) * arange(6)


fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, constrained_layout=True)

axim=ax1.imshow(growth_rate,
                origin='lower',
                extent=(0, 100, 0, 100),
                vmin=r_min,
                vmax=r_max)
ax1.set_ylabel('% uptake of support bubbles')
ax1.text(-50., 110., 'a)',
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
                extent=(0, 100, 0, 100),
                vmin=peak_min,
                vmax=peak_max)
ax2.text(-35., 110., 'b)',
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
                extent=(0, 100, 0, 100),
                vmin=R_end_min,
                vmax=R_end_max)
ax3.set_ylabel('% uptake of support bubbles')
ax3.set_xlabel('% reduction in\n between-household\n transmission')
ax3.text(-50., 110., 'c)',
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
                extent=(0, 100, 0, 100),
                vmin=hh_prop_min,
                vmax=hh_prop_max)
ax4.set_xlabel('% reduction in\n between-household\n transmission')
ax4.text(-35., 110., 'd)',
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

axim = ax1.contour(growth_rate,
                  colors='w',
                  levels=rtick,
                  vmin=r_min,
                  vmax=r_max,
                  extent=(0, 100, 0, 100))
ax1.clabel(axim, fontsize=9, inline=1, fmt='%1.1f')

axim = ax2.contour(peaks,
                  colors='w',
                  levels=peaktick,
                  vmin=peak_min,
                  vmax=peak_max,
                  extent=(0, 100, 0, 100))
ax2.clabel(axim, fontsize=9, inline=1, fmt='%1.1f')

axim = ax3.contour(R_end,
                  colors='w',
                  levels=R_endtick,
                  vmin=R_end_min,
                  vmax=R_end_max,
                  extent=(0, 100, 0, 100))
ax3.clabel(axim, fontsize=9, inline=1, fmt='%1.1f')

axim = ax4.contour(hh_prop,
                  colors='w',
                  levels=hh_proptick,
                  vmin=hh_prop_min,
                  vmax=hh_prop_max,
                  extent=(0, 100, 0, 100))
ax4.clabel(axim, fontsize=9, inline=1, fmt='%1.1f')

fig.savefig('plots/long_term_bubbles/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()
