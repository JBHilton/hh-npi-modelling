'''This plots the mixing sweep results
'''
from os import mkdir
from os.path import isdir
from math import ceil, floor
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn import heatmap

if isdir('plots/mixing_sweep') is False:
    mkdir('plots/mixing_sweep')

with open('outputs/mixing_sweep/results.pkl','rb') as f:
    (growth_rate,
    peaks,
    R_end,
    hh_prop,
    attack_ratio,
    ar_by_size,
    first_pass_ar,
    internal_mix_range,
    external_mix_range) = load(f)

internal_mix_len = len(internal_mix_range)
external_mix_len = len(external_mix_range)

r_min = 0.01 * floor(100 * growth_rate.min())
r_max = 0.01 * floor(100 * growth_rate.max())
rtick = r_min + 0.25 * (r_max - r_min) * arange(5)
peak_min = floor(peaks.min())
peak_max = 25
peaktick = peak_min + 0.25 * (peak_max - peak_min) * arange(5)
R_end_min = floor(R_end.min())
R_end_max = 10 * ceil(R_end.max() / 10)
R_endtick = R_end_min + 0.25 * (R_end_max - R_end_min) * arange(5)
hh_prop_min = floor(hh_prop.min())
hh_prop_max = 10 * ceil(hh_prop.max() / 10)
hh_proptick = hh_prop_min + 0.25 * (hh_prop_max - hh_prop_min) * arange(5)
attack_ratio_min = attack_ratio.min()
attack_ratio_max = attack_ratio.max()

# fig, ax = subplots(1, 1, sharex=True)
# axim=ax.imshow(growth_rate,
#                 origin='lower',
#                 extent=(0,1,0,1),
#                 vmin=r_min,
#                 vmax=r_max)
# ax.set_ylabel('% reduction in\n within-household\n transmission')
# ax.set_xlabel('% reduction in\n between-household\n transmission')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(axim, label="Growth rate", cax=cax)
#
# fig.savefig('plots/mixing_sweep/growth_rate.png',
#             bbox_inches='tight',
#             dpi=300)
# close()
#
# fig, ax = subplots(1, 1)
# axim=ax.imshow(peaks,
#                 origin='lower',
#                 extent=(0,1,0,1),
#                 vmin=peak_min,
#                 vmax=peak_max)
# ax.set_ylabel('% reduction in\n within-household\n transmission')
# ax.set_xlabel('% reduction in\n between-household\n transmission')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(axim, label="Peak % prevalence", cax=cax)
#
# fig.savefig('plots/mixing_sweep/peaks.png',
#             bbox_inches='tight',
#             dpi=300)
# close()
#
# fig, ax = subplots(1, 1)
# axim=ax.imshow(R_end,
#                 origin='lower',
#                 extent=(0,1,0,1),
#                 vmin=R_end_min,
#                 vmax=R_end_max)
# ax.set_ylabel('% reduction in\n within-household\n transmission')
# ax.set_xlabel('% reduction in\n between-household\n transmission')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(axim, label="% population immunity at end of simulation", cax=cax)
#
# fig.savefig('plots/mixing_sweep/R_end.png',
#             bbox_inches='tight',
#             dpi=300)
# close()
#
# fig, ax = subplots(1, 1)
# axim=ax.imshow(hh_prop,
#                 origin='lower',
#                 extent=(0,1,0,1),
#                 vmin=hh_prop_min,
#                 vmax=hh_prop_max)
# ax.set_ylabel('% reduction in\n within-household\n transmission')
# ax.set_xlabel('% reduction in\n between-household\n transmission')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(axim, label="% of households infected", cax=cax)
#
# fig.savefig('plots/mixing_sweep/hh_prop.png',
#             bbox_inches='tight',
#             dpi=300)
# close()
#
# fig, ax = subplots(1, 1)
# axim=ax.imshow(attack_ratio,
#                 origin='lower',
#                 extent=(0,1,0,1),
#                 vmin=attack_ratio_min,
#                 vmax=attack_ratio_max)
# ax.set_ylabel('% reduction in\n within-household\n transmission')
# ax.set_xlabel('% reduction in\n between-household\n transmission')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(axim, label="% attack ratio", cax=cax)
#
# fig.savefig('plots/mixing_sweep/attack_ratio.png',
#             bbox_inches='tight',
#             dpi=300)
# close()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, constrained_layout=True)

axim=ax1.imshow(growth_rate,
                origin='lower',
                extent=(0,1,0,1),
                vmin=r_min,
                vmax=r_max)
ax1.set_ylabel('% reduction in\n within-household\n transmission')
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
ax3.set_ylabel('% reduction in\n within-household\n transmission')
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

fig.savefig('plots/mixing_sweep/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()

baseline_ar = ar_by_size[0,0,:] # AR by size with no interventions
ar_int_25 = ar_by_size[5,0,:] # AR with 25% internal mixing reduction
ar_ext_25 = ar_by_size[0,5,:] # AR with 25% external reduction
ar_both_25 = ar_by_size[5,5,:] # AR with both interventions
ar_int_50 = ar_by_size[10,0,:] # AR with 50% internal mixing reduction
ar_ext_50 = ar_by_size[0,10,:] # AR with 50% external reduction
ar_both_50 = ar_by_size[10,10,:] # AR with both interventions

# fig, ax = subplots(1,1)
# ax.plot(range(1,7), baseline_ar, '-o', label = 'No interventions')
# ax.plot(range(1,7), ar_int_50, label = '50% within-hh reduction', linestyle='dotted', marker='s')
# ax.plot(range(1,7), ar_ext_50, label = '50% between-hh reduction', linestyle='dashed', marker='v')
# ax.plot(range(1,7), ar_both_50, label = '50% reduction on both levels', linestyle='dashdot', marker='X')
# ax.set_xlabel('Household size')
# ax.set_ylabel('Expected secondary\n attack ratio')
# ax.set_aspect(1.0/ax.get_data_ratio())
# ax.legend()
#
# fig.savefig('plots/mixing_sweep/ar_by_size_50.png')
#
# close()

baseline_fpar = first_pass_ar[0,0,:] # AR by size with no interventions
fpar_int_25 = first_pass_ar[5,0,:] # AR with 25% internal mixing reduction
fpar_int_50 = first_pass_ar[10,0,:] # AR with 50% internal mixing reduction

# fig, ax = subplots(1,1)
# ax.plot(range(1,7), baseline_fpar, '-o', label = 'No interventions')
# ax.plot(range(1,7), fpar_int_50, label = '50% within-hh reduction', linestyle='dotted', marker='s')
# ax.set_xlabel('Household size')
# ax.set_ylabel('Expected secondary\n attack ratio')
# ax.set_aspect(1.0/ax.get_data_ratio())
# ax.legend()
#
# fig.savefig('plots/mixing_sweep/fpar_by_size_50.png')
#
# close()

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2)
fig.tight_layout()
ax1.plot(range(1,7), baseline_ar, '-o', label = 'No interventions')
ax1.plot(range(1,7), ar_int_25, ':s', label = 'Within-hh controls')
ax1.plot(range(1,7), ar_ext_25, '--v', label = 'Between-hh controls')
ax1.plot(range(1,7), ar_both_25, '-.x', label = 'Controls on both levels')
ax1.set_xlabel('Household size')
ax1.set_ylabel('Expected secondary\n attack ratio')
ax1.set_aspect(1.0/ax1.get_data_ratio())
ax1.text(-1.5, 1.0, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))

ax2.plot(range(1,7), baseline_ar, '-o', label = 'No interventions')
ax2.plot(range(1,7), ar_int_50, ':s', label = 'Within-hh controls')
ax2.plot(range(1,7), ar_ext_50, '--v', label = 'Between-hh controls')
ax2.plot(range(1,7), ar_both_50, '-.x', label = 'Controls on both levels')
ax2.set_xlabel('Household size')
ax2.set_ylabel('Expected secondary\n attack ratio')
ax2.set_aspect(1.0/ax2.get_data_ratio())
ax2.text(-1.0, 1.0, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax3.plot(range(1,7), baseline_fpar, '-o', label = 'No interventions')
ax3.plot(range(1,7), fpar_int_25, ':s', label = 'Within-hh controls')
ax3.set_xlabel('Household size')
ax3.set_ylabel('Expected first\n pass secondary\n attack ratio')
ax3.set_aspect(1.0/ax3.get_data_ratio())
ax3.text(-1.5, 0.8, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))

ax4.plot(range(1,7), baseline_fpar, '-o', label = 'No interventions')
ax4.plot(range(1,7), fpar_int_50, ':s', label = 'Within-hh controls')
ax4.set_xlabel('Household size')
ax4.set_ylabel('Expected first\n pass secondary\n attack ratio')
ax4.set_aspect(1.0/ax4.get_data_ratio())
ax4.text(-1.0, 0.8, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))

fig.savefig('plots/mixing_sweep/ar_grid.png',
            bbox_inches='tight',
            dpi=300)
close()
