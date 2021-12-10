'''This plots the bubble results
'''
from os import mkdir
from os.path import isdir
from math import ceil, floor
from pickle import dump, load
from numpy import arange, array, atleast_2d, hstack, where, zeros
from matplotlib.pyplot import axes, close, colorbar, imshow, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

if isdir('plots/temp_bubbles') is False:
    mkdir('plots/temp_bubbles')

with open('outputs/temp_bubbles/baseline_results.pkl', 'rb') as f:
    (baseline_peak_data,
     baseline_end_data,
     baseline_ar_data,
     baseline_hh_prop_data) = load(f)

with open('outputs/temp_bubbles/results.pkl', 'rb') as f:
    (peak_data0,
     end_data0,
     ar_data0,
     hh_prop_data0,
     peak_data1,
     end_data1,
     ar_data1,
     hh_prop_data1,
     peak_data2,
     end_data2,
     ar_data2,
     hh_prop_data2,
     peak_data3,
     end_data3,
     ar_data3,
     hh_prop_data3,
     peak_data4,
     end_data4,
     ar_data4,
     hh_prop_data4,
     unmerged_exponents,
     merged_exponents) = load(f)

baseline_peak_array = 0 * end_data0
for i in range(merged_exponents.shape[1]):
    baseline_peak_array[:,i] = baseline_peak_data
baseline_end_array = 0 * end_data0
for i in range(merged_exponents.shape[1]):
    baseline_end_array[:,i] = baseline_end_data

fig, ((ax_0, ax_1),
      (ax_2, ax_3),
      (ax_4, ax_5)) = subplots(3, 2, sharex=True, sharey=True)

fig.tight_layout()

vmin = 0.5*floor(2*100*baseline_peak_array.min())
vmax = 0.5*ceil(2*100*baseline_peak_array.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_0.imshow(100 * baseline_peak_array,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_0.set_ylabel('Single household\n density exponent')
ax_0.text(-1, 1, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_0)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence",
                ticks=vtick)
ax_0.spines['top'].set_visible(False)
ax_0.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin = floor(100*peak_data0.min())
vmax = ceil(100*peak_data0.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_1.imshow(100 * peak_data0,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_1.set_ylabel('Single household\n density exponent')
ax_1.text(-0.6, 1, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_1)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence",
                ticks=vtick)
ax_1.spines['top'].set_visible(False)
ax_1.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*peak_data2.min())
vmax=ceil(100*peak_data2.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_2.imshow(100 * peak_data2,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_2.set_ylabel('Single household\n density exponent')
ax_2.text(-1, 1, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_2)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence",
                ticks=vtick)
ax_2.spines['top'].set_visible(False)
ax_2.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*peak_data1.min())
vmax=ceil(100*peak_data1.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_3.imshow(100 * peak_data1,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_3.set_ylabel('Single household\n density exponent')
ax_3.text(-0.6, 1, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_3)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence",
                ticks=vtick)
ax_3.spines['top'].set_visible(False)
ax_3.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*peak_data4.min())
vmax=ceil(100*peak_data4.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_4.imshow(100 * peak_data4,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_4.set_ylabel('Single household\n density exponent')
ax_4.set_xlabel('Bubbled density\n exponent')
ax_4.text(-1, 1, 'e)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_4)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence",
                ticks=vtick)
ax_4.spines['top'].set_visible(False)
ax_4.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*peak_data3.min())
vmax=ceil(100*peak_data3.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_5.imshow(100 * peak_data3,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_5.set_ylabel('Single household\n density exponent')
ax_5.set_xlabel('Bubbled density\n exponent')
ax_5.text(-0.6, 1, 'f)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_5)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Peak %\n prevalence",
                ticks=vtick)
ax_5.spines['top'].set_visible(False)
ax_5.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

fig.savefig('plots/temp_bubbles/peak_prev_grid.png',bbox_inches='tight', dpi=300)
close()


fig, ((ax_0, ax_1),
      (ax_2, ax_3),
      (ax_4, ax_5)) = subplots(3, 2, sharex=True, sharey=True)

fig.tight_layout()

vmin=floor(100*baseline_end_array.min())
vmax=ceil(100*baseline_end_array.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_0.imshow(100 * baseline_end_array,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_0.set_ylabel('Single household\n density exponent')
ax_0.text(-1, 1, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_0)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence",
                ticks=vtick)
ax_0.spines['top'].set_visible(False)
ax_0.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*end_data0.min())
vmax=ceil(100*end_data0.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_1.imshow(100 * end_data0,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_1.set_ylabel('Single household\n density exponent')
ax_1.text(-0.6, 1, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_1)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence",
                ticks=vtick)
ax_1.spines['top'].set_visible(False)
ax_1.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*end_data2.min())
vmax=ceil(100*end_data2.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_2.imshow(100 * end_data2,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_2.set_ylabel('Single household\n density exponent')
ax_2.text(-1, 1, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_2)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence",
                ticks=vtick)
ax_2.spines['top'].set_visible(False)
ax_2.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*end_data1.min())
vmax=ceil(100*end_data1.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_3.imshow(100 * end_data1,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_3.set_ylabel('Single household\n density exponent')
ax_3.text(-0.6, 1, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_3)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence",
                ticks=vtick)
ax_3.spines['top'].set_visible(False)
ax_3.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*end_data4.min())
vmax=ceil(100*end_data4.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_4.imshow(100 * end_data4,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_4.set_ylabel('Single household\n density exponent')
ax_4.set_xlabel('Bubbled density\n exponent')
ax_4.text(-1, 1, 'e)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_4)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence",
                ticks=vtick)
ax_4.spines['top'].set_visible(False)
ax_4.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

vmin=floor(100*end_data3.min())
vmax=ceil(100*end_data3.max())
vtick = vmin + 0.25 * (vmax - vmin) * arange(5)
axim = ax_5.imshow(100 * end_data3,
                  origin='lower',extent=(0, 1, 0, 1),
                  vmin=vmin,
                  vmax=vmax)
ax_5.set_ylabel('Single household\n density exponent')
ax_5.set_xlabel('Bubbled density\n exponent')
ax_5.text(-0.6, 1, 'f)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax_5)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = colorbar(axim,
                cax=cax,
                orientation='vertical',
                label="Cumulative %\n prevalence",
                ticks=vtick)
ax_5.spines['top'].set_visible(False)
ax_5.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

fig.savefig('plots/temp_bubbles/cum_prev_grid.png',bbox_inches='tight', dpi=300)
close()
