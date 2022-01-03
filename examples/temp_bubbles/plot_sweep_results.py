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
      (ax_4, ax_5)) = subplots(3, 2, sharex=True, sharey=True, figsize=(6,8))

vmin_bl = 0.5*floor(2*100*baseline_peak_array.min())
vmax_bl = 0.5*ceil(2*100*baseline_peak_array.max())
vmin_0 = 0.5*floor(2*100*peak_data0.min())
vmax_0 = 0.5*ceil(2*100*peak_data0.max())
vmin_2 = 0.5*floor(2*100*peak_data2.min())
vmax_2 = 0.5*ceil(2*100*peak_data2.max())
vmin_1 = 0.5*floor(2*100*peak_data1.min())
vmax_1 = 0.5*ceil(2*100*peak_data1.max())
vmin_4 = 0.5*floor(2*100*peak_data4.min())
vmax_4 = 0.5*ceil(2*100*peak_data4.max())
vmin_3 = 0.5*floor(2*100*peak_data3.min())
vmax_3 = 0.5*ceil(2*100*peak_data3.max())

vmin = min(vmin_bl, vmin_0, vmin_1, vmin_2, vmin_3, vmin_4)
vmax = max(vmax_bl, vmax_0, vmax_1, vmax_2, vmax_3, vmax_4)
vtick = arange(vmin, vmax+0.5, 0.5)

axim = ax_0.imshow(100 * baseline_peak_array,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_0.set_ylabel('Single household\n density exponent', fontsize=12)
ax_0.text(-0.5, 1, 'a)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_0.spines['top'].set_visible(False)
ax_0.spines['right'].set_visible(False)

axim = ax_1.imshow(100 * peak_data0,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_1.text(-0.2, 1, 'b)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_1.spines['top'].set_visible(False)
ax_1.spines['right'].set_visible(False)

axim = ax_2.imshow(100 * peak_data2,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_2.set_ylabel('Single household\n density exponent', fontsize=12)
ax_2.text(-0.5, 1, 'c)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_2.spines['top'].set_visible(False)
ax_2.spines['right'].set_visible(False)

axim = ax_3.imshow(100 * peak_data1,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_3.text(-0.2, 1, 'd)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_3.spines['top'].set_visible(False)
ax_3.spines['right'].set_visible(False)

axim = ax_4.imshow(100 * peak_data4,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_4.set_ylabel('Single household\n density exponent', fontsize=12)
ax_4.set_xlabel('Bubbled density\n exponent', fontsize=12)
ax_4.text(-0.5, 1, 'e)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_4.spines['top'].set_visible(False)
ax_4.spines['right'].set_visible(False)

axim = ax_5.imshow(100 * peak_data3,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_5.set_xlabel('Bubbled density\n exponent', fontsize=12)
ax_5.text(-0.2, 1, 'f)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_5.spines['top'].set_visible(False)
ax_5.spines['right'].set_visible(False)

cax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
cbar = fig.colorbar(axim,
                cax=cax,
                orientation='vertical',
                ticks=vtick)
cbar.set_label("Peak %\n prevalence", fontsize=12)
cbar.outline.set_visible(False)

fig.savefig('plots/temp_bubbles/peak_prev_grid.png',bbox_inches='tight', dpi=300)
close()

fig, ((ax_0, ax_1),
      (ax_2, ax_3),
      (ax_4, ax_5)) = subplots(3, 2, sharex=True, sharey=True, figsize=(6,8))

vmin_bl = 0.5*floor(2*100*baseline_end_array.min())
vmax_bl = 0.5*ceil(2*100*baseline_end_array.max())
vmin_0 = 0.5*floor(2*100*end_data0.min())
vmax_0 = 0.5*ceil(2*100*end_data0.max())
vmin_2 = 0.5*floor(2*100*end_data2.min())
vmax_2 = 0.5*ceil(2*100*end_data2.max())
vmin_1 = 0.5*floor(2*100*end_data1.min())
vmax_1 = 0.5*ceil(2*100*end_data1.max())
vmin_4 = 0.5*floor(2*100*end_data4.min())
vmax_4 = 0.5*ceil(2*100*end_data4.max())
vmin_3 = 0.5*floor(2*100*end_data3.min())
vmax_3 = 0.5*ceil(2*100*end_data3.max())

vmin = min(vmin_bl, vmin_0, vmin_1, vmin_2, vmin_3, vmin_4)
vmax = max(vmax_bl, vmax_0, vmax_1, vmax_2, vmax_3, vmax_4)
vtick = arange(vmin, vmax+0.5, 0.5)

axim = ax_0.imshow(100 * baseline_end_array,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_0.set_ylabel('Single household\n density exponent', fontsize=12)
ax_0.text(-0.5, 1, 'a)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_0.spines['top'].set_visible(False)
ax_0.spines['right'].set_visible(False)

axim = ax_1.imshow(100 * end_data0,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_1.text(-0.2, 1, 'b)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_1.spines['top'].set_visible(False)
ax_1.spines['right'].set_visible(False)

axim = ax_2.imshow(100 * end_data2,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_2.set_ylabel('Single household\n density exponent', fontsize=12)
ax_2.text(-0.5, 1, 'c)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_2.spines['top'].set_visible(False)
ax_2.spines['right'].set_visible(False)

axim = ax_3.imshow(100 * end_data1,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_3.text(-0.2, 1, 'd)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_3.spines['top'].set_visible(False)
ax_3.spines['right'].set_visible(False)

axim = ax_4.imshow(100 * end_data4,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_4.set_ylabel('Single household\n density exponent', fontsize=12)
ax_4.set_xlabel('Bubbled density\n exponent', fontsize=12)
ax_4.text(-0.5, 1, 'e)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_4.spines['top'].set_visible(False)
ax_4.spines['right'].set_visible(False)

axim = ax_5.imshow(100 * end_data3,
                  origin='lower',
                  vmin=vmin,
                  vmax=vmax,
                  extent=(0, 1, 0, 1),
                  aspect=1)
ax_5.set_xlabel('Bubbled density\n exponent', fontsize=12)
ax_5.text(-0.2, 1, 'f)',
            fontsize=12,
            verticalalignment='top',
            fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
ax_5.spines['top'].set_visible(False)
ax_5.spines['right'].set_visible(False)

cax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
cbar = fig.colorbar(axim,
                cax=cax,
                orientation='vertical',
                ticks=vtick)
cbar.set_label("Cumulative %\n prevalence", fontsize=12)
cbar.outline.set_visible(False)

fig.savefig('plots/temp_bubbles/cum_prev_grid.png',bbox_inches='tight', dpi=300)
close()
