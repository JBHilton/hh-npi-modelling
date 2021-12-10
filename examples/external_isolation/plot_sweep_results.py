'''This plots the results of the parameter sweep for the OOHI
example.
'''
from os import mkdir
from os.path import isdir
from math import ceil, floor
from pickle import load
from numpy import arange, array, atleast_2d, hstack, sum, where, zeros
from matplotlib.pyplot import close, colorbar, imshow, set_cmap, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

if isdir('plots/oohi') is False:
    mkdir('plots/oohi')

with open('outputs/oohi/results.pkl','rb') as f:
    (vuln_peaks,
     vuln_end,
     iso_peaks,
     cum_iso,
     iso_rate_range,
     iso_prob_range) = load(f)

vp_min = 0
vp_max = ceil(vuln_peaks.max())
vptick = vp_min + 0.25 * (vp_max - vp_min) * arange(5)
ve_min = 5 * floor(vuln_end.min() / 5)
ve_max = 5 * ceil(vuln_end.max() / 5)
vetick = ve_min + 0.2 * (ve_max - ve_min) * arange(6)
ip_min = 0
ip_max = 0.1 * ceil(iso_peaks.max() / 0.1)
iptick = ip_min + 0.25 * (ip_max - ip_min) * arange(5)
ci_min = 0
ci_max = 5 * ceil(cum_iso.max() / 5)
citick = ci_min + 0.25 * (ci_max - ci_min) * arange(5)

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2,
                                         2,
                                         sharex=True,
                                         constrained_layout=True)

axim=ax1.imshow(vuln_peaks,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=vp_min,
           vmax=vp_max)
ax1.set_ylabel('Isolation probability')
ax1.text(-0.5, 1.1, 'a)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Peak % prevalence\n in vulnerable population",
                cax=cax,
                ticks=vptick)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

axim=ax2.imshow(vuln_end,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=ve_min,
           vmax=ve_max)
ax2.text(-0.3, 1.1, 'b)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = colorbar(axim,
                label="Cumulative % infected\n in vulnerable population",
                cax=cax,
                ticks=vetick)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

axim=ax3.imshow(iso_peaks,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=ip_max)
ax3.set_xlabel('Detection rate')
ax3.set_ylabel('Isolation probability')
ax3.text(-0.5, 1.1, 'c)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Peak % population\n in isolation",
                cax=cax,
                ticks=iptick)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

axim=ax4.imshow(cum_iso,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=ci_max)
ax4.set_xlabel('Detection rate')
ax4.text(-0.3, 1.1, 'd)',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Cumulative % population\n in isolation",
                cax=cax,
                ticks=citick)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
cbar.outline.set_visible(False)

fig.savefig('plots/oohi/grid_plot.png',
            bbox_inches='tight',
            dpi=300)
close()

fig, ((ax1, ax2)) = subplots(2,1)

axim=ax1.imshow(vuln_end,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=ve_max)
ax1.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Cumulative clinically\n vulnerable % prevalence",
                cax=cax)

axim=ax2.imshow(cum_iso,
           origin='lower',
           extent=(iso_rate_range[0],iso_rate_range[-1],0,1),
           vmin=0,
           vmax=ci_max)
ax2.set_xlabel('Detection rate (1/days)')
ax2.set_ylabel('Isolation probability')

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = colorbar(axim,
                label="Total % of population\n who isolate",
                cax=cax)

fig.savefig('plots/oohi/poster_plot.png',
            bbox_inches='tight',
            dpi=300)
close()
