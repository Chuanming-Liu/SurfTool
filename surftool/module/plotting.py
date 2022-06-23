#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib
import pdb
import copy



def plot_setting(fontsize=15):
    plt.rcParams['font.family']         = "Times New Roman"
    plt.rcParams['font.weight']         = 'bold'
    plt.rcParams['axes.labelsize']      = fontsize
    plt.rcParams['axes.labelweight']    = 'bold'
    plt.rcParams['xtick.labelsize']     = fontsize
    plt.rcParams['ytick.labelsize']     = fontsize
    matplotlib.rcParams.update({'font.size': fontsize,'legend.fontsize': fontsize-4})
    return


# plot data set
class plotdata(object):

    def __init__(self):
        self.data           = data.data1d()
        self.data1          = data.data1d()
        self.data2          = data.data1d()
        self.data3          = data.data1d()
        self.datafit        = False
        return

    def plot_data_fitting(self, outdir='.', fname='sys_data_tti'):
        """plot data fitting
        """
        plot_setting(fontsize=12)
        fig, axs              = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        self.plot_disp(ax1=axs[0], wtype='both')
        self.plot_azimuth(ax1=axs[1], ymin=-30., ymax=210.)
        self.plot_aziamp(ax1=axs[2], ymin=0., ymax=1.5)
        # pdb.set_trace()
        fig.set_tight_layout(True)

        plt.tight_layout()
        plt.savefig(outdir+'/'+fname+'.pdf',dpi='figure',format='pdf')
        plt.close('all')

        return

    def plot_disp(self, alpha=0.5, ax1=None, ymin=3.0, ymax=4.8, \
                  wtype='ray', outdir='./',fname='Fitting_Disp', label=''):
        """
        plot phase/group dispersion curves
        =================================================================================================
        ::: input :::
        title       - title for the figure
        obsdisp     - plot observed disersion curve or not
        mindisp     - plot minimum misfit dispersion curve or not
        avgdisp     - plot the dispersion curve corresponding to the average of accepted models or not
        assemdisp   - plot the dispersion curves corresponding to the assemble of accepted models or not
        =================================================================================================
        """
        if ax1 is None:
            plot_setting(fontsize=14)
            fig                         = plt.figure(figsize=(5,5))
            ax                          = plt.subplot()
        else:
            ax                          = ax1


        if wtype == 'ray':
            ax.errorbar(self.data.dispR.pper, self.data.dispR.pvelo, yerr=self.data.dispR.stdpvelo, fmt='o', color='b', zorder=4, lw=2, ms=1, capsize=1.5, label='observed')
        elif wtype == 'lov':
            ax.errorbar(self.data.dispL.pper, self.data.dispL.pvelo, yerr=self.data.dispL.stdpvelo, fmt='o', color='b', zorder=4, lw=2, ms=1, capsize=1.5, label='observed')
        else:
            ax.errorbar(self.data.dispR.pper, self.data.dispR.pvelo, yerr=self.data.dispR.stdpvelo, fmt='o', color='b', zorder=4, lw=2, ms=1, capsize=1.5, label='observed ray')
            ax.errorbar(self.data.dispL.pper, self.data.dispL.pvelo, yerr=self.data.dispL.stdpvelo, fmt='o', color='k', zorder=4, lw=2, ms=1, capsize=1.5, label='observed lov')
        # ax.errorbar(self.data.dispR.pper, self.data.dispR.pvelo, yerr=self.data.dispR.stdpvelo, zorder=4, fmt='o', color='b', alpha=0.7, lw=1, ms=1, label='obs', ecolor='b', capsize=1.5)
        if self.datafit:
            ax.plot(self.data2.dispL.pper, self.data2.dispL.pvelo, 'g-', lw=2, ms=10, label='pred lov')
            ax.plot(self.data2.dispR.pper, self.data2.dispR.pvelo, 'r-', lw=2, ms=10, label='pred ray')


        period                          = self.data.dispR.pper

        ax.xaxis.set_ticks(np.arange(0, period[-1]+2., step=10.))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_ticks(np.arange(ymin, ymax+0.1, step=0.5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlim([period[0]-1, period[-1]+1])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel('period (s)')
        ax.set_ylabel('phasse velocity (km/s)')
        plt.tight_layout()
        ax.legend(loc=0, fontsize=12)

        if ax1 is None:
            plt.legend(loc=0, fontsize=13)
            plt.tight_layout()
            plt.savefig(outdir+'/'+fname+'.pdf',dpi='figure',format='pdf')
            plt.close('all')
        return

    def plot_azimuth(self, ax1=None, ymin=0.0, ymax=180., outdir='./',fname='Fitting_AziPhi', label=''):
        """
        plot phase/group dispersion curves
        =================================================================================================
        ::: input :::
        title       - title for the figure
        obsdisp     - plot observed disersion curve or not
        mindisp     - plot minimum misfit dispersion curve or not
        avgdisp     - plot the dispersion curve corresponding to the average of accepted models or not
        assemdisp   - plot the dispersion curves corresponding to the assemble of accepted models or not
        =================================================================================================
        """
        if ax1 is None:
            plot_setting(fontsize=14)
            fig                         = plt.figure(figsize=(5,5))
            ax                          = plt.subplot()
        else:
            ax                          = ax1

        ax.errorbar(self.data.dispAz.pper, self.data.dispAz.pphio, yerr=self.data.dispAz.stdpphio, fmt='o', color='b', zorder=4, lw=2, ms=1, capsize=1.5, label='observed')

        if self.datafit:
            ax.plot(self.data2.dispAz.pper, self.data2.dispAz.pphio, 'r-', lw=2, ms=10, label='pred')

        period                          = self.data.dispAz.pper
        ax.xaxis.set_ticks(np.arange(0, period[-1]+2., step=10.))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_ticks(np.arange(ymin, ymax+0.1, step=30))
        ax.yaxis.set_minor_locator(AutoMinorLocator(3))
        ax.set_xlim([period[0]-1, period[-1]+1])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel('period (s)')
        ax.set_ylabel('fast azimuth (deg)')
        plt.tight_layout()
        ax.legend(loc=0, fontsize=13)

        if ax1 is None:
            plt.legend(loc=0, fontsize=13)
            # plt.tight_layout()
            plt.savefig(outdir+'/'+fname+'.pdf',dpi='figure',format='pdf')
            plt.close('all')
        return

    def plot_aziamp(self, ax1=None, ymin=0.0, ymax=2., outdir='./',fname='Fitting_AziAmp', label=''):
        """
        plot phase/group dispersion curves
        =================================================================================================
        ::: input :::
        title       - title for the figure
        =================================================================================================
        """
        if ax1 is None:
            plot_setting(fontsize=14)
            fig                         = plt.figure(figsize=(5,5))
            ax                          = plt.subplot()
        else:
            ax                          = ax1


        temp                            = self.data.dispAz.pampo/self.data.dispR.pvelo *100
        std                             = self.data.dispAz.stdpampo/self.data.dispR.pvelo*100
        ax.errorbar(self.data.dispAz.pper, temp, yerr=std, fmt='o', color='b', zorder=4, lw=2, ms=1, capsize=1.5, label='observed')

        if self.datafit:
            temp                        = self.data2.dispAz.pampo/self.data2.dispR.pvelo *100
            ax.plot(self.data2.dispAz.pper, temp, 'r-', lw=2, ms=10, label='pred')



        # pdb.set_trace()
        while (ymax< np.max(temp)):
            ymax                       += 0.5

        period                          = self.data.dispR.pper
        ax.xaxis.set_ticks(np.arange(0, period[-1]+2., step=10.))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_ticks(np.arange(ymin, ymax+0.1, step=0.5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(3))
        ax.set_xlim([period[0]-1, period[-1]+1])
        ax.set_ylim([ymin-0.05, ymax])
        ax.set_xlabel('period (s)')
        ax.set_ylabel('Azi. Aniso. amp (%)')
        # plt.tight_layout()
        ax.legend(loc=0, fontsize=13)

        if ax1 is None:
            plt.legend(loc=0, fontsize=13)
            # plt.tight_layout()
            plt.savefig(outdir+'/'+fname+'.pdf',dpi='figure',format='pdf')
            plt.close('all')
        return

