
"""
To plot tomography results 
"""
import itertools as it
import os
from os.path import join, exists

import string
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean
import pycpt
import shapefile
import copy

import numpy as np
from astropy import convolution as ap_conv
from scipy.optimize import curve_fit

from .plot_setting import VAR2PARAM
from surftool.plot import vrange
import pickle
from tqdm import tqdm 
import pdb

## 
def gaussian_filter(a, x_stddev, y_stddev=None):
    """
    http://docs.astropy.org/en/stable/convolution/index.html
    """
    kwargs_def = {
        'fill_value': np.nan,
        'preserve_nan': True,
        # 'boundary': 'extend',
        # 'nan_treatment': 'interpolate',
    }
    a2 = ap_conv.convolve(a, kernel=ap_conv.Gaussian2DKernel(x_stddev, y_stddev), **kwargs_def)

    return a2

def T2str(per):
    return f'{int(per):03d}_s'

def T2dep_slab(per):
    """
    """
    slab = {10: 20, 20: 20, 30:40, 40:60, 50:60, 60: 80, 70: 100, 80: 100}

    if per <= 16:
        slab_depth = 10
    elif per > 16 and per <= 23:
        slab_depth = 20
    elif per > 23 and per <= 28:
        slab_depth = 20
    elif per > 28 and per <= 35: 
        slab_depth = 40
    elif per > 35 and per < 40:
        slab_depth = 40
    else:
        try: 
            slab_depth = slab[int(per)]
        except:
            slab_depth = None

    return slab_depth

def func_amp(x, y):
    z = np.zeros(x.shape)
    nlat = x.shape[0]
    nlon = x.shape[1]
    for ilat, ilon in it.product(range(nlat), range(nlon)):
        if np.abs(x[ilat, ilon]*100) > 40 and np.abs(y[ilat, ilon]*100.) > 40.:
            z[ilat, ilon] = 100.
    return z 


def fit_dist(y,  **kwargs):
    """
    Fit histogram to a distribution.
    curve_fit
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    # Compute the histogram of a dataset.
    # bins=bins, weights=weights
    # hist: The values of the histogram
    hist, bin_edges = np.histogram(y, **kwargs)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    p0 = [np.max(hist), np.mean(y), np.std(y)]

    # Use non-linear least squares to fit a function, f, to data.
    # x: bin_centers
    # y: hist
    # p0: Initial guess for the parameters 
    # popt: Optimal values for the parameters 
    # pcov: covariance of popt
    popt, pcov = curve_fit(_normal, bin_centers, hist, p0=p0)
    popt[2] = abs(popt[2])
    hist_fit = _normal(bin_centers, *popt)

    return popt, bin_centers, hist, hist_fit

def _normal(x, *p):
    """
    Definition of Gauss function to fit.
    """
    A, mu, sigma = p
    pdf = A * np.exp(-(x-mu)**2 / (2*sigma**2))

    return pdf

# level_diff 
def diff_mesh_kwargs(var_nm, normal=True, _abs=True, diff_helm=False):

    if var_nm == 'psi_2':
        kwargs = {'ndecimal': 0}
        if normal:
            kwargs.update({
                'label':f'Normalized fast direction difference',
                'unit': ''})
            if _abs:
                kwargs.update({
                'abs': True,
                'cmap': 'hot_r',
                'max_diff': 10, 'min_diff': 0, 'nbins': 10, 'n': 16, 
                'levels': np.linspace(0, 8, 9),
                'ylim': [0, 60], 
                })
            else:
                kwargs.update({
                'abs': False, 
                'cmap': 'BlueWhiteOrangeRed_r',
                'max_diff':8, 'min_diff': -8, 'nbins': 17, 'n': 16, 
                'levels': np.linspace(-8, 8, 9), 
                })
        else: 
            if _abs:
                kwargs.update({
                    'abs': True,
                    'cmap': 'hot_r',                
                    'label':r'Fast direction difference (${}^\circ$)',
                    'unit': r'${}^\circ$', 
                    'max_diff': 90, 'min_diff': 0, 'n': 18, 'nbins': 9, 
                    'levels': np.linspace(0, 90, 10),
                    'ylim': [0, 50], 
                    })
            else:
                kwargs.update({
                'abs': False, 
                'cmap': 'BlueWhiteOrangeRed_r',
                'label':r'Fast direction difference (${}^\circ$)',
                'unit': r'${}^\circ$', 
                'max_diff':90, 'min_diff': -90, 'nbins': 20, 'n': 16, 
                'levels': np.linspace(-90, 90, 10),
                'ylim': [0, 30], 
                })  # for colorbar and hist           
            

        sigma_kwargs={'label': r'Combined uncertainties (${}^\circ$)', 'cmap': 'Reds', 'ndecimal': 0,
                    # 'levels': [0, 3, 6, 12, 18, 30],
                    'levels': np.linspace(0, 24, 5)}


    elif var_nm == 'A_2':
        kwargs = {'cmap': 'BlueWhiteOrangeRed_r', 'ndecimal': 0, 'abs': False}     

        if normal: 
            kwargs.update({
                'label': f'Normalized amplitude',
                'unit': '',
                'max_diff': 8, 'min_diff': -8,  'ndecimal': 0, 'nbins': 17, 'n': 16, 
                'levels': np.linspace(-8, 8, 9)})
        else:
            kwargs.update({
                'label': f'Amplitude difference (%)',
                'unit': '%', 
                'max_diff': 3, 'min_diff': -3, 'ndecimal': 1, 'nbins': 17, 'n': 12, # nbins for hist bin; n for colorbar
                'levels': np.linspace(-3, 3, 7),
                'ylim': [0, 35],
                })

        sigma_kwargs={'label': f'Combined uncertainties (%)', 'cmap': 'Reds',
                    'levels': np.linspace(0, 0.5, 6)}

    elif var_nm in ['vel_iso', 'A_0']:
        kwargs = {
            'cmap': 'BlueWhiteOrangeRed_r', # works for [-8, 8]
            # 'cmap': 'BlWhR',
            'ndecimal': 0, 'abs': False} 

        if normal:
            kwargs.update({
                'label': f'Normalized phase speed difference',
                'unit': '',
                'max_diff': 10, 'min_diff': -10, 'nbins': 20, 'n': 8,
                # 'levels': np.linspace(-8, 8, 9)})   
                'levels': np.linspace(-10, 10, 11)})       
        else:
            if diff_helm:
                kwargs.update({
                    'label': f'Phase speed difference (m/s)',
                    'unit': 'm/s', 
                    # 'max_diff': 200, 'min_diff': -200,'nbins': 40, 'n': 8,
                    # 'levels': np.linspace(-200, 200, 5),
                    'max_diff': 50, 'min_diff': -50,'nbins': 20, 'n': 8,
                    'levels': np.linspace(-50, 50, 5),    
                    'ylim': [0, 30],            
                    })                
            else:
                kwargs.update({
                    'label': f'Phase speed difference (m/s)',
                    'unit': 'm/s', 
                    # 'max_diff': 200, 'min_diff': -200,'nbins': 40, 'n': 8,
                    # 'levels': np.linspace(-200, 200, 5),
                    'max_diff': 100, 'min_diff': -100,'nbins': 20, 'n': 8,
                    'levels': np.linspace(-100, 100, 5),    
                    'ylim': [0, 30],            
                    })


        sigma_kwargs={'label': f'Combined uncertainties (m/s)', 'cmap': 'afmhot_r',
                    'levels': [0, 4, 8, 12, 16, 20]}

    elif var_nm == 'A_1':
        kwargs = {'cmap': 'BlueWhiteOrangeRed_r', 'ndecimal': 0, 'abs': False}     


        kwargs.update({
            'label': fr'$A_1$ difference (%)',
            'unit': '%', 
            'max_diff': 3, 'min_diff': -3, 'ndecimal': 1, 'nbins': 17, 'n': 12, # nbins for hist bin; n for colorbar
            'levels': np.linspace(-3, 3, 7),
            'ylim': [0, 35],
            })

        sigma_kwargs={'label': f'Combined uncertainties (%)', 'cmap': 'Reds',
                    'levels': np.linspace(0, 0.5, 6)}

    else: 
        pdb.set_trace('unknow var_nm')
    return kwargs, sigma_kwargs


# mesh plot of ratios
def ratio_mesh_kwargs(var_nm):
    kwargs = {'ndecimal': 0}
    if var_nm == 'A_2':
        kwargs.update({
        'cmap': 'hot_r',
        'xmax':10, 'xmin': 0, 
        'nbins': 12, # for hist
        'n': 6, # for meshplot
        'levels': [0, 1, 3, 5, 7, 9], 
        })
    return kwargs

def avg_mesh_kwargs(var_nm):
    kwargs = {'ndecimal': 0}
    if var_nm == 'A_1':
        kwargs.update({
            'label': f'Amplitude (%)',
            'unit': '%', 
            'max_v': 6, 'min_v': 0, 'ndecimal': 1, 'nbins': 18, 'n': 12, # nbins for hist bin; n for colorbar
            'levels': np.linspace(0, 5, 5),
            'ylim': [0, 35],
            })
    elif var_nm == 'A_2':
        kwargs.update({
            'label': f'Amplitude (%)',
            'unit': '%', 
            'max_v': 6, 'min_v': 0, 'ndecimal': 1, 'nbins': 17, 'n': 12, # nbins for hist bin; n for colorbar
            'levels': np.linspace(0, 6, 6),
            'ylim': [0, 35],
            })

    elif var_nm == 'misfit':
        kwargs.update({
            'label': r'$\chi_\nu$',
            'max_v': 3, 'min_v': 0, 'ndecimal': 1, 'nbins': 17, 'n': 12,
            'levels': np.arange(0, 3, 0.5),
            'ylim': [0, 35],
            })    
    else: 
        pdb.set_trace('unknow var_nm')
    return kwargs

def upscale_sigma(tomoh5, per, **kwargs):
    """
    sigma :: sigma matrix

    misfit <= 1: up=1
    misfit > 1 : up = misfit
    
    """
    scale_type = kwargs.get('scale_type', 'linear')
    wtype = kwargs.get('wtype', 'Ray')
    # additional upscale for A0
    ups_a0 = kwargs.get('ups_a0', False)
    para_nm = kwargs.get('para_nm', None)
    for_inv = kwargs.get('for_inv', False)

    # chisqr = misfit**2
    # exp_scale = np.exp(chisqr/6)
   
    if wtype == 'Ray':
        stackid = kwargs.get('stackid', 1)
        misfit = tomoh5.get_var(per=per, nms=['misfit'], **{'stackid': stackid})[0]
        misfit = np.array(misfit)
        scale = misfit.copy()
        index = (misfit<1)
        scale[index] = 1.
        # addtional upscaling for A0
        if ups_a0:
            scale *= 1.4
        if for_inv:
            if para_nm == 'sigma_A_0':
                scale *= 3.6
            elif para_nm == 'sigma_psi_2':
                scale *= 1.5
            elif para_nm == 'sigma_A_2':
                scale *= 1.
        # if para_nm == 'sigma_A_0':
        #     pdb.set_trace()
        
    else:
        # for love wave 
        scale = 3.
        if for_inv:
            scale *= 2.
            if per < 28:
                scale *= 1.2
            elif per >= 28 and per <= 50:
                scale *= 1.2
            else:
                scale *= 0.5

    # if scale_type == 'linear': 
    #     scale = lin_scale
    #     # index = (misfit<1.5) & (lin_scale<exp_scale)
    #     # scale[index] = exp_scale[index]
    #     index = (misfit<1)
    #     scale[index] = 1.
    # else:
    #     # exponential scale
    #     scale = exp_scale
    #     pdb.set_trace()

    return scale

 

class plot_basemap(object):
    """
    """
    def interface(self, ax, tomoh5=None, per=None, bmap=None, var_nm='vel_iso',
        perturbation=False, **kwargs):
        """
        parameters
        -----
        ax :: ax of matplotlib
        tomoh5 :: h5 object of tomo
        per :: period (int)
        var_nam :: variable name
        """
        custom_mask = kwargs.get('custom_mask', False)
        fmask = kwargs.get('fmask')
        region = kwargs.get('region', 'AA')
        wtype = kwargs.get('wtype', 'Ray')
        upscale = kwargs.get('upscale', True)
        anios_space = kwargs.get('anios_space', 2.8)
        azi_scale = kwargs.get('azi_scale', 50) # smaller, longer
        maprange = kwargs.get('maprange', 'Ray')
        smooth = kwargs.get('smooth', True)
        mask_fig = kwargs.get('mask_fig', True)
        map_key = kwargs.get('map_key', 'all')
        bar_color = kwargs.get('bar_color', 'red')
        dlon = kwargs.get('dlon', 0.2)
        stackid = kwargs.get('stackid', 0)
        plot_hist = kwargs.get('plot_hist', False)
        
        slab_contour = kwargs.get('slab_contour', False)
        text_basin = kwargs.get('mark_basin', False)
        text_pattern = kwargs.get('text_pattern', False)

        paper = kwargs.get('paper', False)

        z = tomoh5.get_var(per=per, nms=[var_nm], **kwargs)[0]
        ax.set_rasterized(True)

        if mask_fig:
            mask = tomoh5.get_mask(per=per, var=var_nm, **kwargs)
            if custom_mask: 
                with open(fmask, 'rb') as f: 
                    mark_pre = pickle.load(f)
                mask |= mark_pre
        else:
            mask = np.full(z.shape, False)

        # sigma upscale
        if var_nm in ['sigma_psi_2', 'vel_sem', 'sigma_A_0', 'sigma_A_2' ] and upscale:
            if var_nm == 'sigma_A_0':
                kwargs['ups_a0'] = True
            lamb = upscale_sigma(tomoh5, per, **kwargs)
            z *= lamb

        if perturbation:
            ref = np.mean(z[~mask])
            z = (z - ref)/ ref * 100.
        
        if smooth: 
            z = np.ma.array(z, mask=mask, fill_value=np.nan)
            sigma = 0.1
            std = sigma / tomoh5.dlon
            z = gaussian_filter(z, x_stddev=std)

        x, y = tomoh5.lon_grd, tomoh5.lat_grd
        z = np.ma.array(z, mask=mask)


        # plot c-map
        kwargs_mesh = VAR2PARAM[var_nm].kwargs_mesh

        # plot histgrams
        if plot_hist:
            self.hist_plot(ax=ax, para=z, xlabel=f'{var_nm}', **{**kwargs, **kwargs_mesh})
            return 


        kwargs['depth'] = T2dep_slab(per)
        if bmap is None:
            bmap = self.plot_basemap_base(ax, **kwargs)


        if var_nm in ['vel_iso', 'A_0']:
            levels = get_levels(region, wtype, per)
            # kwargs_mesh['cmap'] = 'epsl'  # 'cv'
            kwargs_mesh['levels'] = levels
            if perturbation:
                kwargs_mesh['cmap'] = 'epsl' 
                kwargs_mesh['levels'] = np.linspace(-3, 3., 7) # np.linspace(-4, 4, 5)


        elif var_nm in ['sigma_A_0', 'vel_sem']:
            levels = get_unc_levels(wtype, paper)
            kwargs_mesh['levels'] = levels
            kwargs_mesh['n'] = (len(levels)-1)*2

        elif var_nm in ['sigma_A_2'] and paper:
            levels = [0., 0.1, 0.2, 0.4, 0.5]
            kwargs_mesh['levels'] = levels
            kwargs_mesh['n'] = (len(levels)-1)*2

        if var_nm not in ['psi_2', 'psi_1']:
            #-----------------------------------#
            # plot maps
            kwargs_mesh = {**kwargs, **kwargs_mesh}
            imap = plt_mesh(bmap, x, y, z, **kwargs_mesh)
            if text_basin: 
                mark_basin(bmap=bmap, ax=ax)
        else:
            #-----------------------------------#
            # plot Azi map
            if var_nm == 'psi_2':
                headaxislength = 0
                headlength = 0
                npct = 2
                scale = 38 # 45 for AGU;
                # 'width':0.01 too thick # 0.006 
                quiver_kwargs = {'width':0.007, 'edgecolor': 'black', 'linewidth': 0.5, 'headwidth': 0, 'alpha':1} 

            elif var_nm == 'psi_1':
                z  = (z - 180) % 360
                headaxislength = 2
                headlength = 2
                npct = 3
                scale = 70
                quiver_kwargs = {'width':0.004} # 0.006 

            # anios_space = 1.4  (for AA)
            # dlon = 0.2 # default 0.2
            step = int(anios_space/dlon/2)
            amp = tomoh5.get_var(per=per, nms=f'A_{var_nm[-1]}', **{'stackid': stackid})[0]

            if maprange == 'Ray':
                kwargs_key={'X': .05, 'Y': .4, 'U': npct, 'label': f'{npct}%'}
            else:
                kwargs_key={'X': .9, 'Y': .15, 'U': npct, 'label': f'{npct}%'}


            if map_key == 'offshore':
                scale = 20  # smaller, longer    
                plot_volcano(bmap)

            elif map_key == 'all' and var_nm == 'psi_2':
                plt_AK_fault(bmap)
                # plot_Yakutat(bmap)

            if slab_contour:
                plt_slabcontour(bmap, **kwargs)

            imap = plt_quiver(
            bmap, x, y, amplitude=amp, azimuth=z, ax=ax, step=step,
            headaxislength=headaxislength, headlength=headlength,
            scale=scale, color=bar_color, kwargs_key=kwargs_key, **quiver_kwargs)
            # imap = (im, cm, norm)
   

        return imap, bmap



    def interface_two_ratio(self, ax, tomoh5=None, per=None, bmap=None, var_nms=['A_2', 'sigma_A_2'],
        perturbation=False, **kwargs):
        """
        parameters
        -----
        ax :: ax of matplotlib
        tomoh5 :: h5 object of tomo
        per :: period (int)
        var_nam :: variable name
        """
        custom_mask = kwargs.get('custom_mask', False)
        fmask = kwargs.get('fmask')
        region = kwargs.get('region', 'AA')
        wtype = kwargs.get('wtype', 'Ray')
        upscale = kwargs.get('upscale', True)
        anios_space = kwargs.get('anios_space', 2.8)
        azi_scale = kwargs.get('azi_scale', 50) # smaller, longer
        maprange = kwargs.get('maprange', 'Ray')
        smooth = kwargs.get('smooth', True)
        mask_fig = kwargs.get('mask_fig', True)
        map_key = kwargs.get('map_key', 'all')
        bar_color = kwargs.get('bar_color', 'red')
        dlon = kwargs.get('dlon', 0.2)
        stackid = kwargs.get('stackid', 0)
        plot_hist = kwargs.get('plot_hist', False)
        
        slab_contour = kwargs.get('slab_contour', False)
        text_basin = kwargs.get('mark_basin', False)
        text_pattern = kwargs.get('text_pattern', False)

        z1 = tomoh5.get_var(per=per, nms=[var_nms[0]], **kwargs)[0]
        z2 = tomoh5.get_var(per=per, nms=[var_nms[1]], **kwargs)[0]
        # mask = tomoh5.get_mask(per=per, var=var_nms[0], **kwargs)

        mask = np.full(z1.shape, False)
        if custom_mask: 
            with open(fmask, 'rb') as f: 
                mark_pre = pickle.load(f)
            mask |= mark_pre

        # sigma upscale
        if var_nms[1] in ['sigma_psi_2', 'vel_sem', 'sigma_A_0', 'sigma_A_2' ] and upscale:
            lamb = upscale_sigma(tomoh5, per, **kwargs)
            z2 *= lamb

        z = z1/z2

        if smooth: 
            z = np.ma.array(z, mask=mask, fill_value=np.nan)
            sigma = 0.1
            std = sigma / tomoh5.dlon
            z = gaussian_filter(z, x_stddev=std)

        x, y = tomoh5.lon_grd, tomoh5.lat_grd
        # z = np.ma.array(z, mask=mask)

        # plot c-map
        kwargs_ratio = ratio_mesh_kwargs(var_nm=var_nms[0])

        xlabel = f'{var_nms[0]} /sigma'
        # plot histgrams
        if plot_hist:
            self.hist_plot(ax=ax, para=z, xlabel=xlabel, **{**kwargs, **kwargs_ratio})
            return 

        # plot maps
        kwargs['depth'] = T2dep_slab(per)

        bmap = self.plot_basemap_base(ax, **kwargs)
        kwargs_mesh = {**kwargs, **kwargs_ratio}
        ax.set_rasterized(True)
        imap = plt_mesh(bmap, x, y, z, **kwargs_mesh)

        return imap, bmap


    def interface_AziCov(self, ax, tomoh5=None, per=None, bmap=None, **kwargs):
        """ 
        """

        zhist = tomoh5.get_var(per=per, nms=['histArr'])[0]
        x, y = tomoh5.lon_grd, tomoh5.lat_grd
        nlon, nlat = tomoh5.nlon, tomoh5.nlat

        nlat_grad = 1 
        nlon_grad = 1
        dy = 10
        dx = 10
        
        mask = tomoh5.get_mask(per, var='vel_iso')
        # z = np.ma.array(z, mask=mask)
        
        # plot basemap
        if bmap is None:
            bmap = self.plot_basemap_base(ax, **kwargs)


        # set colormap 
        cmap = copy.copy(mpl.cm.get_cmap('viridis'))
        cmap = discrete_cmap(20,  cmap)
        cmap = cmap.reversed()      

        # norm = mpl.colors.Normalize(vmin=0, vmax=200)  
        levels = [0, 10, 50, 100, 200, 300]
        norm = PiecewiseNorm(levels=levels)

        _kwargs = {'cmap': cmap, 'norm': norm}

        for ilat, ilon in tqdm(list(it.product(range(nlat), range(nlon)))):
            # QC: periphery
            cond = (nlat_grad + dy) <= ilat <= (nlat - 1 - nlat_grad - dy)
            cond &= (nlon_grad + dx) <= ilon <= (nlon - 1 - nlon_grad - dx)
            if not cond:
                continue
            
            # if mask[ilat, ilon]:
            #     continue

            ix = ilat - nlat_grad
            iy = ilon - nlon_grad
            mapx = x[ilat, ilon]
            mapy = y[ilat, ilon]


            if mapx >= 230.:
                continue

            if is_whole(mapx) and is_whole(mapy):
                histArr = zhist[:, ix, iy]
                if histArr.sum() > 50:
                    mapx, mapy = bmap(mapx, mapy)
                    plt_wedge_diagram(ax, mapx, mapy, histArr, **_kwargs)

        # add colorbar 
        ckwargs = {'decimal':0, 'pad': 0.4}
        cb = plt_colorbar(bmap=bmap, cm=cmap, norm=norm, **ckwargs)

        return bmap

    def interface_A2_sin_cos(self, ax, tomoh5=None, per=None, bmap=None, var_nm='sigma_A2_sin',
        perturbation=False, smooth=True, mask_fig=True, **kwargs):
        """
        parameters
        -----
        ax :: ax of matplotlib
        tomoh5 :: h5 object of tomo
        per :: period (int)
        var_nam :: variable name
        """
        custom_mask = kwargs.get('custom_mask', False)
        fmask = kwargs.get('fmask')
        region = kwargs.get('region', 'AA')
        wtype = kwargs.get('wtype', 'Ray')


        if mask_fig:
            mask = tomoh5.get_mask(per, var=var_nm)
        else:
            mask = np.full(z.shape, False)

        A2_sin, A2_cos, sigma_A2_sin, sigma_A2_cos = tomoh5.get_var(per=per, nms=['A2_sin', 'A2_cos', 'sigma_A2_sin', 'sigma_A2_cos'])

        if var_nm == 'A2_sin':
            z = A2_sin*1000.
        elif var_nm == 'A2_cos':
            z = A2_cos*1000. 
        elif var_nm == 'sigma_A2_sin':
            z = np.abs(sigma_A2_sin/A2_sin)*100
        elif var_nm == 'sigma_A2_cos': 
            z = np.abs(sigma_A2_cos/A2_cos)*100
        elif var_nm == 'A2_amp':
            A_0 = tomoh5.get_var(per=per, nms=['A_0'])[0]
            z = np.sqrt(A2_sin**2 + A2_cos**2)/A_0*100.*2
        elif var_nm == 'sigma_A2':
            # z = np.sqrt((sigma_A2_sin/A2_sin)**2 + (sigma_A2_cos/A2_cos)**2)*100
            z = func_amp(sigma_A2_sin/A2_sin, sigma_A2_cos/A2_cos)
        


        if custom_mask: 
            with open(fmask, 'rb') as f: 
                mark_pre = pickle.load(f)
            mask |= mark_pre

        if perturbation:
            ref = np.mean(z[~mask])
            z = (z - ref)/ ref * 100.
        
        if smooth: 
            z = np.ma.array(z, mask=mask, fill_value=np.nan)
            sigma = 0.1
            std = sigma / tomoh5.dlon
            z = gaussian_filter(z, x_stddev=std)

        x, y = tomoh5.lon_grd, tomoh5.lat_grd
        z = np.ma.array(z, mask=mask)

        # plot basemap
        kwargs['depth'] = T2dep_slab(per)
        if bmap is None:
            bmap = self.plot_basemap_base(ax, **kwargs)

        # plot c-map
        kwargs_mesh = VAR2PARAM[var_nm].kwargs_mesh
        ax.set_rasterized(True)
        imap = plt_mesh(bmap, x, y, z, **kwargs_mesh)

        return imap, bmap

    def interface_diff(self, ax=None, tomo1=None, tomo2=None, per=None, var_nm='',
        smooth=True, **kwargs):
        """
        difference of parameters and statistic

        """
        custom_mask = kwargs.get('custom_mask', False)
        fmask = kwargs.get('fmask')
        normal = kwargs.get('normal', False)
        plot_type = kwargs.get('plot_type', None)        
        key_abs = kwargs.get('abs', True)
        upscale = kwargs.get('upscale', False)
        stackid1 = kwargs.get('stackid1', 0)
        stackid2 = kwargs.get('stackid2', 0)   
        diff_helm = kwargs.get('diff_helm', False)

        kwargs_diff, sigma_kwargs = diff_mesh_kwargs(var_nm=var_nm, normal=normal, _abs=key_abs, diff_helm=diff_helm)

        ndecimal = kwargs_diff.get('ndecimal', 0)
        unit = kwargs_diff.get('unit')
        levels = kwargs_diff.get('levels')
        label = kwargs_diff.get('label')
        kwargs_diff['colorbar'] = kwargs.get('colorbar', False)

        var_un = VAR2PARAM[var_nm].un
        z1, sigma1 = tomo1.get_var(per=per, nms=[var_nm, var_un], **{'stackid': stackid1})
        z2, sigma2 = tomo2.get_var(per=per, nms=[var_nm, var_un], **{'stackid': stackid2})
        mask = tomo1.get_mask(per=per, var=var_nm, **kwargs)

        if custom_mask: 
            with open(fmask, 'rb') as f: 
                mark_pre = pickle.load(f)
            mask |= mark_pre

        # angle [0, 180]
        diff = z1 - z2
        if var_nm in ['vel_iso', 'A_0']:
            diff *= 1000

        elif var_nm == 'psi_2':
            """
            psi2 is within [0, 180]
            """
            # M1
            # diff[abs(diff)>90] = 180 -  abs(diff[abs(diff)>90])
            # M2
            diff[abs(diff)>90] = np.sign(diff[abs(diff)>90]) * (180 - abs(diff[abs(diff)>90]))

        if key_abs: 
            diff = np.abs(diff)

        diff_raw = np.ma.array(diff, mask=mask, fill_value=np.nan)


        if upscale:
            kwargs['stackid'] = stackid1
            lambda1 = upscale_sigma(tomo1, per, **kwargs)
            sigma1 *= lambda1
            # for 2
            kwargs['stackid'] = stackid2
            lambda2 = upscale_sigma(tomo2, per, **kwargs)
            sigma2 *= lambda2 

        # normalization by uncertainties
        if (sigma1 is not None) and (sigma2 is not None):
            sigma_map = np.sqrt(sigma1**2 + sigma2**2)
            sigma = np.sqrt(sigma1**2 + sigma2**2)[~mask]
            sigma_mean = np.nanmean(sigma)

        if normal:
            try:
                diff = diff_raw.copy()
                diff[~mask] /= sigma
            except:
                raise ValueError('Uncertainty not given!!')
        else:
            diff = diff_raw 

        mean = np.nanmean(diff)
        std = np.nanstd(diff)

        if smooth:
            plot_sigma = 0.1
            s_std = plot_sigma / tomo1.dlon
            diff = gaussian_filter(diff, x_stddev=s_std)

        #--- plot c-map --#
        if plot_type == 'diff_map':
            x, y = tomo1.lon_grd, tomo1.lat_grd
            z = np.ma.array(diff, mask=mask)

            # if bmap is None:
            bmap = self.plot_basemap_base(ax, **kwargs)
            
            ax.set_rasterized(True)
            
            imap = plt_mesh(bmap, x, y, z, **kwargs_diff)

        if plot_type == 'sigma_map':
            x, y = tomo1.lon_grd, tomo1.lat_grd
            z = np.ma.array(sigma_map, mask=mask)

            # if bmap is None:
            bmap = self.plot_basemap_base(ax, **kwargs)
            
            ax.set_rasterized(True)
            imap = plt_mesh(bmap, x, y, z, **sigma_kwargs)

        #----------------#
        #--- histgram ---#
        min_diff = kwargs_diff.get('min_diff')
        max_diff = kwargs_diff.get('max_diff')
        nbins = kwargs_diff.get('nbins')
        ylim = kwargs_diff.get('ylim', None)
        unit = kwargs_diff.get('unit')

        xlim = [min_diff, max_diff]
        bins = np.linspace(*xlim, nbins+1)
        diff = np.ma.array(diff, mask=mask).compressed()
        diff = diff[~np.isnan(diff)]
        diff = diff[np.abs(diff)<max_diff]
        weights = np.ones(diff.size)/diff.size * 100.
        gauss_fit =True

        # fit the gaussian distribution
        try: 
            popt, bin_centers, hist, hist_fit = fit_dist(diff, bins=bins, weights=weights)
            if gauss_fit and not key_abs:
                mean, std = popt[1:3]
        except RuntimeError:
            gauss_fit = False 
        

        if plot_type == 'hist':
            kwargs_hist = {
                'edgecolor': 'k',
                'alpha': 0.5, 
                'bins': bins,
                'weights': weights 
            }

            ax.hist(diff, **kwargs_hist)
            if gauss_fit and not key_abs: 
                x_interp = np.linspace(-max_diff, max_diff, 10*nbins)
                y_interp = _normal(x_interp, *popt)
                ax.plot(x_interp, y_interp, lw=2, alpha=0.8)
            ax.axvline(0, c='k', lw=0.5, ls='--')
            # title = f'{mean:.{ndecimal+1}f} $\pm$ {std:.{ndecimal+1}f} {unit}'

            ax.set_xticks(levels)
            ax.tick_params(axis='x', length=0)
            # minor_ticks = levels
            # ax.set_xticks(minor_ticks[::2], minor=True)

            if key_abs:  
                xlim[0] = 0
            ax.set_xlim(*xlim)
            
            if ylim is None:
                ylim = ax.get_ylim()

            if ylim[1]>50:
                yticks_major = 10
                yticks_minor = 2
            else:
                yticks_major = 5
                yticks_minor = 1
            ax.set_yticks(np.arange(0, 100, yticks_major))
            ax.set_yticks(np.arange(0, 100, yticks_minor), minor=True)

            ax.set_ylim(*ylim)

            ax.set_xlabel(label)
            if kwargs.get('ylabel', True): 
                ax.set_ylabel('Percentage (%)')
            else:
                ax.axes.yaxis.set_ticklabels([])

            ax.tick_params(which='major', direction='in')
            ax.tick_params(which='minor', direction='in')
            # ax.yaxis.set_label_position('right')
            # ax.yaxis.tick_right()

            _txt = '\n'.join([
                fr'Mean: {mean:.1f} {unit}',
                fr'SD: {std:.1f} {unit}',
                # fr'$\langle \delta c \rangle = {sigma_mean:.0f}$ m/s',
            ])
            ax.text(.6, .85, f'{_txt}', fontsize=8, transform=ax.transAxes)


        if plot_type == 'diff_map': 
            return imap, bmap, mean, std, sigma_mean 
        else: 
            return mean, std, sigma_mean

    def hist_plot(self, ax=None, para=None, **kwargs):

        #--- histgram ---#
        xmin = kwargs.get('xmin')
        xmax = kwargs.get('xmax')
        nbins = kwargs.get('nbins')
        levels = kwargs.get('levels')
        ylim_max = kwargs.get('ymax_hist', None)
        unit = kwargs.get('unit', '')
        xlabel = kwargs.get('label', '')
        key_abs = kwargs.get('abs', '')


        xlim = [xmin, xmax]
        bins = np.linspace(*xlim, nbins+1)
        para = para[~np.isnan(para)]
        para = para[np.abs(para)<xmax]
        weights = np.ones(para.size)/para.size * 100.
        gauss_fit =True

        # fit the gaussian distribution
        try: 
            popt, bin_centers, hist, hist_fit = fit_dist(para, bins=bins, weights=weights)
            if gauss_fit:
                mean, std = popt[1:3]
        except RuntimeError:
            gauss_fit = False 
        

        kwargs_hist = {
            'edgecolor': 'k',
            'alpha': 0.5, 
            'bins': bins,
            'weights': weights 
        }

        ax.hist(para, **kwargs_hist)
        if gauss_fit: 
            x_interp = np.linspace(-xmax, xmax, 10*nbins)
            y_interp = _normal(x_interp, *popt)
            ax.plot(x_interp, y_interp, lw=2, alpha=0.8)
        ax.axvline(0, c='k', lw=0.5, ls='--')
        # title = f'{mean:.{ndecimal+1}f} $\pm$ {std:.{ndecimal+1}f} {unit}'

        ax.set_xticks(levels)
        ax.tick_params(axis='x', length=0)
        # minor_ticks = levels
        # ax.set_xticks(minor_ticks[::2], minor=True)

        ax.set_xlim(*xlim)
        
        if ylim_max is None:
            ylim = ax.get_ylim()
        else:
            ylim = [0, ylim_max]

        if ylim[1]>50:
            yticks_major = 10
            yticks_minor = 2
        else:
            yticks_major = 5
            yticks_minor = 1
        ax.set_yticks(np.arange(0, 100, yticks_major))
        ax.set_yticks(np.arange(0, 100, yticks_minor), minor=True)

        ax.set_ylim(*ylim)

        ax.set_xlabel(xlabel)
        if kwargs.get('ylabel', True): 
            ax.set_ylabel('Percentage (%)')
        else:
            ax.axes.yaxis.set_ticklabels([])

        ax.tick_params(which='major', direction='in')
        ax.tick_params(which='minor', direction='in')
        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()

        _txt = '\n'.join([
            fr'Mean: {mean:.1f} {unit}',
            fr'SD: {std:.1f} {unit}',
            # fr'$\langle \delta c \rangle = {sigma_mean:.0f}$ m/s',
        ])
        ax.text(.6, .85, f'{_txt}', fontsize=8, transform=ax.transAxes)
        return 

    def stat_diff_vs_T(self, axes=None, tomo1=None, tomo2=None, pers=None, **kwargs):
        """
        three rows normalized plot (0: mean_diff; 1: N_std; 2: mean_sigma)
        """
        mean = []
        std = []
        sigma_mean = [] 

        diff_kwargs = {'plot_type': 'None', **kwargs}
        plot_mean = kwargs.get('plot_mean', True)
        plot_std = kwargs.get('plot_std', True)
        plot_sigma = kwargs.get('plot_sigma', True)
        ups_a0 = kwargs.get('ups_a0', False)
        upscale = kwargs.get('upscale', True)
        var_nm = kwargs.get('var_nm')
        wtype = kwargs.get('wtype', 'Ray')

        for per in pers:
            _mean, _std, _sigma_mean = self.interface_diff(tomo1=tomo1, tomo2=tomo2, per=per, **diff_kwargs)
            mean.append(_mean)
            std.append(_std)
            sigma_mean.append(_sigma_mean)

        if kwargs.get('stop', False):
            pdb.set_trace()

        # Plot 
        kwargs_plt = {'marker': 'o',  'ms': 3, 'ls': '--', 'lw': .5} 
        if axes is not None:
            ax_id = 0
            if len(axes) > 1:
                ax = axes[ax_id]
            else:
                ax = axes[0]
            if plot_mean:
                if wtype == 'Ray':
                    ax.plot(pers, mean, c='tab:orange', **kwargs_plt)
                else:
                    ax.plot(pers, mean, c='tab:blue', **kwargs_plt)

                ax.axhline(0, c='k', lw=0.5, alpha=.5, ls=':')
                ax_id += 1 

            if plot_std:
                if var_nm in ['A_0', 'vel_iso']:
                    color = 'green'
                    if (wtype=='Ray' and ups_a0) or (wtype=='Lov' and upscale):    
                        color = 'tab:orange'
                else:
                    color = 'tab:orange'
                # if wtype == 'Lov':
                #     pdb.set_trace()

                axes[ax_id].plot(pers, std, c=color, **kwargs_plt)
                axes[ax_id].axhline(1, c='k', lw=.6, alpha=.75, ls=':')
                ax_id += 1
                
            if plot_sigma:
                axes[ax_id].plot(pers, sigma_mean, c='tab:red', **kwargs_plt)
        # ax.set_xlabel('Period (s)')
        # ax.set_ylabel(f'Difference ({unit})')
        # axes[0].set_ylim(-max_diff/2, max_diff/2)
        # ax.tick_params(which='both', right=True, labelright=True)
        # ax.grid(axis='y', ls='--')
        # if fig is not None: fig.savefig(f'{fout}.pdf')
        return 

    def stat_avg(self, tomo=None, per=None, var_nm='', ax=None,  smooth=True, **kwargs):
        """
        difference of parameters and statistic

        """
        custom_mask = kwargs.get('custom_mask', False)
        fmask = kwargs.get('fmask')
        plot_type = kwargs.get('plot_type', None)        

        stackid = kwargs.get('stackid', 0)
        kwargs_avg = avg_mesh_kwargs(var_nm=var_nm)

        ndecimal = kwargs_avg.get('ndecimal', 0)
        unit = kwargs_avg.get('unit')
        levels = kwargs_avg.get('levels')
        label = kwargs_avg.get('label')


        var_un = VAR2PARAM[var_nm].un
        try:
            z1, sigma1 = tomo.get_var(per=per, nms=[var_nm, var_un], **{'stackid': stackid})
        except:
            return 0, 0
            
        
        mask = tomo.get_mask(per=per, var=var_nm, **kwargs)
        if custom_mask: 
            with open(fmask, 'rb') as f: 
                mark_pre = pickle.load(f)
            mask |= mark_pre

        z1 = np.ma.array(z1, mask=mask, fill_value=np.nan)
        mean = np.nanmean(z1)
        std = np.nanstd(z1)

        if smooth:
            plot_sigma = 0.1
            s_std = plot_sigma / tomo.dlon
            z1 = gaussian_filter(z1, x_stddev=s_std)

        #----------------#
        #--- histgram ---#
        min_v = kwargs_avg.get('min_v')
        max_v = kwargs_avg.get('max_v')
        nbins = kwargs_avg.get('nbins')
        ylim = kwargs_avg.get('ylim', None)
        unit = kwargs_avg.get('unit')

        xlim = [min_v, max_v]
        bins = np.linspace(*xlim, nbins+1)
        z1 = np.ma.array(z1, mask=mask).compressed()
        z1 = z1[~np.isnan(z1)]
        z1 = z1[np.abs(z1)<max_v]
        weights = np.ones(z1.size)/z1.size * 100.
        gauss_fit =True

        # fit the gaussian distribution
        try: 
            popt, bin_centers, hist, hist_fit = fit_dist(z1, bins=bins, weights=weights)
            if gauss_fit:
                mean, std = popt[1:3]
        except RuntimeError:
            gauss_fit = False 
        

        if ax is not None:
            kwargs_hist = {
                'edgecolor': 'k',
                'alpha': 0.5, 
                'bins': bins,
                'weights': weights 
            }

            ax.hist(diff, **kwargs_hist)
            if gauss_fit and not key_abs: 
                x_interp = np.linspace(-max_v1, max_v1, 10*nbins)
                y_interp = _normal(x_interp, *popt)
                ax.plot(x_interp, y_interp, lw=2, alpha=0.8)
            ax.axvline(0, c='k', lw=0.5, ls='--')
            # title = f'{mean:.{ndecimal+1}f} $\pm$ {std:.{ndecimal+1}f} {unit}'


            ax.set_xticks(levels)
            ax.tick_params(axis='x', length=0)
            # minor_ticks = levels
            # ax.set_xticks(minor_ticks[::2], minor=True)

            if key_abs:  
                xlim[0] = 0
            ax.set_xlim(*xlim)
            
            if ylim is None:
                ylim = ax.get_ylim()

            if ylim[1]>50:
                yticks_major = 10
                yticks_minor = 2
            else:
                yticks_major = 5
                yticks_minor = 1
            ax.set_yticks(np.arange(0, 100, yticks_major))
            ax.set_yticks(np.arange(0, 100, yticks_minor), minor=True)

            ax.set_ylim(*ylim)

            ax.set_xlabel(label)
            if kwargs.get('ylabel', True): 
                ax.set_ylabel('Percentage (%)')
            else:
                ax.axes.yaxis.set_ticklabels([])

            ax.tick_params(which='major', direction='in')
            ax.tick_params(which='minor', direction='in')
            # ax.yaxis.set_label_position('right')
            # ax.yaxis.tick_right()

            _txt = '\n'.join([
                fr'Mean: {mean:.1f} {unit}',
                fr'SD: {std:.1f} {unit}',
            ])
            ax.text(.6, .85, f'{_txt}', fontsize=8, transform=ax.transAxes)


        return mean, std

    def stat_vs_T(self, ax=None, tomo=None, var_nm=None, pers=None, **kwargs):
        """
        statistic avg vs T
        """
        mean = []
        std = []

        plot_mean = kwargs.get('plot_mean', True)
        plot_unc = kwargs.get('plot_unc', True)
        wtype = kwargs.get('wtype', 'Ray')
        color = kwargs.get('color', 'tab:orange')
        label = kwargs.get('label', 'eik')


        for per in pers:
            _mean, _std = self.stat_avg(tomo=tomo, var_nm=var_nm, per=per, **kwargs)
            mean.append(_mean)
            std.append(_std)

        # Plot 
        kwargs_plt = {'marker': 'o',  'ms': 3, 'ls': '--', 'lw': .5} 
        if ax is not None:
            ax.plot(pers, mean, c=color, label=label, **kwargs_plt)
            ax.axhline(0, c='k', lw=0.5, alpha=.5, ls=':')

        # ax.set_xlabel('Period (s)')
        # ax.set_ylabel(f'Difference ({unit})')
        # axes[0].set_ylim(-max_diff/2, max_diff/2)
        # ax.tick_params(which='both', right=True, labelright=True)
        # ax.grid(axis='y', ls='--')
        # if fig is not None: fig.savefig(f'{fout}.pdf')
        return mean

    def interface4para(self, ax, tomoh5=None,  var_nm='lambda', per=None, bmap=None, **kwargs):
        """
        parameters
        -----
        ax :: ax of matplotlib
        tomoh5 :: h5 object of tomo
        per :: period (int)
        var_nam :: variable name
        """
        custom_mask = kwargs.get('custom_mask', False)
        fmask = kwargs.get('fmask')
        smooth = kwargs.get('smooth', True)
        mask_fig = kwargs.get('mask_fig', True)
        stackid = kwargs.get('stackid', 0)


        z = tomoh5.get_var(per=per, nms=[var_nm], **kwargs)[0]
        # mask = tomoh5.get_mask(per=per, var='psi_2')
        if mask_fig:
            mask = tomoh5.get_mask(per=per, var=var_nm, **kwargs)
        else:
            mask = np.full(z.shape, False)

        #
        if var_nm == 'lambda':
            lambda1 = upscale_sigma(tomoh5, per, **kwargs)


        if custom_mask: 
            with open(fmask, 'rb') as f: 
                mark_pre = pickle.load(f)
            mask |= mark_pre

        if smooth: 
            z = np.ma.array(z, mask=mask, fill_value=np.nan)
            sigma = 0.1
            std = sigma / tomoh5.dlon
            z = gaussian_filter(z, x_stddev=std)

        x, y = tomoh5.lon_grd, tomoh5.lat_grd
        z = np.ma.array(z, mask=mask)
        # mask
        z_val = z[~np.isnan(z)]
        mean = np.mean(z_val)
        ax.text(0.5, 0.95, f'Mean: {mean:.1f}', fontsize=10, transform=ax.transAxes)

        # plot basemap
        kwargs['depth'] = T2dep_slab(per)
        if bmap is None:
            bmap = self.plot_basemap_base(ax, **kwargs)

        # plot c-map
        kwargs_mesh = VAR2PARAM[var_nm].kwargs_mesh

        ax.set_rasterized(True)
        imap = plt_mesh(bmap, x, y, z, **kwargs_mesh)

        return 

    def _plt_diff(self, z1, z2, ax1, ax2, kwargs_hist={}, **kwargs):
        """
        Plot difference map and histograms.

        :param relative: If relative to uncertainty
        """
        var         = kwargs.get('var', 'vel_iso')
        plot        = kwargs.get('plot', True)
        x           = kwargs.get('x', self.lon_grd)
        y           = kwargs.get('y', self.lat_grd)
        mask1       = kwargs.get('mask1', False)
        mask2       = kwargs.get('mask2', False)
        relative    = kwargs.get('relative', False)
        
        sigma1      = kwargs.get('sigma1', None)
        sigma2      = kwargs.get('sigma2', None)
        unit        = kwargs.get('unit', VAR2PARAM[var].diff['unit'])
        max_diff    = kwargs.get('max_diff', VAR2PARAM[var].diff['max_diff'])
        min_diff    = kwargs.get('min_diff', VAR2PARAM[var].diff['min_diff'])
        gauss_fit   = kwargs.get('gauss_fit', True)
        ndecimal    = kwargs.get('ndecimal', VAR2PARAM[var].diff['ndecimal'])
        nbins       = kwargs.get('nbins', VAR2PARAM[var].diff['nbins'])
        abs_        = kwargs.get('abs', VAR2PARAM[var].diff.get('abs'))

        diff        = z1 - z2
        if abs_:
            diff    = np.abs(diff)
        if var in ['vel_iso', 'A_0']:
            diff   *= 1000

        mask        = mask1 | mask2 #| (np.abs(diff) > max_diff)

        # by Liu mask outside
        if PARAM['mask']['val']:
            with open(PARAM['mask']['fmeta'], 'rb') as f: 
                mark_pre = pickle.load(f)
            mask   |= mark_pre

        diff_raw    = np.ma.array(diff, mask=mask, fill_value=np.nan)
        # print(np.nanmean(sigma1), np.nanmean(sigma2))

        sigma_mean  = 0
        if (sigma1 is not None) and (sigma2 is not None):
            sigma   = np.sqrt(sigma1**2 + sigma2**2)[~mask]
            sigma_mean = np.nanmean(sigma)

        if relative:
            try:
                diff = diff_raw.copy()
                diff[~mask] /= sigma
            except:
                raise ValueError('Uncertainty not given for normalized difference')
            label   = 'Normalized difference'
        else:
            diff    = diff_raw
            label   = f'Difference ({unit})'

        mean        = np.nanmean(diff)
        std         = np.nanstd(diff)

        if kwargs.get('smooth', False):
            diff    = self._gaussian_filter(diff)

        # Difference map
        if plot:
            
            m       = self._map(ax1, **kwargs)
            kwargs_def = {
                'levels': [min_diff, min_diff/2, 0, max_diff/2, max_diff],
                'ndecimal': ndecimal,
                'label': label,
                'cmap': 'BlueWhiteOrangeRed_r',
                # 'cmap': 'cv',
            }
            kwargs  = {**kwargs_def, **VAR2PARAM[var].diff, **kwargs}
            diff    = np.ma.array(diff, mask=mask)

            # 
            
            im, cmap, norm = my.maplt.plt_mesh(m, x, y, diff, **kwargs)
            # if kwargs.get('mask_ocean', True):
            #     m.drawlsmask(ocean_color='aqua', lakes=False, alpha=.5)

        levels      = kwargs.get('levels')

        # --- Difference histogram --- #
        # max_diff = 10
        # nbins = 21
        xlim        = [min_diff, max_diff]
        bins        = np.linspace(*xlim, nbins+1)
        diff        = np.ma.array(diff, mask=(mask1 | mask2)).compressed()
        diff        = diff[~np.isnan(diff)]
        diff        = diff[np.abs(diff) < max_diff]
        weights     = np.ones(diff.size) / diff.size * 100


        try:
            popt, bin_centers, hist, hist_fit = my.signal.fit_dist(diff, bins=bins, weights=weights)
            if gauss_fit and not abs_:
                mean, std   = popt[1:3]
        except RuntimeError:
            gauss_fit       = False

        if plot:
            kwargs_def      = {
                'edgecolor': 'k',
                'alpha': .5, 'bins': bins, 'weights': weights}
            kwargs_hist     = {**kwargs_def, **kwargs_hist}
            ax2.hist(diff, **kwargs_hist)
            if gauss_fit and not abs_:
                x_interp    = np.linspace(-max_diff, max_diff, 10*nbins)
                y_interp    = my.signal._normal(x_interp, *popt)
                ax2.plot(x_interp, y_interp, lw=2, alpha=.8)
            ax2.axvline(0, c='k', lw=.5)
            title           = fr'{mean:.{ndecimal+1}f} $\pm$ {std:.{ndecimal+1}f} {unit}'
            # if sigma_mean != 0:
            #     title += r' ($\bar{\sigma} = $' + f'{sigma_mean:.0f} m/s)'
            if abs_:
                title       = f'SD = {std:.{ndecimal+1}f} {unit}'
            # ax2.set_title(title, pad=3)
            if abs_:
                xlim[0]     = 0

            
            ax2.set_xlim(*xlim)
            # ax2.set_xticks(levels[::])
            # minor_ticks = np.linspace(levels.min(), levels.max(), 2*levels.size-1)
            minor_ticks     = levels


            ax2.set_xticks(minor_ticks[::2], minor=True)
            _ylim           = ax2.get_ylim()
            ax2.set_yticks(np.arange(0, 100, 5))
            ax2.set_ylim(_ylim)
            ax2.set_xlabel(label)
            # ax2.set_ylabel('Percentage (%)')
            # ax2.yaxis.set_label_position('right')
            # ax2.yaxis.tick_right()

            # _txt = '\n'.join([
            #     fr'Mean: ${mean:.0f}$ m/s',
            #     fr'SD: ${std:.0f}$ m/s',
            #     # fr'$\langle \delta c \rangle = {sigma_mean:.0f}$ m/s',
            # ])
            # ax2.text(.6, .85, f'{_txt}', fontsize=8, transform=ax2.transAxes)

        if plot:
            return im, cmap, norm, mean, std, sigma_mean
        else:
            return mean, std, sigma_mean

    def plot_basemap_base(self, ax, **kwargs):

        maprange = kwargs.get('maprange', 'Ray')
        switch = {'Ray': map_Ray_setting, 
            'Ray_mask': map_Ray_mask_setting,
            'sta': map_sta_setting,
            'AA': map_AA_setting,
            'slab': map_slab_setting,
            'onshore': map_onshore_setting,
            'Lov': map_Lov_setting,
         }
        param = switch[maprange](**kwargs)

        bmap = plt_map(ax, **{**param, **kwargs})
        return bmap


def is_whole(x):
    return x %1 == 0

def map_Ray_setting(**kwargs):
    """
    https://matplotlib.org/basemap/users/cyl.html
    """
    proj = kwargs.get('proj', 'lcc')
    # proj = kwargs.get('proj', 'cea')


    param = {
        # 'minlo': -166+360,  # 
        # 'maxlo':  -115.5+360., #

        'minlo': -163.5+360., 
        'maxlo': -120.+360.,
        'minla': 52,
        'maxla': 70, 

        # projection
        'projection': proj,

        # for Lambert pro
        # 'lon_0': -149,
        'lon_0': -144.5,
        'lat_0': 61,
        'lat_1': 52,
        'lat_2': 70,

        # for ploting parallels and meridians
        'bxa': 10,
        'bya': 5,

        # for AK.
        # positive extend; negative shrink
        'bxl': -3,
        'bxr': 4,
        'byu': -2., # up
        'byl': 1., 

        'resolution': 'l',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.7, #0.8 linewidth
        'lw_country': 0.2,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', False), 

        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param

def map_Ray_mask_setting(**kwargs):
    """
    https://matplotlib.org/basemap/users/cyl.html
    Failed to find a propriate projection; such as gmt Jx4c/8c
    """
    # proj = kwargs.get('proj', 'lcc')
    # proj = kwargs.get('proj', 'mill')



    param = {
        'minlo': -165+360,  # 
        'maxlo': -125+360, # -115.5+360.
        'minla': 50,
        'maxla': 71, 

        # projection
        'projection': proj,

        # for ploting parallels and meridians
        'bxa': 10,
        'bya': 5,

        # for AK.
        # positive extend; negative shrink
        'bxl': -3,
        'bxr': 4,
        'byu': -2., # up
        'byl': 1., 

        'resolution': 'l',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.3, #0.8 linewidth
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', False), 

        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param

def map_onshore_setting(**kwargs):

    param = {
        'minlo': 195,  # -166+360
        'maxlo': 235,  # -125+360.
        'minla': 55,
        'maxla': 70, 

        # projection
        'projection': 'lcc',

        # for Lambert pro
        'lon_0': -149,
        'lat_0': 55,
        'lat_1': 52,
        'lat_2': 72,

        # for ploting parallels and meridians
        'bxa': 10,
        'bya': 5,

        # for AK.
        # positive extend; negative shrink
        'bxl': -3,
        'bxr': 4,
        'byu': -2., # up
        'byl': 1., 

        'resolution': 'l',  # c, l, i, h, f, for boundary
        'lw_coastline': 1, #0.8 linewidth
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', True), 

        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param

def map_Lov_setting(**kwargs):

    param = {
        'minlo': 195,  # -166+360
        'maxlo': 240,  # -120+360.
        'minla': 55,
        'maxla': 70, 

        # projection
        'projection': 'lcc',

        # for Lambert pro
        'lon_0': -149,
        'lat_0': 55,
        'lat_1': 52,
        'lat_2': 72,

        # for ploting parallels and meridians
        'bxa': 10,
        'bya': 5,

        # for AK.
        # positive extend; negative shrink
        # 'bxl': -3,
        # 'bxr': 4,
        # 'byu': -2., # up
        # 'byl': 1., 

        'resolution': 'l',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.3, #0.8 linewidth
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', False), 

        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param

# larmbert projection
def map_AA_lcc_setting(**kwargs):
    param = {
        'minlo': 196., #-165.+360.
        'maxlo': 213., #-147+360.
        'minla': 52.,
        'maxla': 60., 

        # projection
        'projection': 'lcc',

        # for Lambert pro
        'lon_0': -156.5, # central point
        'lat_0': 57, # central point
        'lat_1': 53,
        'lat_2': 61,

        # for ploting parallels and meridians
        'bxa': 5,
        'bya': 2.,

        # for AK.
        # positive extend; negative shrink
        'bxl': 0,
        'bxr': 0,   # east
        'byu': 0., # north
        'byl': 0,  # south

        'resolution': 'l',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.8,
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', False), 
        'plot_seg': kwargs.get('plot_seg', True),


        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param

# Mercator projection
def map_AA_setting(**kwargs):


    param = {

        'minlo': 196., #-165.+360.
        'maxlo': 212., #-147+360.
        'minla': 52.,
        'maxla': 62., 

        # projection
        'projection': 'merc',


        # for ploting parallels and meridians
        'bxa': 5,
        'bya': 2.,


        'resolution': 'i',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.8,
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', True), 
        'plot_seg': kwargs.get('plot_seg', False),
        'plot_volc': True,


        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param


# Mercator projection
def map_slab_setting(**kwargs):

    param = {
        'minlo': -165.+360.,
        'maxlo': -142+360.,
        'minla': 53.2,
        'maxla': 66, 

        # projection
        'projection': 'merc',

        # for ploting parallels and meridians
        'bxa': 5,
        'bya': 2.,

        'resolution': 'i',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.8,
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', True), 
        'plot_seg': kwargs.get('plot_seg', False),
        'plot_volc': True,


        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param


def map_sta_setting(**kwargs):

    param = {
        'minlo': 188,  # -172
        'maxlo': 238., # -122+360.
        'minla': 51.,
        'maxla': 72., 

        # projection
        'projection': 'lcc', #lcc

        # for Lambert pro
        'lon_0': -150, # central point
        'lat_0': 60, # central point
        'lat_1': 50,
        'lat_2': 70,

        # for ploting parallels and meridians
        'bxa': 10,
        'bya': 5,

        # for AK.
        # positive extend; negative shrink
        'bxl': 3,
        'bxr': 10,   # east
        'byu': -3., # north
        'byl': 5.,  # south

        'resolution': 'l',  # c, l, i, h, f, for boundary
        'lw_coastline': 0.8,
        'lw_country': -0.1,
        'lw_state': -1,  
        'lw_gc': .1,

        'plot_trench': True,
        'plot_ak': True,

        'drawlsmask': True, 
        'land_color': 'white', 
        'ocean_color': 'gray',
        'mask_ocean': False, 
        'lakes': False, 
        'mask_fig': True,

        'plot_ak': kwargs.get('plot_ak', True), 
        'plot_trench': kwargs.get('plot_trench', True), 
        'plot_slab': kwargs.get('plot_slab', False), 

        'ctr': False,
        'c_mask': [1, 1, 1, 0],

    }      

    return param

# --------------#
def get_levels(region, wtype, per):
    if wtype=='Ray':
        # if region=='AK':
        #     levels  = vrange.LEVELS_C_AK_Ray.get(int(per))
        # elif region=='AA':
        levels  = vrange.LEVELS_C_AA_Ray.get(int(per))
    else:
        levels  = vrange.LEVELS_C_AK_Lov.get(int(per))
    return levels

def get_unc_levels(wtype, paper=False):
    if wtype == 'Ray':
        levels = [0, 5, 10, 15, 20, 25]
    else: 
        levels = [0, 10, 20, 30, 40]

    if paper: 
        levels = [0, 5, 10, 15, 20]
    return levels
        
# --------------#    
def plt_map(ax, **kwargs):
    """
    Return a Basemap
    https://matplotlib.org/basemap/users/cyl.html

    """
    minla = kwargs.get('minla')
    maxla = kwargs.get('maxla')
    minlo = kwargs.get('minlo')
    maxlo = kwargs.get('maxlo')

    proj = kwargs.get('projection', 'merc')
    res = kwargs.get('resolution', 'l')
    res = 'i' # The options are c (crude, the default), l (low), i (intermediate), h (high), f (full) or None.
    bx = kwargs.get('bx', 1)
    bxl = kwargs.get('bxl', bx)
    bxr = kwargs.get('bxr', bx)
    by = kwargs.get('by', bx/2)
    bxa = kwargs.get('bxa', 1)
    # by Liu for AK
    byu = kwargs.get('byu', bx/2)
    byl = kwargs.get('byl', bx/2)

    meridians = kwargs.get('meridians', np.arange(-180, 180, bxa).astype('int'))
    bya = kwargs.get('bya', bxa)
    parallels = kwargs.get('parallels', np.arange(-90, 90, bya).astype('int'))
    lw_gc = kwargs.get('lw_gc', .1)
    # lw_coastline = kwargs.get('lw_coastline', .5)
    # lw_country = kwargs.get('lw_country', .5)
    lw_state = kwargs.get('lw_state', -0.2)
    fillcontinents = kwargs.get('fillcontinents', False)

    drawlsmask = kwargs.get('drawlsmask', True)
    land_color = kwargs.get('land_color', 'gray')
    ocean_color = kwargs.get('ocean_color', 'white')
    lakes = kwargs.get('lakes', False)
    label_parallel = kwargs.get('label_parallel', [1, 0, 0, 0])  # WENS
    label_meridian = kwargs.get('label_meridian', [0, 0, 0, 1]) #  [left, right, top, bottom]
    alpha = kwargs.get('alpha', .6)

    plot_ak = kwargs.get('plot_ak', True)
    plot_trench = kwargs.get('plot_trench', True)
    plot_slab = kwargs.get('plot_slab', False)
    plot_seg = kwargs.get('plot_seg', False)
    plot_volc = kwargs.get('plot_volc', False)

    fontsize = kwargs.get('fontsize', 8)

    # important setting
    lw_coastline = 0.6
    lw_country = 0.2
    alpha = 0.4


    kwargs['ax'] = ax

    # basemap
    if proj in ['lcc', 'omerc']:
        lon_0 = kwargs.get('lon_0')
        lat_0 = kwargs.get('lat_0')
        lat_1 = kwargs.get('lat_1')
        lat_2 = kwargs.get('lat_2')
        width = kwargs.get('width', 12_000_000)
        height = kwargs.get('height', 9_000_000)
        bmap = Basemap(
            lon_0=lon_0,
            lat_0=lat_0,
            lat_1=lat_1,
            lat_2=lat_2,
            # width=width,
            # height=height,
            # By. Liu. For AK
            llcrnrlon=minlo,
            urcrnrlon=maxlo,
            llcrnrlat=minla,
            urcrnrlat=maxla,
            projection=proj,
            resolution=res,
            rsphere=(6378137.00,6356752.3142),
            ax=ax,
        )  

    elif proj in ['cea', 'merc', 'mill', 'gall', 'cyl']:

        bmap = Basemap(
            llcrnrlon=minlo,
            urcrnrlon=maxlo,
            llcrnrlat=minla,
            urcrnrlat=maxla,
            projection=proj,
            resolution=res,
            ax=ax,
        )         
    else:
        raise ValueError('Wrong projection!')
    
    if lw_gc > 0:
        text_kwargs = {'fontsize': fontsize, 'zorder': -1} #'family':'sans-serif'
        d = bmap.drawparallels(parallels, labels=label_parallel, linewidth=lw_gc, **text_kwargs)
        for v in d.values():
            v[0][0].set_alpha(0.2)
        text_kwargs = {'fontsize': fontsize}
        d = bmap.drawmeridians(meridians, labels=label_meridian, linewidth=lw_gc, **text_kwargs)
        for v in d.values():
            v[0][0].set_alpha(0.2)
        # add line meridian
        # text_inv_kwargs = {'fontsize': 1,  'family':'sans-serif'}
        # bmap.drawmeridians(np.arange(-180, 180, 5).astype('int'), labels=[0, 0, 0, 0], linewidth=lw_gc, **text_inv_kwargs)

    if lw_coastline > 0.:
        coasts = bmap.drawcoastlines(zorder=1, color='0.9', linewidth=0.0001) #0.0001
        _plt_coast(bmap, coasts, lw_coastline)
    
    if lw_country > 0.:
        bmap.drawcountries(linewidth=lw_country)
    
    if drawlsmask:
        # bmap.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=lakes, alpha=alpha)
        # bmap.drawlsmask(land_color='whitesmoke', ocean_color='darkgray', lakes=lakes, alpha=alpha)
        bmap.drawlsmask(land_color='darkgray', ocean_color='lightgray', lakes=lakes, alpha=alpha)

    if plot_ak:
        plt_AK_fault(bmap)
    
    
    if plot_slab:
        plt_slab(bmap, **kwargs)

    
    if plot_trench:
        plot_trench_boundary(bmap)
    
    # if plot_volc:
    #     plot_volcano(bmap)

    # earthquake
    if plot_seg:
        plt_segment(bmap)

    return bmap

# --------------#
# plot geological features
# --------------#
def _plt_coast(m, coasts, lw_coastline=1.):
    """
    replot coast line without rivers
    https://stackoverflow.com/questions/14280312/world-map-without-rivers-with-matplotlib-basemap?noredirect=1&lq=1
    """
    coasts_paths  = coasts.get_paths()

    # In order to see which paths you want to retain or discard you'll need to plot them one
    # at a time noting those that you want etc.
    # poly_stop  = 10
    poly_stop  = 12
    for ipoly in range(len(coasts_paths)):
        if ipoly > poly_stop: break
        r = coasts_paths[ipoly]
        # Convert into lon/lat vertices
        polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in r.iter_segments(simplify=False)]
        px = [polygon_vertices[i][0] for i in range(len(polygon_vertices))]
        py = [polygon_vertices[i][1] for i in range(len(polygon_vertices))]
        m.plot(px, py, 'k-', linewidth=lw_coastline, alpha=0.5)  
    return 

def plt_AK_fault(m, **kwargs):
    """
    Plot tectonic features of AK
    """
    kwargs.pop('val', None)

    # For Fig.11 thicker faults
    kwargs_def = {'c': 'snow', 'lw': 1.1, 'alpha': 0.7, 'zorder': 1, 'path_effects':[pe.Stroke(linewidth=1.8, foreground='black', alpha=0.6),  pe.Normal()]}  

    # kwargs_def = {'c': 'black', 'lw': 0.8, 'alpha': 0.6, 'zorder': 1}

    datadir  = kwargs.get('datadir', '/home/liu/map_data/AK')
    for fnm in ['AK_Faults.txt']:
        fin     = join(datadir, fnm)
        x, y    = np.loadtxt(fin, unpack=True, dtype='str')
        id      = np.where(x == '>')[0]

        for _x, _y in zip(np.split(x, id), np.split(y, id)):
            m.plot(_x[1:-1].astype(np.float), _y[1:-1].astype(np.float), latlon=True,  **{**kwargs_def, **kwargs})
    return 

def plt_slab(bmap, **kwargs):
    """
    slab 2.0
    https://www.sciencebase.gov/catalog/item/5aa2c535e4b0b1c392ea3ca2
    line:
    https://stackoverflow.com/questions/12729529/can-i-give-a-border-outline-to-a-line-in-matplotlib-plot-function/26955697
    """
    
    fshp = kwargs.get('fshp', '/home/liu/map_data/slab2_alu/alu_depth.shp')
    # alpha = kwargs.get('alpha', 0.1)
    depth = kwargs.get('depth')
    maprange = kwargs.get('maprange', 'Ray')

    ax = kwargs.get('ax')
    # 
    # lkwargs = {'lw': 1.2, 'color': 'white', 'path_effects':[pe.Stroke(linewidth=1.8, foreground='gray'), pe.Normal()] }
    lkwargs = {'lw': 2.5, 'color': 'gray', 'alpha': 0.6 }
    fkwargs = {'fontsize': 6, 'ha': 'center', 'va': 'center', 'bbox': dict(facecolor='white', alpha=0.5)}

    
    sf = shapefile.Reader(fshp)

    for sr in sf.shapeRecords():

        if (sr.record.DEPTH == depth):
            
            bmap.plot(*zip(*sr.shape.points), latlon=True, **lkwargs)

            if maprange == 'Ray':
                lon, lat = (-165, 55.5)
            elif maprange == 'AA':
                lon, lat = (196, 55.5)
            else:
                lon, lat = (-163.1, 54.2)
            x, y = bmap(lon, lat)
            ax.text(x, y, f'{depth} km', **fkwargs)
    return 

def plt_segment(bmap, **kwargs):
    fold = kwargs.get('ftxt', '/home/liu/map_data/AACSE')
    segment = ['semidi']

    alpha = kwargs.get('alpha', 0.5)

    ax = kwargs.get('ax')
    lkwargs = {'linestyle':'dashed', 'lw': 1.5, 'color': 'black' }
    fkwargs = {'fontsize': 6, 'ha': 'center', 'va': 'center', 'bbox': dict(facecolor='white', alpha=0.5)}

    
    for seg in segment:
        ftxt = join(fold, f'{seg}_segment.txt')
        if exists(ftxt):
            x, y = np.loadtxt(ftxt, delimiter=',', unpack=True)
            bmap.plot(x, y, latlon=True,  alpha=alpha, **lkwargs)
        else:
            print(f'Not exists: {ftxt}')
            # lon, lat = (-169, 55)
            # x, y = bmap(lon, lat)
            # ax.text(x, y, f'{depth} km', **fkwargs)
    return    

def plot_trench_boundary(m, bm=True, **kwargs):
    """
    Plot plate boundaries.

    :param bm: Basemap vs. Cartopy
    """
    fshp = kwargs.get('fshp', '/home/liu/map_data/plate_boundaries/tectonicplates/PB2002_boundaries.shp')
    # alpha = kwargs.get('alpha', 0.5)
    boundaries = kwargs.get('boundaries')
    # lkwargs = {'lw':3, 'color': 'purple', 'markeredgecolor':'black', 'alpha':0.5 }  
    lkwargs = {'linestyle':'dashdot','lw': 2.1, 'color': 'lime', 'path_effects':[pe.Stroke(linewidth=2.8, foreground='black',alpha=0.5), pe.Normal()], 'alpha':1 }
    green = (0, 1, 0)
    # lkwargs = {'linestyle':'dashdot','lw': 2.7, 'color': green, 'alpha':0.6 }
    
    sf = shapefile.Reader(fshp)
    for sr in sf.shapeRecords():
        if (boundaries is None) or (sr.record['Name'] in boundaries):
            if bm:
                dlines = m.plot(*zip(*sr.shape.points), latlon=True, **lkwargs)
                for line in dlines:
                    line.set_zorder(1)
            else:
                m.plot(*zip(*sr.shape.points), c=c, alpha=alpha, lw=lw, transform=CCRSPC)

    return

def plot_volcano(m, bmap=True, **kwargs):
    """
    Plot plate boundaries.

    :param bm: Basemap vs. Cartopy
    """

    fvolc = kwargs.get('fvolc', '/home/liu/map_data/Alaska_volcano.txt')
    lkwargs = {'linewidths': 0.5, 'edgecolors': 'white', 'color': 'red', 'marker': '^', 'alpha':1}

    if exists(fvolc):
        x, y = np.loadtxt(fvolc, unpack=True)
        m.scatter(x, y, latlon=True, **lkwargs)
    return

def plot_small_volcano(m, bmap=True, **kwargs):
    """
    Plot plate boundaries.

    :param bm: Basemap vs. Cartopy
    """

    fvolc = kwargs.get('fvolc', '/home/liu/map_data/Alaska_volcano.txt')
    lkwargs = {'linewidths': 0.1, 'edgecolors': 'black', 'color': 'red', 'marker': '^', 's': 7, 'alpha':0.6}

    if exists(fvolc):
        x, y = np.loadtxt(fvolc, unpack=True)
        m.scatter(x, y, latlon=True, **lkwargs)
    return

def plot_Yakutat(m, bmap=True, **kwargs):

    ftxt = '/home/liu/map_data/YAK_extent.txt'
    alpha = 0.5
    lkwargs = {'lw': 2, 'color': 'black' }
    x, y = np.loadtxt(ftxt, unpack=True)
    m.plot(x, y, latlon=True,  alpha=alpha, **lkwargs)
    
def mark_basin(bmap, ax, **kwargs):
    wtype = kwargs.get('wtype')
    fkwargs = {'fontsize': 8, 'ha': 'center', 'va': 'center', 'bbox': dict(facecolor='white', alpha=0.5)}
    basin_nm = ['CB', 'CIB', 'YB']

    basin_text = ["1", "2", "3"]
    if wtype == 'Ray':
        basin_loc = [(-163.11, 68.685), (-151.31, 60.27), (-143.6419, 59.518)]
        txt_loc = [(-169.3, 66.41), (-148.5, 54.17), (-141.41, 57.386)]
    else:
        basin_loc = [(-163.11, 68.685), (-151.31, 59.7), (-145., 60.2)]
        txt_loc = [(-169.3, 66.41), (-149.5, 57.5), (-141.41, 57.386)]  

    for txt, _basin_loc, _txt_loc in zip(basin_text, basin_loc, txt_loc):
        x1, y1 = bmap(_basin_loc[0], _basin_loc[1])
        x2, y2 = bmap(_txt_loc[0], _txt_loc[1])
        ax.annotate(txt, xy=(x1, y1), xytext=(x2, y2), arrowprops=dict(facecolor='black', shrink=0.1, width=1., headlength=5, headwidth=2.3), alpha=1)
    return 

def mark_pattern(bmap, ax, per, **kwargs):

    fkwargs = {'fontsize': 8, 'ha': 'center', 'va': 'center', 'bbox': dict(facecolor='white', alpha=0.5)}

    pattern_text = ["1", "2", "3"]

    if per == 20:
        start = [-148.385, 58.40]
        end = [-144.265, 57.2698]
        txt = '1'

    elif per == 30:
        start = [-146.261, 59.755]
        end = [-145.148, 57.924]
        txt = '2'        
    elif per == 50:
        start = [-146.261, 59.755]
        end = [-145.148, 57.924]
        txt = '2'   
    elif per == 80:
        start = [-151.136, 57.47]
        end = [-149.98, 55.326]
        txt = '3'  

    
        x1, y1 = bmap(start[0], start[1])
        x2, y2 = bmap(end[0], end[1])

        ax.annotate(txt, xy=(x1, y1), xytext=(x2, y2), arrowprops=dict(facecolor='black', shrink=0.1, width=1., headlength=5, headwidth=2.3), alpha=1)
    return 

def plt_slabcontour(bmap, **kwargs):
    """
    contour of 80 s Rayleigh high-speed anomaly 
    """

    ax = kwargs.get('ax')
    lkwargs = {'linestyle':'dashed', 'lw': 1.5, 'color': 'black' }
    fkwargs = {'fontsize': 6, 'ha': 'center', 'va': 'center', 'bbox': dict(facecolor='white', alpha=0.5)}
    
    fnm = '/home/liu/map_data/AK/slab_80s_contour.txt'
    if exists(fnm):
        x, y = np.loadtxt(fnm, dtype='float', unpack=True)
        bmap.plot(x, y, latlon=True,  alpha=1, **lkwargs)
    else:
        print(f'Not exists: {fnm}')
    return 
# --------------#
# plot meshgrid
# --------------#
def plt_mesh(bmap, x, y, z, **kwargs):
    """
    Mesh plot on map.
    TODO: kwargs_mesh, kwargs_cbar

    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html
    https://basemaptutorial.readthedocs.io/en/latest/plotting_data.html#pcolormesh
    parameter
    -----
    vmin/vmax :: colorbar range (it is deprecated to use vmin/vmax when norm is given)
    norm :: normalize instance scales the data values to canonical colormap range [0, 1]
    latlon :: if latlon keyword is set to True, x,y are interpreted as longitude and latitude in degrees
    levels :: 
    """
    vmin = kwargs.get('vmin')
    vmax = kwargs.get('vmax')
    # alpha  = kwargs.get('alpha', 1)
    shading = kwargs.get('shading', 'gouraud')
    
    latlon  = kwargs.get('latlon', True)
    colorbar = kwargs.get('colorbar', True)
    cm = get_cmap(**kwargs)

    levels = kwargs.get('levels', None)
    norm = None
    if levels is not None:
        norm = PiecewiseNorm(levels=levels)

    im = bmap.pcolormesh(
        x, y, z,
        latlon=latlon,
        cmap=cm,
        shading=shading,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        zorder = 1,
    )

    if colorbar:
        cb = plt_colorbar(bmap=bmap, cm=cm, im=im, norm=norm, **kwargs)

    return im, cm, norm

# --------------#
# plot quiver #
# --------------#
def plt_quiver(bmap, x, y, u=None, v=None, amplitude=None, azimuth=None, ax=None, step=1, kwargs_key={}, **kwargs):
    """
    Quiver plot on map.

    :param step: Sampling interval.
    """
    kwargs_def = {
        'latlon': True,
        'pivot': 'middle',
        'headaxislength': 0,
        'headlength': 0,
        'headwidth': 3,
        'color': 'r',
        'width':0.006 #0.005,
    }

    if u is None:
        u = amplitude * np.sin(np.deg2rad(azimuth))
        v = amplitude * np.cos(np.deg2rad(azimuth))

    # u and v are left-right and up-down
    # https://basemaptutorial.readthedocs.io/en/latest/plotting_data.html#quiver
    u, v = bmap.rotate_vector(uin=u, vin=v, lons=x, lats=y)

    n = step
    # pdb.set_trace()
    q = bmap.quiver(
        x[::n, ::n], y[::n, ::n], u[::n, ::n], v[::n, ::n],
        **{**kwargs_def, **kwargs}
    )
    
    kwargs_key_def = {
        'X': .9,
        'Y': .05,
        'U': 2,
        'coordinates': 'axes',
        'labelpos': 'S',
        'labelsep': .03,
        'label': '',
        'fontproperties': {'size': 10},
        
    }
    if ax is not None:
        dquivers = ax.quiverkey(q, zorder=10, **{**kwargs_key_def, **kwargs_key})
        dquivers.set_zorder(10)
    return q

# --------------#
# plot scater #
# --------------#
def plt_scatter(bmap, lon, lat, labels=[], ka_mk={}, ka_txt={}):
    """
    Scatter plot on map.
    parameter
    ----- 
    ka_mk :: for scatter setting
    ka_txt :: for label txt setting
    :param tx, ty: shift of station names
    """
    ka_def = {'latlon': True, 'linewidths': 0, 'edgecolors': 'k', 'color':'red', 'marker': '^' }

    cmap = get_cmap(**ka_mk)
    norm = None

    levels = ka_mk.get('levels', None)
    if levels is not None:
        norm = PiecewiseNorm(levels=levels)
    ka_mk.update({'cmap': cmap, 'norm': norm})

    im = bmap.scatter(lon, lat, **{**ka_def, **ka_mk})

    if ka_mk.get('c') and (cmap is not None) and ka_mk.get('colorbar', True):
        ka_mk.pop('cmap')
        plt_colorbar(bmap, cmap, im=im, norm=norm, **ka_mk)

    if labels:
        tx = ka_txt.get('tx', 10000)
        ty = ka_txt.get('ty', tx)
        fontdict = ka_txt.get('fontdict', {})
        x, y = bmap(lon, lat)
        for _x, _y, lab in zip(x, y, labels):
            plt.text(_x+tx, _y+ty, lab, **fontdict)

    return bmap, cmap, norm

# --------------#
# plot wedge fan #
# --------------#
def plt_wedge_diagram(ax, mapx, mapy, histArr, **kwargs):
    """
    parameters
    -----
    histArr :: e.g. np.array([24., 7., 9.,4., 4., 0.,2.,0.,3.,5.,3., 3.,2., 367., 765., 934., 787., 223.])

    reference
    -----
    plot on basemap:
    https://stackoverflow.com/a/55890475
    https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.inset_axes.html
    """
    mapwidth = kwargs.get('mapwidth',  0.3)
    cmap = kwargs.get('cmap', None)
    norm = kwargs.get('norm', None)

    ax_h = inset_axes(ax, width=mapwidth, height=mapwidth, loc=10, \
        bbox_to_anchor=(mapx, mapy), \
        bbox_transform=ax.transData,\
        borderpad=0, \
        axes_kwargs={'alpha': 1, 'visible':True})
    

    gap = 20.
    theta = np.arange(18) * gap
    # histArr = np.ma.masked_where(histArr==0., histArr)
    # x = np.ma.array(x, mask=histArr.mask)
    ind = np.where(histArr==0)
    histArr = np.delete(histArr, ind)
    if len(ind) == 20:
        return 
    theta = np.delete(theta, ind)

    # cmap
    if cmap is None:
        cmap = copy.copy(mpl.cm.get_cmap('viridis'))
        cmap = discrete_cmap(20,  cmap)
        cmap = cmap.reversed()
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=500)

    colors = cmap(norm(histArr))

    patches = []
    for i, (_theta, _num) in enumerate(zip(theta, histArr)):
        r = 0.5
        _theta -= 90.
        _theta *= -1 
        theta1 = _theta - 10.
        theta2 = _theta + 10. 
        color = colors[i, :]
        wedge = Wedge((0.5, 0.5), r, theta1, theta2)
        patches.append(wedge)

    pcollection = PatchCollection(patches, alpha=1, cmap=cmap, norm=norm)
    pcollection.set_array(histArr)


    ax_h.add_collection(pcollection)
    ax_h.axis('off')
    # sm = ScalarMappable(cmap=cmap, norm=data_norm)
    # cbar = plt.colorbar(sm, pad=0.1)
    return 

# --------------#
# color map #
# --------------#
def get_cmap(**kwargs):

    cmnm = kwargs.get('cmap', None)
    if cmnm is None:
        return
    cpt_dir = kwargs.get('cpt_dir', '/home/liu/cpt')

    try:
        cmap = mpl.cm.get_cmap(cmnm)
    except ValueError:
        try:
            cmap = vars(cmocean.cm)[cmnm]
        except KeyError:
            try:
                cmap = pycpt.load.gmtColormap(os.path.join(cpt_dir, f'{cmnm}.cpt'))
            except FileNotFoundError:
                cmap = getattr(scm, cmnm)

    n = kwargs.get('n')
    if n:
        return discrete_cmap(n, cmap)

    c_mask = kwargs.get('c_mask', [1, 1, 1, 0])
    if c_mask is not None:
        cmap.set_bad(c_mask)
        # cmap.set_under(c_mask)
        # cmap.set_over(c_mask)

    return cmap

def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a

        Note that if base_cmap is a string or None, you can simply do
           return plt.cm.get_cmap(base_cmap, N)
        The following works for string, None, or a colormap instance:
    """
    # base = plt.cm.get_cmap(base_cmap)
    base = get_cmap(cmap=base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    # https://stackoverflow.com/a/43385541
    return mpl.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

# --------------#
# color bar #
# --------------#
def plt_colorbar(bmap, cm, im=None, norm=None, **kwargs):
    """
    https://basemaptutorial.readthedocs.io/en/latest/utilities.html#colorbars
    """
    pad = kwargs.get('pad', .3)
    label = kwargs.get('label', '')
    ndecimal = kwargs.get('ndecimal', 2)
    fontsize = kwargs.get('fontsize', 10)

    if norm is not None:
        # https://stackoverflow.com/a/22533565/8877268
        sm = plt.cm.ScalarMappable(cmap=cm)
        sm._A = []
        cb = bmap.colorbar(sm, location='bottom', pad=pad, shrink=0.5, extend=kwargs.get('extend_colorbar', 'neither'))
        cb.set_ticks(norm.normed)
        cb.set_ticklabels([f'{level:.{ndecimal}f}' for level in norm.levels])

    else:
        cb = bmap.colorbar(im, 'bottom', pad=pad, extend=kwargs.get('extend_colorbar', 'neither'))
    cb.set_label(label, fontsize=fontsize)

    return cb

def add_colorbar(fig, cmap, norm, rect=[0.25, 0.05, 0.5, 0.02], ndecimal=2, label='', **kwargs):
    """
    Add a colorbar to figure.

    :param rect: [left, bottom, width, height]
    quantities are in fractions of figure width and height.
    """
    # labelsize = kwargs.get('labelsize', 10)
    fontsize = kwargs.get('fontsize', 10)
    labelsize = fontsize
    orientation = kwargs.get('orientation', 'horizontal')

    # axes_kwargs = {'anchor'}

    color_kwargs = {'anchor': 'C'}
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    cb = fig.colorbar(sm, cax=cax, orientation=orientation, extend=kwargs.get('extend_colorbar', 'neither'))
    cb.set_ticks(norm.normed)
    cb.set_ticklabels([f'{level:.{ndecimal}f}' for level in norm.levels])
    # mute
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=labelsize)
    cb.ax.minorticks_off()

    return cb

def addCAxes(ax, location='right', size=0.05, pad=0.13):
    if type(ax) is list:
        axes = ax
        bounds = np.zeros((len(axes),4))
        for i,ax in enumerate(axes):
            bbox = ax.get_position()
            bounds[i,:2] = bbox.intervalx
            bounds[i,2:] = bbox.intervaly
        y = bounds[:,2].min()
        x = 0.5  # bounds[:,0].min(),
        w = bbox.width #   bounds[:,1].max()-bounds[:,0].min(),\
        h = bounds[:,3].max()-bounds[:,2].min()
                
    else:
        bbox = ax.get_position()
        x,y,w,h = bbox.x0,bbox.y0,bbox.width,bbox.height
    if location == 'right':
        pad = 0.05 if pad is None else pad
        rect = [x+w+w*pad,y, w*size, h]
    elif location == 'bottom':
        pad = 0.05 if pad is None else pad
        # rect = [x, y-h*pad-h*size, w, h*size]
        rect = [x - 0.5*w, y-h*pad-h*size, w, h*size]
    else:
        raise ValueError('Not ready yet')
    # return plt.axes(rect)
    return rect


# normalization for color
class PiecewiseNorm(mpl.colors.Normalize):
    """
    Piecewise normalization of a colormap to make it nonlinear.
    previous answer:
    https://stackoverflow.com/a/20146989/8877268
    source: 
    https://stackoverflow.com/a/33883947/8877268
    """
    def __init__(self, levels, vmin=None, vmax=None, clip=False):
        
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        # the input levels
        self.levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self.normed = np.linspace(0, 1, len(levels))


    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        return np.ma.masked_array(np.interp(value, self.levels, self.normed), mask=result.mask)



# --------------#
# lable axes #
# --------------#

def _axes_label(axes, transpose=True):
    """
    Transpose and flatten.
    """
    if transpose:
        axes = map(list, zip(*axes))

    return [i for sub in axes for i in sub]

def label_axes(fig=None, axes=None, loc=(-0.1, 1.1),
               labels=string.ascii_lowercase, brackets='()', **kwargs):
    """
    Walks through axes and labels each.
    https://gist.github.com/tacaswell/9643166

    Parameters
    ----------
    fig : Figure
         Figure object to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.
    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    kwargs : passed to `annotate`
    """
    if axes is None:
        axes = fig.axes

    # re-use labels rather than stop labeling
    labels = it.cycle(labels)

    for ax, lab in zip(axes, labels):
        if brackets == '()':
            lab = f'({lab})'
        elif brackets == ')':
            lab = f'{lab})'

        ax.annotate(lab, xy=loc, xycoords='axes fraction', **kwargs)

    return

