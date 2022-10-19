import numpy as np

class VarParam(object):
    """
    Save parameters for a variable.

    :param factor:
    """
    def __init__(self, name='vel_iso', unit='km/s', mask='mask', un='vel_sem', factor=1,
        diff={'unit': None, 'max_diff': None, 'ndecimal': None, 'nbins': 40},
        kwargs_mesh={'cmap': 'cv', 'ndecimal': 2, 'name': 'Phase speed'},
    ):
        self.name = name
        self.unit = unit
        self.un = un
        self.mask = mask
        self.factor = factor
        self.diff = diff
        self.kwargs_mesh = kwargs_mesh

        if 'unit' not in diff:
            diff['unit'] = unit
        if 'ndecimal' not in diff:
            diff['ndecimal'] = kwargs_mesh['ndecimal']

VAR2PARAM = {
    # isotropic c 
    'vel_iso': VarParam(
        name='vel_iso',
        diff={
            'cmap': 'BlueWhiteOrangeRed_r', 'n': 8,
            'ndecimal': 0,
            'relative': False, 'unit': 'm/s', 'max_diff': 200,  'min_diff': -200,'nbins': 40, 'levels': np.linspace(-200, 200, 5),
            'colorbar': True,
        },
        kwargs_mesh={'cmap': 'cv', 'ndecimal': 2, 'name': 'Phase speed', 'label': 'Phase speed (km/s)', 'perturbation': False},
        ),

    # isotropic c uncertainty (SDOM)
    'vel_sem': VarParam(
        name='vel_sem', 
        unit='m/s', 
        factor=1000,
        kwargs_mesh={'cmap': 'afmhot_r', 'ndecimal': 0, 'name': 'Phase speed uncertainty', 'label': 'Phase speed uncertainty (m/s)',
            'levels': [0, 10, 20, 30, 40],
            'n': 8,
            # 'cmap': 'hot',
            # 'levels': np.arange(0, 550, 50),
            # 'levels': [0, 5, 10, 15, 20, 25],
            # 'n': 5,
        },
        diff={'max_diff': 10, 'min_diff': -10, 'nbins': 20},
        ),

    # mask
    'mask': VarParam(name='mask'),

    # number of source
    'num_ev_qc': VarParam(
        name='num_ev_qc', unit='#',
        kwargs_mesh={'name': 'No. of events', 'cmap': 'YlGnBu',
                     'levels': np.linspace(0, 2500, 6), 'n': 10,
                    #  'levels': np.linspace(0, 1000, 6), 'n': 10,
                     'ndecimal': 0,
                     'label': 'Raypath density',
                     },
        ),

    # A0 
    'A_0': VarParam(
        'A_0', unit='km/s', mask='mask_ani',
        un='vel_sem', # un='sigma_A_0',
        diff={'unit': 'm/s', 'ndecimal': 0, 'max_diff': 8, 'min_diff': -8, 'nbins': 17, 'n': 16, 'levels': np.linspace(-8, 8, 9),},
        kwargs_mesh={'cmap': 'cv', 'ndecimal': 2, 'name': 'Phase speed', 'label': 'Phase speed (km/s)', 'perturbation': False},
        # kwargs_mesh={'cmap': 'broc', 'reverse': True, 'name': 'Phase speed perturbation'}, # roma/vik/broc
        ),

    # uncertainty of A0
    # c: 'hot_r'
    'sigma_A_0': VarParam(
        name='sigma_A_0', unit='m/s', mask='mask_ani', factor=1000,
        kwargs_mesh={'name': r'$\sigma_{A_0}$', 'label': 'Phase speed uncertainty (m/s)', 'cmap': 'afmhot_r', 'levels': [0, 10, 20, 30, 40], 'n': 8, 'ndecimal': 0},
        ),

    # A2
    'A_2': VarParam(
        # Peak-to-peak amplitude in %
        name='A_2', unit='%', mask='mask_ani', un='sigma_A_2', factor=100,
        # for abs
        # diff={'unit': '%', 'max_diff': 3, 'min_diff': -3, 'ndecimal': 1, 'nbins': 17, 'n': 12},

        # for relative
        diff={'cmap': 'BlueWhiteOrangeRed_r', 'unit': '$\sigma$', 'max_diff': 8, 'min_diff': -8,  'ndecimal': 0, 'nbins': 17, 'n': 16, 'levels': np.linspace(-8, 8, 9)},
        kwargs_mesh={'name': 'Anisotropy amplitude',
                     'cmap': 'YlGnBu',
                     # 'cmap': 'lapaz',
                     # 'cmap': 'viridis',
                     # 'cmap': 'cequal',
                     # 'reverse': True,
                     'levels': [0., 1., 2., 3.,],
                     'label': r'$2\psi$ amplitude (%)',
                     'ndecimal': 0,
                     'n': 6, # n>0, descrete
                     },
        # _kwargs_mesh.update({'cmap': 'cv', 'levels': [0, .25, .5, 1, 1.5, 2, 3]})
        ),

    # uncertainty of A2
    'sigma_A_2': VarParam(
        'sigma_A_2', '%', 'mask_ani',
        factor=100,
        kwargs_mesh={'name': 'Anisotropy amplitude SDOM', 
                     'cmap': 'Blues', 
                     'levels': [0., 0.2, 0.4, 0.6, 0.8, 1], 
                     'ndecimal': 1,
                     'n': 10,
                     'label': 'Amplitude uncertainty (%)',
                     }
        ),

    #
    'psi_2': VarParam(
        name='psi_2', unit='degree', mask='mask_ani', un='sigma_psi_2',
        # 'cmap': 'BlueWhiteOrangeRed_r'
        # for abs
        # diff={'abs': True, 'cmap': 'hot_r', 'name': 'Fast direction', 'unit': r'${}^\circ$', 'max_diff': 90, 'min_diff': 0, 'n': 18, 'ndecimal': 0, 'nbins': 9, 'levels': np.linspace(0, 90, 10)},
        # for relative
        # diff={'abs': True, 'cmap': 'hot_r', 'unit': '', 'max_diff': 8, 'min_diff': 0, 'ndecimal': 0, 'nbins': 8, 'n': 8, 'levels': np.linspace(0, 8, 9)},
        # max_diff and min_diff will affect vs_T result by fitting the histgram
 
        diff={'abs': True, 'cmap': 'BlueWhiteOrangeRed', 'unit': '', 'max_diff': 8, 'min_diff': -8, 'ndecimal': 0, 'nbins': 17, 'n': 16, 'levels': np.linspace(-8, 8, 9)},
        ),

    'sigma_psi_2': VarParam(
        'sigma_psi_2', 'degree', 'mask_ani', 
        kwargs_mesh={'name': 'Fast direction SDOM', 'cmap': 'Reds',
                     # 'levels': [0, 3, 6, 12, 18, 30],
                    #  'levels': np.linspace(0, 24, 5),
                     'levels': [0, 10, 20, 30, 40, 50],
                     'label': r'Fast direction uncertainty (${}^\circ$)',
                     'ndecimal': 0,
                     'n': 10,
                     },
        # _kwargs_mesh.update({'cmap': 'cv_nowhite', 'levels': [0, 3, 6, 12, 18, 30], 'ndecimal': 0})
        ),

    'A_1': VarParam(
        # Peak-to-peak amplitude in %
        name='A_1', unit='%', mask='mask_ani', un='sigma_A_1', factor=100,
        # diff={'unit': '%', 'max_diff': 3, 'ndecimal': 1, 'nbins': 40, 'n': 12},
        diff={'cmap': 'PiYG', 'unit': '$\sigma$', 'max_diff': 8, 'min_diff': -8, 'ndecimal': 0, 'nbins': 40, 'n': 16, 'levels': np.linspace(-8, 8, 9)},
        kwargs_mesh={'name': r'$1\psi$ amplitude', 'cmap': 'GnBu', 'levels': [0, 1, 2, 3, 4, 5], 'ndecimal': 0, 'n': 5}, # 'n': 6 # for descrete colorbar
        # _kwargs_mesh.update({'cmap': 'cv', 'levels': [0, .25, .5, 1, 1.5, 2, 3]}) # n setting the descrete colorbar
        ),

    'sigma_A_1': VarParam(
        'sigma_A_1', '%' 'mask_ani', factor=100,
        kwargs_mesh={'name': 'Anisotropy amplitude SDOM', 'cmap': 'Blues', 'levels': np.linspace(0, .8, 5), 'n': 8}
        ),

    'psi_1': VarParam(
        name='psi_1', unit='degree', mask='mask_ani', un='sigma_psi_1',
        diff={'name': 'Fast direction', 'unit': r'${}^\circ$', 'max_diff': 50, 'n': 20, 'ndecimal': 0, 'nbins': 40, 'cmap': 'BlueWhiteOrangeRed_r'},
        # diff={'cmap': 'RdBu', 'unit': '$\sigma$', 'max_diff': 8, 'ndecimal': 0, 'nbins': 40, 'n': 16, 'levels': np.linspace(-8, 8, 9)},
        ),

    'sigma_psi_1': VarParam(
        'sigma_psi_1', 'degree', 
        kwargs_mesh={'name': 'Fast direction SDOM', 'cmap': 'Reds',
                     # 'levels': [0, 3, 6, 12, 18, 30],
                     'levels': np.linspace(0, 24, 5),
                     'ndecimal': 0,
                     'n': 8},
        # _kwargs_mesh.update({'cmap': 'cv_nowhite', 'levels': [0, 3, 6, 12, 18, 30], 'ndecimal': 0})
        ),

    'mask_ani': VarParam( name='mask_ani'),
    'slownessAni': VarParam(name='slownessAni'),
    'mask_bins': VarParam(name='mask_bins'),
    'velAnisem': VarParam(name='velAnisem'),
    'vel_std': VarParam(name='vel_std', unit='m/s', factor=1000,
        kwargs_mesh={'cmap': 'afmhot_r', 'ndecimal': 0, 'name': 'Phase speed uncertainty', 'label': 'SD (m/s)',
            'levels': [0, 100, 200, 300, 400, 500], 'n': 10},
    ),

    'Nbin': VarParam(
        name='Nbin', unit='%',
        kwargs_mesh={'ndecimal': 0, 'name': 'Azimuthal gap', 'levels': np.linspace(0, 20, 6), 'n': 10, 'cmap': 'Blues'},
    ),

    'misfit': VarParam(
        name='misfit', unit='',
        kwargs_mesh={'ndecimal': 1, 'name': r'$\chi_\nu$', 'cmap': 'Reds', 
        'levels': [0, 1, 2, 3, 4, 5], 
        'n': 5, # continuous
        'label':'Misfit',
        'xmin': 0, 
        'xmax': 5,
        'ymax_hist': 30.,
        'nbins': 20,
        'ndecimal': 0}, # 'reverse':True
    ),

    'histArr': VarParam(name='histArr'),

    # added for A2_sin, A2_cos
    'A2_sin': VarParam(
        'A2_sin', unit='m/s', mask='mask_ani', factor=1,
        kwargs_mesh={'name': r'$\{A_2sin}$', 'label': r'$A_{2}sin$ (m/s)', 'cmap': 'BlueWhiteOrangeRed_r', 'levels': np.arange(-80, 100, 20), 'n': 12, 'ndecimal': 0},
        ),    

    'A2_cos': VarParam(
        'A2_cos', unit='m/s', mask='mask_ani', factor=1,
        kwargs_mesh={'name': r'$\{A_2cos}$', 'label': r'$A_{2}cos$ (m/s)', 'cmap': 'BlueWhiteOrangeRed_r', 'levels': np.arange(-80, 100, 20), 'n': 12, 'ndecimal': 0},
        ),  

    # no output; only for plot
    'A2_amp': VarParam(
        'A2_amp', unit='%', mask='mask_ani', factor=100,
        kwargs_mesh={'name': r'$\{A_2}$', 'label': 'A2_amp (%)', 'cmap': 'YlGnBu', 'levels': np.linspace(0, 3, 4), 'ndecimal': 0},
        ), 

    # plot only (output)
    'sigma_A2_sin': VarParam(
        'sigma_A2_sin', unit='m/s', mask='mask_ani', factor=1,
        kwargs_mesh={'name': r'$\sigma_{A_2sin}$', 'label': r'$\frac{sigma_{A_{2}sin}}{A_{2}sin}$ (%)', 'cmap': 'afmhot_r', 'levels': [0, 20, 40, 60, 80], 'n': 10, 'ndecimal': 0},
        ),          
    'sigma_A2_cos': VarParam(
        'sigma_A2_cos', unit='m/s', mask='mask_ani', factor=1,
        kwargs_mesh={'name': r'$\sigma_{A_2cos}$', 'label': r'$\frac{sigma_{A_{2}cos}}{A_{2}cos}$ (%)', 'cmap': 'afmhot_r', 'levels': [0, 20, 40, 60, 80], 'n': 10, 'ndecimal': 0},
        ),    

    # plot only, no output
    'sigma_A2': VarParam(
        'sigma_A2', unit='%', mask='mask_ani', factor=100,
        kwargs_mesh={'name': r'$\sigma_{A_2}$', 'label': 'A2 unc ratio (%)', 'cmap': 'afmhot_r', 'levels': [0, 20, 40, 60, 80], 'n': 10, 'ndecimal': 0},
        ), 
    # plot only, no output    
    'lambda':VarParam(
        'lambda', unit='', mask='mask_ani', factor=1,
        kwargs_mesh={'name': r'$\lambda$', 'label': r'$\lambda$', 'cmap': 'Reds', 'levels': [0, 1, 2, 3, 4, 5], 'n': 5, 'ndecimal': 0},
        ), 

}
