import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# default parameters for all plots
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = ''.join([r'\usepackage{{amsmath}}',
                                               r'\usepackage{mathtools}',
                                               r'\usepackage{helvet}'])

plt.rcParams.update({
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "grid.color": (1, 1, 1, 0),
    "font.family": "sans-serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
         r'\usepackage{{amsmath}}',            # load additional packages
         r'\usepackage{mathtools}',   # unicode math setup
         r'\usepackage{helvet}'
    ])
})

# where to save figures
fig_path = "../figures/prx-resub/"

# define consistent colors and color maps for all figures
nrn_clr = '#cc4f27'
nrn_clr2 = '#f8992e'
syn_clr = '#810f7c'
nrn_clrs = ['#fec551', '#eb7124', '#983720']
nrn_clrs4 = ['#f8992e', '#eb7124', '#cc4f27', '#983720']
nrn_clrs5 = ['#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']
syn_clrs = ['#8c96c6', '#8856a7', '#810f7c']
syn_clrs4 = ['#8c96c6', '#8c6bb1', '#88419d', '#6e016b']
syn_clrs5 = ['#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b']
context_cmap = ['#231f20', '#ec1b8e', '#1b75ba', '#00a14b', '#7e3f98']
synref_clrs = ['#c2e699', '#78c679', '#238443', '#006837']
synref_clrs5 = ['#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32']
alt_synref_clrs = ['#7fcdbb', '#1d91c0', '#253494']

cm = 1./2.54  # centimeters

# font sizes
fs1 = 5.1
fs2 = 6.1
fs3 = 7.1
fs4 = 9.1

panel_prms = {'color': 'black', 'fontsize': fs4}
