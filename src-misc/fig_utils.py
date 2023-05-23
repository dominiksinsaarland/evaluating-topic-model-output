import matplotlib.style
import matplotlib as mpl
from cycler import cycler

FONT_MONOSPACE = {'fontname':'monospace'}
MARKERS = "o^s*DP1"
COLORS = [
    "#b7423c",
    "#71a548",
    "salmon",
    "darkseagreen",
    "cornflowerblue",
    "orange",
    "seagreen",
    "dimgray",
    "purple",
]

mpl.rcParams['axes.prop_cycle'] = cycler(color=COLORS)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.family'] = "serif"

METRIC_PRETTY_NAME = {
    "bleu": "BLEU",
    "ter": "TER",
    "meteor": "METEOR",
    "chrf": "ChrF",
    "comet": "COMET",
    "bleurt": "BLEURT"
}

COLORS_EXTRA = ["#9c2963", "#fb9e07"]