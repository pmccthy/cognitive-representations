"""
Shared plotting style and helpers.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE_PATH = Path(__file__).parent / "cogreps.mplstyle"


def use_style():
    """Apply the repo-wide matplotlib style globally."""
    plt.style.use(str(STYLE_PATH))


def smooth(x, window):
    """Centred moving average over `window` samples.

    Edges average over the samples that exist, so the output has the same length as the
    input and isn't pulled towards zero at the start and end.
    """
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window)
    total = np.convolve(x, kernel, mode="same")
    count = np.convolve(np.ones_like(x), kernel, mode="same")
    return total / count
