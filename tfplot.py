# Python Open Room Correction (PORC)
# Copyright (c) 2012 Mason A. Green
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# TFPLOT - Smoothed transfer fucntion plotting
#   TFPLOTS(IMPRESP,COLOR, fs, FRACT)
#   Logarithmic transfer function plot from impluse response IMPRESP.
#   A half hanning window is applied before a 2^18 point FFT, then the data is colleced
#   into logaritmically spaced bins and the average power is computed for
#   each bin (100/octave). Then this is power-smoothed by a hanning window, where
#   FRACT defines the fractional-octave smoothing (default is 3, meaning third-octave).
#   The length of the smoothing hanning window is the double compared to the distance
#   defined by FRACT.
#   The sampling frequency is set by FS (default is 44.1 kHz) and the plotting color is set by the COLOR variable
#   (default is 'b').
#
#   TFPLOT(IMPRESP, COLOR, FS, OCTBIN, AVG)
#   Logarithmic transfer function plot from impluse response IMPRESP.
#   A half hanning window is applied before a 2^18 point FFT, then the
#   data is colleced into logaritmically spaced bins and the average
#   response is computed for each bin. OCTBIN sets the number of bins
#   in one octave, the default is 100 (lower numbers mean more smoothing).
#   The sampling frequency is set by FS (default is 44.1 kHz) and the
#   plotting color is set by the COLOR variable (default is 'b').
#
#   If the AVG variable is set to 'power' then the power is averaged
#   in the logaritmic bin, if it is 'abs' then the absolute value. If the
#   AVG parameter is set to 'comp' or omitted, it averages the complex
#   magnitude (i.e., this is the default).
#
#   C. Balazs Bank, 2006-2007.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


def fftfilt(b, x):
    """Smoothed filter frequency response.

    Ported from Octave fftfilt.m
    """
    # Use FFT with the smallest power of 2 which is >= length (x) +
    # length (b) - 1 as number of points ...
    c_x = x.size
    c_b = b.size
    n = int(np.power(2, np.ceil(np.log(c_x + c_b) / np.log(2)), dtype=np.float32))
    y = ifft(fft(x, n) * fft(b, n))
    # Final cleanups: Both x and b are real; y should also be
    return np.real(y)


def plot(data, fs=41000, color='b', fract=3, avg='comp', plots=False):
    """Plotting routine."""
    octbin = 100
    fftsize = 2**18
    logfact = 2**(1. / octbin)
    logn = np.floor(np.log(fs / 2) / np.log(logfact))
    # logarithmic scale from 1 Hz to fs/2
    logscale = np.power(logfact, np.r_[:logn])

    # creating a half hanning window
    wl = data.size
    hann = sp.hanning(wl * 2)
    endwin = hann[wl:2 * wl]
    tf = fft(data * endwin, fftsize)
    magn = np.abs(tf[:int(fftsize / 2)])
    if plot:
        compamp = tf[:int(fftsize / 2)]
    logmagn = np.empty(int(logn))
    fstep = fs / np.float64(fftsize)

    for k in range(logscale.size):
        start = int(np.round(logscale[k] / np.sqrt(logfact) / fstep))
        start = int(np.maximum(start, 1))
        start = int(np.minimum(start, fftsize / 2))
        if plots:
            stop = int(np.round(logscale[k] * np.sqrt(logfact) / fstep) - 1)
            stop = int(np.maximum(stop, start))
        else:
            stop = int(np.round(logscale[k] * np.sqrt(logfact) / fstep))
        stop = int(np.maximum(stop, 1))
        stop = int(np.minimum(stop, fftsize / 2))

        if plots:
            logmagn[k] = np.sqrt(np.mean(np.power(magn[start - 1:stop], 2)))
        else:
            # averaging the complex transfer function
            if avg is 'comp':
                logmagn[k] = np.abs(np.mean(compamp[start - 1:stop]))
            elif avg is 'abs':
                logmagn[k] = np.mean(np.abs(compamp[start - 1:stop]))
            elif avg is 'power':
                logmagn[k] = np.sqrt(np.mean(np.abs(np.power(compamp[start - 1:stop], 2))))

    if plots:
        # creating hanning window
        # fractional octave smoothing
        hl = int(2 * np.round(octbin / fract))
        hh = sp.hanning(hl)

        l = logmagn.size
        logmagn[l - 1:l + hl] = 0

        # Smoothing the log. spaced data by convonvling with the hanning window
        tmp = fftfilt(hh, np.power(logmagn, 2))
        smoothmagn = np.sqrt(tmp[int(hl / 2):int(hl / 2 + l)] / hh.sum(axis=0))

        plt.semilogx(logscale, 20 * np.log10(smoothmagn), color)

    else:
        plt.semilogx(logscale, 20 * np.log10(logmagn), color)
