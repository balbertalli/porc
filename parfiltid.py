# Python Open Room Correction (PORC)
# Copyright (c) 2012 mason A. Green
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
# PARFILTID - System identification in the form of second-order parallel filters for a given
# pole set.
#
# [Bm,Am,FIRcoeff]=parfiltid(INPUT,OUTPUT,P,nfir); identifies the second-order sections
# [Bm,Am] and the coefficients of the FIR part (FIRcoeff) for a given
# pole set P. The parameters are set such that, when the INPUT signal is
# filtered by the parallel filter, it gives an output which is the closest
# to the OUTPUT vector in the lS sense. The number of taps in the parallel
# FIR filter is set by nfir. The default is nfir=1, in this case FIRcoeff
# is a simple gain. The only difference from the PARFIlTDES function is that
# now the input can be arbitrary, and not just a unit pulse as for filter design.
#
# The Bm and Am matrices are containing the [b0 b1]' and [1 a0 a1]'
# coefficients for the different sections in their columns. For example,
# Bm(:,3) gives the [b0 b1]' parameters of the third second-order
# section. These can be used by the filter command separatelly (e.g., by
# y=filter(Bm(:,3),Am(:,3),x), or by the PARFIlT command.
#
# Note that this function does not support pole multiplicity, so P should
# contain each pole only once.
#
# more details about the parallel filter can be found in the papers
#
# Balazs Bank, "Perceptually motivated Audio Equalization Using Fixed-Pole Parallel
# Second-Order Filters", IEEE Signal Processing letters, 2008.
# http://www.acoustics.hut.fi/go/spl08-parfilt
#
# Balazs Bank, "Direct Design of Parallel Second-order Filters for
# Instrument Body modeling", International Computer music Conference,
# Copenhagen, Denmark, Aug. 2007.
# http://www.acoustics.hut.fi/go/icmc07-parfilt
# C. Balazs Bank, Helsinki University of Technology, 2007.

import numpy as np
import scipy.signal as sig


def parfiltid(input, out, p, nfir=1):
    """Parallel filter design."""
    # We don't want to have any poles in the origin; For that we have the parallel FIR part.
    # Remove nonzeros
    p = p[p.nonzero()]

    # making the filter stable by flipping the poles into the unit circle
    for k in range(p.size):
        if abs(p[k]) > 1:
            p[k] = 1.0 / np.conj(p[k])

    # Order it to complex pole pairs + real ones afterwards
    p = np.sort_complex(p)

    # in order to have second-order sections only (i.e., no first order)
    pnum = len(p)  # number of poles
    ppnum = 2 * np.floor(pnum / 2)  # the even part of pnum
    odd = 0

    # if pnum is odd
    if pnum > ppnum:
        odd = 1

    outl = len(out)
    inl = len(input)

    # making input the same length as the output

    if inl > outl:
        input = input[:outl]

    if inl < outl:
        input = np.hstack([input, np.zeros(outl - inl, dtype=np.float64)])

    l = outl

    # Allocate memory
    m = np.zeros((input.size, p.size + nfir), dtype=np.float64)

    # constructing the modeling signal matrix
    for k in range(0, int(ppnum), 2):  # second-order sections
        # impluse response of the two-pole filter
        resp = sig.lfilter(np.array([1]), np.poly(p[k:k + 2]), input)
        m[:, k] = resp
        # the response delayed by one sample
        m[:, k + 1] = np.hstack((0., resp[:l - 1]))

    # if the number of poles is odd, we have a first-order section
    if odd:
        resp = sig.lfilter(np.array([1]), np.poly(p[-1]), input)
        m[:, pnum - 1] = resp

    # parallel FIR part
    for k in range(0, nfir):
        m[:, pnum + k] = np.hstack([np.zeros(k, dtype=np.float64), input[:l - k + 1]])

    y = out
    # looking for min(||y-m*par||) as a function of par:
    # least squares solution by equation solving
    mconj = m.conj().T
    a = np.dot(mconj, m)
    b = np.dot(mconj, y)
    par = np.linalg.solve(a, b)

    # print (np.dot(a, par) == b).all()

    # allocate memory
    size = int(np.ceil(ppnum / 2))
    am = np.zeros((3, size), dtype=np.float64)
    bm = np.zeros((2, size), dtype=np.float64)

    # constructing the bm and am matrices
    for k in range(0, size):
        am[:, k] = np.poly(p[2 * k:2 * k + 2])
        bm[:, k] = np.hstack(par[2 * k:2 * k + 2])

    # we extend the first-order section to a second-order one by adding zero coefficients
    if odd:
        am = np.append(am, np.vstack(np.hstack([np.poly(p[pnum]), 0.])), 1)
        bm = np.append(bm, np.vstack([par[pnum], 0.]), 1)

    fir = []

    # constructing the fir part
    if nfir > 0:
        fir = np.hstack(par[pnum:pnum + nfir])

    return bm, am, fir
