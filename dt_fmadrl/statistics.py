"""Student-t intervals across independent seeds (half-width, undefined for n<2)."""
import math
from statistics import stdev
import numpy as np

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(128)

def t_critical(df):
    if df < 1: raise ValueError('Positive degrees of freedom required')
    factor = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    def cdf(x):
        z = x*(_NODES+1)/2
        return .5+x/2*np.sum(_WEIGHTS*factor*(1+z*z/df)**(-(df+1)/2))
    low, high = 0., 16.
    for _ in range(60):
        mid = (low+high)/2
        if cdf(mid)<.975: low=mid
        else: high=mid
    return (low+high)/2

def ci95(values):
    if len(values)<2: return None
    return t_critical(len(values)-1)*stdev(values)/math.sqrt(len(values))
