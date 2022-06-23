import obspy 
import numpy as np
import math 
from obspy.core import read, Stream, Trace, AttribDict 
from scipy.signal import savgol_filter
import pdb


def traceshift(trace, tt):
    """
    Function to shift traces in time given travel time
    from: https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
    """

    # Define frequencies
    nt = trace.stats.npts
    dt = trace.stats.delta
    freq = np.fft.fftfreq(nt, d=dt)

    # Fourier transform
    ftrace = np.fft.fft(trace.data)

    # Shift
    for i in range(len(freq)):
        ftrace[i] = ftrace[i]*np.exp(-2.*np.pi*1j*freq[i]*tt)

    # Back Fourier transform and return as trace
    rtrace = trace.copy()
    rtrace.data = np.real(np.fft.ifft(ftrace))

    # Update start time
    rtrace.stats.starttime -= tt

    return rtrace


def QC_streams(start, end, st, verbose=False):
    """
    from:  https://github.com/nfsi-canada/OBStools/blob/master/obstools/atacr/utils.py
    """

    # Check start times
    if not np.all([tr.stats.starttime == start for tr in st]):
        if verbose:
            print("* Start times are not all close to true start: ")
            [print("*   "+tr.stats.channel+" " + str(tr.stats.starttime)+" "+ str(tr.stats.endtime)) for tr in st]
            print("*   True start: "+str(start))
            print("* -> Shifting traces to true start")

        delay = [tr.stats.starttime - start for tr in st]
        st_shifted = Stream(traces=[traceshift(tr, dt) for tr, dt in zip(st, delay)])
        st = st_shifted.copy()
        
    # # Check sampling rate
    # sr = st[0].stats.sampling_rate
    # sr_round = float(floor_decimal(sr, 0))
    # if not sr == sr_round:
    #     print("* Sampling rate is not an integer value: ", sr)
    #     print("* -> Resampling")
    #     st.resample(sr_round, no_filter=False)

    # Try trimming
    dt = st[0].stats.delta
    try:
        st.trim(start, end-dt, fill_value=0., pad=True)
    except:
        print("* Unable to trim")
        print("* -> Skipping")
        print("**************************************************")
        return False, None

    # Check final lengths - they should all be equal if start times
    # and sampling rates are all equal and traces have been trimmed
    sr = st[0].stats.sampling_rate
    if not np.allclose([tr.stats.npts for tr in st[1:]], st[0].stats.npts):
        print("* Lengths are incompatible: ")
        [print("*     "+str(tr.stats.npts)) for tr in st]
        print("* -> Skipping")
        print("**************************************************")

        return False, None

    elif not np.allclose([st[0].stats.npts], int((end - start)*sr), atol=1):
        print("* Length is too short: ")
        print("*    "+str(st[0].stats.npts) +
              " ~= "+str(int((end - start)*sr)))
        print("* -> Skipping")
        print("**************************************************")

        return False, None

    else:
        return True, st