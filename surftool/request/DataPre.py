import obspy 
import obspy.io.sac
import numpy as np
from os.path import exists, join, isdir, isfile
import os
from numba import jit, float32, int32, boolean, float64, int64
import numba
import pyfftw
import warnings
from tqdm import tqdm 
import tarfile
import shutil
from surftool.meta import Inventory, Eq
from obspy.core import UTCDateTime
import glob
from datetime import datetime
import pdb
from surftool.request import utils
import logging.config


logger = logging.getLogger(__name__)

MONTH = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

# ------------- xcorr specific exceptions ---------------------------------------
class xcorrError(Exception):
    pass

class xcorrIOError(xcorrError, IOError):
    pass

class xcorrHeaderError(xcorrError):
    """
    Raised if header has issues.
    """
    pass

class xcorrDataError(xcorrError):
    """
    Raised if header has issues.
    """
    pass

# ----- functions ----- #
def _tshift_fft(data, dt, tshift):
    """positive means delaying the waveform
    """
    npts    = data.size
    Np2     = int(max(1<<(npts-1).bit_length(), 2**12))
    Xf      = np.fft.rfft(data, n=Np2)
    freq    = 1./dt/Np2*np.arange((Np2/2+1), dtype = float)
    ph_shift= np.exp(-2j*np.pi*freq*tshift)
    Xf2     = Xf*ph_shift
    return np.real(np.fft.irfft(Xf2)[:npts])

@jit(numba.types.Tuple((int64[:, :], int64))(boolean[:]), nopython=True)
def _rec_lst(mask):
    """Get rec list
    """
    reclst = -np.ones((mask.size, 2), dtype = np.int64)
    isrec = False
    irec = 0
    for i in range(mask.size):
        if mask[i]:
            if isrec:
                reclst[irec, 1] = i-1
                irec += 1
                isrec = False
            else:
                continue
        else:
            if isrec:
                # last element
                if i == (mask.size - 1):
                    reclst[irec, 1] = i
                    irec += 1
                continue
            else:
                isrec = True
                reclst[irec, 0] = i
    return reclst, irec

@jit(numba.types.Tuple((int64[:, :], int64))(boolean[:]), nopython=True)
def _gap_lst(mask):
    """Get gap list
    """
    gaplst = -np.ones((mask.size, 2), dtype = np.int64)
    isgap = False
    igap = 0
    for i in range(mask.size):
        if mask[i]:
            if isgap:
                # last element
                if i == (mask.size - 1):
                    gaplst[igap, 1] = i
                    igap += 1
                continue
            else:
                isgap = True
                gaplst[igap, 0] = i
        else:
            if isgap:
                gaplst[igap, 1] = i-1
                igap  += 1
                isgap = False
            else:
                continue
    return gaplst, igap


@jit(float64[:](int64[:, :], int64[:, :], float64[:], int64, int64), nopython=True)
def _fill_gap_vals(gaplst, reclst, data, Ngap, halfw):
    """Get the values for gap fill
    """
    alpha   = -0.5 / (halfw * halfw)
    gaparr  = np.zeros(data.size, dtype = np.float64)
    if gaplst[0, 0] < reclst[0, 0]:
        gaphead = True
    else:
        gaphead = False
    if gaplst[-1, 1] > reclst[-1, 1]:
        gaptail = True
    else:
        gaptail = False
    for igap in range(Ngap):
        igp0    = gaplst[igap, 0]
        igp1    = gaplst[igap, 1]
        tmp_npts= igp1 - igp0 + 1
        if gaphead and igap == 0:
            ilrec   = 0
            irrec   = 0
        elif gaptail and igap == (Ngap - 1):
            ilrec   = -1
            irrec   = -1
        elif gaphead:
            ilrec   = igap - 1
            irrec   = igap
        else:
            ilrec   = igap
            irrec   = igap + 1
        il0     = reclst[ilrec, 0]
        il1     = reclst[ilrec, 1]
        ir0     = reclst[irrec, 0]
        ir1     = reclst[irrec, 1]
        if (il1 - il0 + 1 ) < halfw:
            lmean   = data[il0:(il1+1)].mean()
            lstd    = data[il0:(il1+1)].std()
        else:
            lmean   = data[(il1-halfw+1):(il1+1)].mean()
            lstd    = data[(il1-halfw+1):(il1+1)].std()
        if (ir1 - ir0 + 1 ) < halfw:
            rmean   = data[ir0:(ir1+1)].mean()
            rstd    = data[ir0:(ir1+1)].std()
        else:
            rmean   = data[ir0:(ir0+halfw)].mean()
            rstd    = data[ir0:(ir0+halfw)].std()
        if gaphead and igap == 0:
            lstd= 0
        elif gaptail and igap == (Ngap - 1):
            rstd= 0
        if tmp_npts == 1:
            gaparr[igp0]    = (lmean+rmean)/2. + np.random.uniform(-(lstd+rstd)/2, (lstd+rstd)/2)
        else:
            imid    = int(np.floor(tmp_npts/2))
            for i in range(tmp_npts):
                j           = i + igp0
                slope       = (rmean - lmean)/tmp_npts * i + lmean
                if i < imid:
                    gsamp   = np.exp(alpha * i * i)
                    tmpstd  = lstd
                else:
                    gsamp   = np.exp(alpha * (tmp_npts - i - 1) * (tmp_npts - i - 1))
                    tmpstd  = rstd
                gaparr[j]   = gsamp * np.random.uniform(-tmpstd, tmpstd) + slope
    return gaparr


class DataPre(Inventory, Eq):
    """
    For data pre-processing 
    """
    def __init__(self):
        super().__init__()
        self.catalog = obspy.core.event.Catalog()

        return 

    def seedlabel(self, time):
        """
        """ 
        return f'{time.year}_{MONTH[time.month]}_{time.day}_{time.hour}_{time.minute}_{time.second}'

    def seedlabel_py(self, time):
        return 'E{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour, time.minute, time.second)

    def find_channel_obs(self, trace, chanrank=['L', 'B', 'H'], obs_channels=['HZ', 'H1', 'H2', 'DH']):
        chantype = None
        for tmpchtype in chanrank:
            ich = 0
            for chan in obs_channels:
                if len(trace.select(channel=f'{tmpchtype}{chan}')) > 0:
                    ich += 1
            if ich == len(obs_channels):
                chantype   = tmpchtype
                break
        return chantype

    def mseed2sac_EQ_obs(self, indir, outdir, outpydir, start_date=None, end_date=None, unit_nm=True, sps=1., rmresp=True, 
        ninterp=2,  eq_window=8400., chanrank=['L', 'B', 'H'], obs_channels=['HZ', 'H1', 'H2', 'DH'], 
        perl=5, perh=200., units='DISP', pfx='EQL_'):
        """
        To untar mseed.tar file, and preprocess of SAC
        parameter
        ----
        outpydir :: for pyAFANT
        eq_window :: window of event records
        units :: output seismogram units ('DISP', 'VEL', 'ACC'. [Default 'DISP'])
        """
        stime = UTCDateTime(start_date)
        etime = UTCDateTime(end_date)

        # check station.inventory
        if len(self.inv) == 0:
            raise ValueError('No station inventory loaded')

        # check Eq catalog
        if len(self.catalog) == 0:
            raise ValueError('No event inventory loaded')
        
        #  setting of filter
        f2 = 1./(perh * 1.2)
        f1 = f2 * 0.8
        f3 = 1./(perl*0.8)
        f4 = f3* 1.2
        Nevent = 0
        for id, event in tqdm(enumerate(self.catalog)):
            pmag = event.preferred_magnitude()
            magnitude = pmag.mag 
            Mtype = pmag.magnitude_type 
            porigin = event.preferred_origin()
            otime = porigin.time
            evlo = porigin.longitude
            evla = porigin.latitude
            try:
                evdp = porigin.depth/1000.
            except:
                print('Event evdp missing!!!')
                continue

            if otime < stime or otime > etime:
                continue

            olabel = self.seedlabel(otime)
            labelpy = self.seedlabel_py(otime)

            print(f'[Event] {olabel}')
            tarwd = join(indir, f'{pfx}{olabel}*.tar.mseed')
            tarlst = glob.glob(tarwd)
            if len(tarlst) == 0: 
                print(f'===== No mseed file: {olabel}')
                continue
            elif len(tarlst) > 1:
                raise ValueError(f'===== More than one mseed file: {olabel}')
            tartemp = tarlst[0]
            tardir = join(indir, (tarlst[0].split('/')[-1])[:-10])
            

            # extract tar 
            tmp = tarfile.open(tartemp)
            tmp.extractall(path=indir)
            tmp.close()

            Ndata = 0
            Nnodata = 0
            Nerror = 0

            for station in self.stations:
                net = station.net
                sta = station.sta 
                sta_start = station.start_date 
                sta_end = station.end_date
                stlo = station.lon
                stla = station.lat
                staxml = self.stationXML[station.staid]
                staid = station.staid

                mseedfnm = join(tardir, f'{sta}.{net}.mseed')
                xmlfnm = join(tardir, f'IRISDMC-{sta}.{net}.xml')

                if not exists(mseedfnm):
                    Nnodata  += 1
                    continue

                # load data
                st = obspy.read(mseedfnm)  
                # Get response info from XML
                if not exists(xmlfnm):
                    print(f'==== No RESP file: {staid}')
                    resp_inv = staxml.copy()
                    try:
                        for tr in st:
                            seed_id = f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}'
                            resp_inv.get_response(seed_id=seed_id, datatime=otime)
                    except:
                        print(f"==== No RESP from station XML: {staid}")
                        Nerror += 1
                        continue
                else:
                    resp_inv = obspy.read_inventory(xmlfnm)

                dist, az, baz = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist = dist/1000.
                # starttime       = otime + dist/vmax
                # endtime         = otime + dist/vmin                
                starttime = otime 
                endtime = otime + eq_window

                # merge data
                try:
                    st.merge(method=1, interpolation_samples=ninterp, fill_value='interpolate')
                except:
                    print(f"==== Not same sampling rate: {staid}")
                    Nerror += 1
                    continue

                # choose channel type
                chantype = self.find_channel_obs(trace=st)
                if chantype is None:
                    print(f'==== No channel: {staid}')
                    Nerror += 1
                    continue

                # steam, trim     
                stream = obspy.Stream()
                for chan in obs_channels:
                    temp = st.select(channel=f'{chantype}{chan}')
                    # temp.trim(starttime=starttime, endtime=endtime, pad=False)
                    if len(temp) > 1:
                        print(f'==== More than one locs: {staid}')
                        tNpts = (temp[0].stats.npts)
                        outtr = temp[0].copy()
                        for tmptr in temp:
                            tmptr_N = tmptr.stats.npts
                            if tmptr_N > tNpts:
                                tNpts = tmptr_N
                                outtr = tmptr.copy()
                        stream.append(outtr)
                    else:
                        stream.append(temp[0])
                
                # remove response
                # atarc_download_event.py
                stream.detrend('demean')
                stream.detrend('linear')
                # stream.detrend()

                # problem happend in resample, float divsion by zero. (Feng)
                if abs(stream[0].stats.delta - 1./sps) > (1./sps / 1e3):
                    # print(f'=== Resample: {staid} sps: {stream[0].stats.delta}')
                    stream.filter(type='lowpass', freq=0.5 * sps,  corners=2, zerophase=True) # prefilter
                    stream.resample(sps, no_filter=True)

                # Check stream 
                is_ok, stream = utils.QC_streams(starttime, endtime, stream)
                if not is_ok:
                    Nerror += 1
                    continue
                # need trim 
                # temp.trim(starttime=starttime, endtime=endtime, pad=False)        
                # 
                                        
                if not np.all([tr.stats.npts == int(eq_window*sps) for tr in stream]):
                    raise ValueError(f'=== Error in npts: {staid}') 

                # if not np.all([abs(tr.stats.starttime - starttime) > 1*sps for tr in stream]):
                #     raise ValueError(f'=== Error in starttime: {staid}') 

                # remove responses
                try:
                    stream.remove_response(inventory=resp_inv, pre_filt=[f1, f2, f3, f4])
                    # stream.remove_response(inventory=resp_inv, pre_filt=[f1, f2, f3, f4], output=units)
                except:
                    print(f'=== Error in response remove: {staid}')
                    Nerror  += 1
                    continue

                # used in noise day        
                if unit_nm:
                    for i in range(len(stream)):
                        stream[i].data *= 1e9
                
                # save to SAC
                out_evtdir = join(outdir, olabel)
                outpy_evtdir = join(outpydir, labelpy)

                if not isdir(out_evtdir):
                    os.makedirs(out_evtdir)

                if not isdir(outpy_evtdir):
                    os.makedirs(outpy_evtdir)

                for chan in obs_channels:
                    channel = f'{chantype}{chan}'

                    outfnm = join(out_evtdir, f'{staid}_{channel}.SAC')
                    sactr = obspy.io.sac.SACTrace.from_obspy_trace(stream.select(channel=channel)[0])
                    sactr.o = 0.
                    sactr.evlo = evlo
                    sactr.evla = evla
                    sactr.evdp = evdp
                    sactr.stlo = stlo
                    sactr.stla = stla  
                    sactr.mag = magnitude
                    sactr.dist = dist #(km)
                    sactr.az = az
                    sactr.baz = baz
                    sactr.write(outfnm)
                    # for py-Z
                    if chan[-1] == 'Z':
                        outfnmZ = join(outpy_evtdir, 'EQ_{:s}_{:s}.SAC'.format(labelpy, sta))
                        sactr.write(outfnmZ)

                Ndata += 1
                
            timestr = datetime.now().isoformat().split('.')[0]
            print(f'[EVENT] {Ndata}/{Nerror}/{Nnodata} (data/error/Nodata) groups of traces extracted!')
            shutil.rmtree(tardir)
            if Ndata > 0:
                Nevent += 1
            print(f'[EVENT] extracted event: {Nevent}/{id}')            

    def mseed2sac_ANT(self, indir, outdir, start_date, end_date, unit_nm=True, sps=1., outtype=0, rmresp=True, 
            chtype='LH', channels='ENZ', ntaper=2, halfw=100, tb=1., tlen=86398., tb2=1000., tlen2=84000.,
            perl=5., perh=200., pfx='CML_', delete_tar=False, delete_extract=True, verbose=True, verbose2 = False):
        """
        To untar mseed.tar file, and preprocess of SAC
        parameter
        ----

        """
        if channels != 'EN' and channels != 'ENZ' and channels != 'Z':
            raise xcorrError('Unexpected channels = '+channels)
        starttime = UTCDateTime(start_date)
        endtime = UTCDateTime(end_date)
        curtime = starttime

        Nnodataday = 0
        Nday = 0
        # frequencies for response removal 
        f2 = 1./(perh*1.3)
        f1 = f2*0.8
        f3 = 1./(perl*0.8)
        f4 = f3*1.2
        targetdt = 1./sps

        if ((np.ceil(tb/targetdt)*targetdt - tb) > (targetdt/100.)) or ((np.ceil(tlen/targetdt) -tlen) > (targetdt/100.)) or ((np.ceil(tb2/targetdt)*targetdt - tb2) > (targetdt/100.)) or ((np.ceil(tlen2/targetdt) -tlen2) > (targetdt/100.)):
            raise xcorrError('tb and tlen must both be multiplilier of target dt!')
        
        print ('[%s] [TARMSEED2SAC] Extracting tar mseed from: ' %datetime.now().isoformat().split('.')[0]+indir+' to '+outdir)
        while (curtime <= endtime):
            if verbose:
                print('[%s] [TARMSEED2SAC] Date: ' %datetime.now().isoformat().split('.')[0]+curtime.date.isoformat())
            Nday +=1
            Ndata = 0
            Nnodata = 0
            tarwildcard = join(indir, pfx+str(curtime.year)+'.'+MONTH[curtime.month]+'.'+str(curtime.day)+'.*.tar.mseed')
            tarlst = glob.glob(tarwildcard)
            if len(tarlst) == 0:
                print ('!!! NO DATA DATE: '+curtime.date.isoformat())
                curtime     += 86400
                Nnodataday  += 1
                continue
            elif len(tarlst) > 1:
                print ('!!! MORE DATA DATE: '+curtime.date.isoformat())
            # time stamps for user specified tb and te (tb + tlen)
            tbtime = curtime + tb
            tetime = tbtime + tlen
            tbtime2 = curtime + tb2
            tetime2 = tbtime2 + tlen2
            if tbtime2 < tbtime or tetime2 > tetime:
                raise xcorrError('removed resp should be in the range of raw data ')
            # extract tar files
            tmptar = tarfile.open(tarlst[0])
            tmptar.extractall(path = outdir)
            tmptar.close()
            datedir = join(outdir, (tarlst[0].split('/')[-1])[:-10])
            outdatedir = join(outdir, str(curtime.year)+'.'+ MONTH[curtime.month], str(curtime.year)+'.'+MONTH[curtime.month]+'.'+str(curtime.day))
            # loop over stations
             
            for station in self.stations:
                netcode = station.net
                stacode = station.sta
                staid = station.staid
                skip_this_station = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml = self.stationXML[station.staid]
                mseedfname = join(datedir, stacode+'.'+netcode+'.mseed')
                xmlfname = join(datedir, 'IRISDMC-' + stacode+'.'+netcode+'.xml')
                datalessfname = join(datedir, 'IRISDMC-' + stacode+'.'+netcode+'.dataless')
                # load data 
                if not isfile(mseedfname):
                    if curtime >= staxml[0][0].creation_date and curtime <= staxml[0][0].end_date:
                        print ('*** NO DATA STATION: '+staid)
                        Nnodata  += 1
                    continue
                
                #out SAC file names
                fnameZ = join(outdatedir, 'ft_'+str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'Z.SAC')
                fnameE = join(outdatedir, 'ft_'+str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'E.SAC')
                fnameN = join(outdatedir, 'ft_'+str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'N.SAC')
                if outtype != 0 and channels=='Z':
                    fnameZ  = join(outdatedir, 'ft_'+str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+stacode+'.'+chtype+'Z.SAC')
                # load data
                st = obspy.read(mseedfname)
                st.sort(keys=['location', 'channel', 'starttime', 'endtime']) # sort the stream
                
                #=============================
                # get response information
                # rmresp = True, from XML
                # rmresp = False, from dataless
                #=============================
                if rmresp:
                    if not isfile(xmlfname):
                        print('*** NO RESPXML FILE STATION: '+staid)
                        resp_inv = staxml.copy()
                        try:
                            for tr in st:
                                seed_id = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel
                                resp_inv.get_response(seed_id=seed_id, datatime=curtime)
                        except:
                            print('*** NO RESP STATION: '+staid)
                            Nnodata += 1
                            continue
                    else:
                        try:
                            resp_inv = obspy.read_inventory(xmlfname)
                        except:
                            Nnodata += 1
                            continue
                else:
                    if not isfile(datalessfname):
                        print('*** NO DATALESS FILE STATION: '+staid)
                #===========================================
                # resample the data and perform time shift 
                #===========================================
                ipoplst = []
                
                for i in range(len(st)):
                    # time shift
                    if (abs(st[i].stats.delta - targetdt)/targetdt) < (1e-4) :
                        st[i].stats.delta = targetdt
                        dt = st[i].stats.delta
                        tmpstime = st[i].stats.starttime
                        st[i].data = st[i].data.astype(np.float64) # convert int in gains to float64
                        tdiff = tmpstime - curtime
                        Nt = np.floor(tdiff/dt)
                        tshift = tdiff - Nt*dt
                        if tshift < 0.:
                            raise xcorrError('UNEXPECTED tshift = '+str(tshift)+' STATION:'+staid)
                        # apply the time shift
                        if tshift < dt*0.5:
                            st[i].data = _tshift_fft(st[i].data, dt=dt, tshift = tshift) 
                            st[i].stats.starttime  -= tshift
                        else:
                            st[i].data = _tshift_fft(st[i].data, dt=dt, tshift = tshift-dt ) 
                            st[i].stats.starttime += dt - tshift
                        if tdiff < 0.:
                            print ('!!! STARTTIME IN PREVIOUS DAY STATION: '+staid)
                            st[i].trim(starttime=curtime)

                    # resample and time "shift"
                    else:
                        # print ('!!! RESAMPLING DATA STATION: '+staid)
                        # detrend the data to prevent edge effect when perform prefiltering before decimate
                        st[i].detrend()
                        dt = st[i].stats.delta
                        # change dt
                        factor = np.round(targetdt/dt)
                        if abs(factor*dt - targetdt) < min(dt, targetdt/50.):
                            dt = targetdt/factor
                            st[i].stats.delta = dt
                        else:
                            print('Unexpected dt: ', targetdt, dt)
                            skip_this_station = True
                            # raise ValueError('CHECK!' + staid)
                            break
                        # "shift" the data by changing the start timestamp
                        tmpstime = st[i].stats.starttime
                        tdiff = tmpstime - curtime
                        Nt = np.floor(tdiff/dt)
                        tshift_s = tdiff - Nt*dt
                        if tshift_s < dt*0.5:
                            st[i].stats.starttime -= tshift_s
                        else:
                            st[i].stats.starttime += dt - tshift_s
                        # new start time for trim
                        tmpstime = st[i].stats.starttime
                        tdiff = tmpstime - curtime
                        Nt = np.floor(tdiff/targetdt)
                        tshift_s = tdiff - Nt*targetdt
                        newstime = tmpstime + (targetdt - tshift_s)
                        # new end time for trim
                        tmpetime = st[i].stats.endtime
                        tdiff = tmpetime - curtime
                        Nt = np.floor(tdiff/targetdt)
                        tshift_e = tdiff - Nt*targetdt
                        newetime = tmpetime - tshift_e
                        if newetime < newstime:
                            if tmpetime - tmpstime > targetdt:
                                print (st[i].stats.starttime)
                                print (newstime)
                                print (st[i].stats.endtime)
                                print (newetime)
                                raise ValueError('CHECK!')
                            else:
                                ipoplst.append(i)
                                continue
                        # trim the data
                        st[i].trim(starttime=newstime, endtime=newetime)
                        # decimate
                        try:
                            st[i].filter(type='lowpass', freq=sps/2., zerophase=True) # prefilter
                            st[i].decimate(factor= int(factor), no_filter=True)
                        except:
                            skip_this_station = True
                            break
                        # check the time stamp again, for debug purposes
                        if st[i].stats.starttime != newstime or st[i].stats.endtime != newetime:
                            print (st[i].stats.starttime)
                            print (newstime)
                            print (st[i].stats.endtime)
                            print (newetime)
                            raise ValueError('CHECK start/end time' + staid)
                        if (int((newstime - curtime)/targetdt) * targetdt != (newstime - curtime))\
                            or (int((newetime - curtime)/targetdt) * targetdt != (newetime - curtime)):
                            print (newstime)
                            print (newetime)
                            raise ValueError('CHECK start/end time' + staid)
                
                if skip_this_station:
                    continue
                if len(ipoplst) > 0:
                    print ('!!! poping traces!'+staid)
                    npop = 0
                    for ipop in ipoplst:
                        st.pop(index = ipop - npop)
                        npop += 1
                #====================================================
                # merge the data: taper merge overlaps or fill gaps
                #====================================================

                st2 = obspy.Stream()
                isZ = False
                isEN = False
                locZ = None
                locEN = None
                # Z component
                if channels[-1] == 'Z':
                    StreamZ = st.select(channel=chtype+'Z')
                    StreamZ.sort(keys=['starttime', 'endtime'])
                    try:
                        StreamZ.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                    except: 
                        print(f'!!!!!!-----!!!!! Stream merge problem: {staid}, Incompatible trace, differing sampling rate!!!!')
                        continue   

                    if len(StreamZ) == 0:
                        print ('!!! NO Z COMPONENT STATION: '+staid)
                        Nrec = 0
                        Nrec2 = 0
                    else:
                        trZ = StreamZ[0].copy()
                        gapT = max(0, trZ.stats.starttime - tbtime) + max(0, tetime - trZ.stats.endtime)
                        # more than two traces with different locations, choose the longer one
                        if len(StreamZ) > 1:
                            for tmptr in StreamZ:
                                tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                                if tmpgapT < gapT:
                                    gapT= tmpgapT
                                    trZ = tmptr.copy()
                            if verbose2:
                                print ('!!! MORE Z LOCS STATION: '+staid+', CHOOSE: '+trZ.stats.location)
                            locZ = trZ.stats.location
                        if trZ.stats.starttime > tetime or trZ.stats.endtime < tbtime:
                            print ('!!! NO Z COMPONENT STATION: '+staid)
                            Nrec = 0
                            Nrec2 = 0
                        else:
                            # trim the data for tb and tb + tlen
                            trZ.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                            if isinstance(trZ.data, np.ma.masked_array):
                                maskZ = trZ.data.mask
                                dataZ = trZ.data.data
                                sigstd = trZ.data.std()
                                sigmean = trZ.data.mean()
                                if np.isnan(sigstd) or np.isnan(sigmean):
                                    raise xcorrDataError('NaN Z SIG/MEAN STATION: '+staid)
                                dataZ[maskZ] = 0.
                                # gap list
                                gaparr, Ngap = _gap_lst(maskZ)
                                gaplst = gaparr[:Ngap, :]
                                # get the rec list
                                Nrecarr, Nrec = _rec_lst(maskZ)
                                Nreclst = Nrecarr[:Nrec, :]
                                if np.any(Nreclst<0) or np.any(gaplst<0):
                                    raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                # values for gap filling
                                try:
                                    fillvals = _fill_gap_vals(gaplst, Nreclst, dataZ, Ngap, halfw)
                                except: 
                                    print(f'!!! Gap filling problem: {staid}')
                                    continue
                                trZ.data = fillvals * maskZ + dataZ
                                if np.any(np.isnan(trZ.data)):
                                    raise xcorrDataError('NaN Z DATA STATION: '+staid)
                                # rec lst for tb2 and tlen2
                                im0  = int((tb2 - tb)/targetdt)
                                im1  = int((tb2 + tlen2 - tb)/targetdt) + 1
                                maskZ2 = maskZ[im0:im1]
                                Nrecarr2, Nrec2 = _rec_lst(maskZ2)
                                Nreclst2 = Nrecarr2[:Nrec2, :]
                            else:
                                Nrec = 0
                                Nrec2  = 0
                            st2.append(trZ)
                            isZ = True
                    if Nrec > 0:
                        if not isdir(outdatedir):
                            os.makedirs(outdatedir)
                        with open(fnameZ+'_rec', 'w') as fid:
                            for i in range(Nrec):
                                fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                    if Nrec2 > 0:
                        if not isdir(outdatedir):
                            os.makedirs(outdatedir)
                        print ('!!! GAP Z  STATION: '+staid)
                        with open(fnameZ+'_rec2', 'w') as fid:
                            for i in range(Nrec2):
                                fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                # EN component
                if len(channels)>= 2:
                    if channels[:2] == 'EN':
                        # 2021-07-8: For 2006-03-20, AK.DCPH. incompatible traces with same id
                        try: 
                            StreamE = st.select(channel=chtype+'E')
                            StreamE.sort(keys=['starttime', 'endtime'])
                            StreamE.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                            StreamN = st.select(channel=chtype+'N')
                            StreamN.sort(keys=['starttime', 'endtime'])
                            StreamN.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                        except: 
                            print(f'!!!!!!-----!!!!! Stream merge problem: {staid}, Incompatible traces')
                            continue
                        Nrec = 0
                        Nrec2 = 0
                        if len(StreamE) == 0 or (len(StreamN) != len(StreamE)):
                            if verbose2:
                                print ('!!! NO E or N COMPONENT STATION: '+staid)
                            Nrec = 0
                            Nrec2 = 0
                        else:
                            trE = StreamE[0].copy()
                            trN = StreamN[0].copy()
                            gapT = max(0, trE.stats.starttime - tbtime) + max(0, tetime - trE.stats.endtime)
                            # more than two traces with different locations, choose the longer one
                            if len(StreamE) > 1:
                                for tmptr in StreamE:
                                    tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                                    if tmpgapT < gapT:
                                        gapT= tmpgapT
                                        trE = tmptr.copy()
                                if verbose2:
                                    print ('!!! MORE E LOCS STATION: '+staid+', CHOOSE: '+trE.stats.location)
                                locEN = trE.stats.location
                                trN = StreamN.select(location=locEN)[0]
                            if trE.stats.starttime > tetime or trE.stats.endtime < tbtime or\
                                    trN.stats.starttime > tetime or trN.stats.endtime < tbtime:
                                print ('!!! NO E or N COMPONENT STATION: '+staid)
                                Nrec = 0
                                Nrec2 = 0
                            else:
                                # trim the data for tb and tb+tlen
                                trE.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                                trN.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                                ismask = False
                                if isinstance(trE.data, np.ma.masked_array):
                                    mask = trE.data.mask.copy()
                                    dataE = trE.data.data.copy()
                                    ismask = True
                                else:
                                    dataE = trE.data.copy()
                                if isinstance(trN.data, np.ma.masked_array):
                                    if ismask:
                                        mask += trN.data.mask.copy()
                                    else:
                                        mask = trN.data.mask.copy()
                                        ismask = True
                                    dataN = trN.data.data.copy()
                                else:
                                    dataN = trN.data.copy()
                                allmasked = False
                                if ismask:
                                    allmasked = np.all(mask)
                                if ismask and (not allmasked) :
                                    sigstdE = trE.data.std()
                                    sigmeanE = trE.data.mean()
                                    sigstdN = trN.data.std()
                                    sigmeanN = trN.data.mean()
                                    if np.isnan(sigstdE) or np.isnan(sigmeanE) or \
                                        np.isnan(sigstdN) or np.isnan(sigmeanN):
                                        raise xcorrDataError('NaN EN SIG/MEAN STATION: '+staid)
                                    dataE[mask] = 0.
                                    dataN[mask] = 0.
                                    # gap list
                                    gaparr, Ngap = _gap_lst(mask)
                                    gaplst = gaparr[:Ngap, :]
                                    # get the rec list
                                    Nrecarr, Nrec = _rec_lst(mask)
                                    Nreclst = Nrecarr[:Nrec, :]
                                    if np.any(Nreclst<0) or np.any(gaplst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # values for gap filling
                                    fillvalsE = _fill_gap_vals(gaplst, Nreclst, dataE, Ngap, halfw)
                                    fillvalsN = _fill_gap_vals(gaplst, Nreclst, dataN, Ngap, halfw)
                                    trE.data = fillvalsE * mask + dataE
                                    trN.data = fillvalsN * mask + dataN
                                    if np.any(np.isnan(trE.data)) or np.any(np.isnan(trN.data)):
                                        raise xcorrDataError('NaN EN DATA STATION: '+staid)
                                    if np.any(Nreclst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # rec lst for tb2 and tlen2
                                    im0 = int((tb2 - tb)/targetdt)
                                    im1 = int((tb2 + tlen2 - tb)/targetdt) + 1
                                    mask = mask[im0:im1]
                                    Nrecarr2, Nrec2 = _rec_lst(mask)
                                    Nreclst2 = Nrecarr2[:Nrec2, :]
                                else:
                                    Nrec = 0
                                    Nrec2 = 0
                                if not allmasked:
                                    st2.append(trE)
                                    st2.append(trN)
                                    isEN = True
                            if Nrec > 0:
                                if not isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                with open(fnameE+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                                with open(fnameN+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                            if Nrec2 > 0:
                                if not isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                print ('!!! GAP EN STATION: '+staid)
                                with open(fnameE+'_rec2', 'w') as fid:
                                    for i in range(Nrec2):
                                        fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                                with open(fnameN+'_rec2', 'w') as fid:
                                    for i in range(Nrec2):
                                        fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                if (not isZ) and (not isEN):
                    continue
                if not isdir(outdatedir):
                    os.makedirs(outdatedir)

                #====================================================
                # remove trend, response
                #====================================================                
                if rmresp:
                    if tbtime2 < tbtime or tetime2 > tetime:
                        raise xcorrError('removed resp should be in the range of raw data ')
                    st2.detrend()
                    try:
                        st2.remove_response(inventory=resp_inv, pre_filt=[f1, f2, f3, f4]) # default is velocity
                    except:
                        continue
                    if unit_nm: # convert unit from m/sec to nm/sec
                        for i in range(len(st2)):
                            st2[i].data *= 1e9
                    st2.trim(starttime=tbtime2, endtime=tetime2, pad=True, fill_value=0)
                else:
                    fnameZ = join(outdatedir, str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'Z.SAC')
                    fnameE = join(outdatedir, str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'E.SAC')
                    fnameN = join(outdatedir, str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'N.SAC')
                    if outtype != 0 and channels=='Z':
                        fnameZ = join(outdatedir, str(curtime.year)+'.'+ MONTH[curtime.month]+'.'+str(curtime.day)+'.'+stacode+'.'+chtype+'Z.SAC')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sp = obspy.io.xseed.Parser(datalessfname)
                    sp.write_resp(folder=outdatedir)
                    if locZ is not None:
                        respzlst = glob.glob(join(outdatedir+'RESP.'+staid+'*'+chtype+'Z'))
                        keepfname = join(outdatedir, 'RESP.'+staid+'.'+locZ+'.'+chtype+'Z')
                        for respfname in respzlst:
                            if keepfname != respfname:
                                os.remove(respfname)
                    if locEN is not None:
                        respelst = glob.glob(join(outdatedir, 'RESP.'+staid+'*'+chtype+'E'))
                        keepfname = join(outdatedir, 'RESP.'+staid+'.'+locEN+'.'+chtype+'E')
                        for respfname in respelst:
                            if keepfname != respfname:
                                os.remove(respfname)
                        respnlst = glob.glob(join(outdatedir, 'RESP.'+staid+'*'+chtype+'N'))
                        keepfname = join(outdatedir, 'RESP.'+staid+'.'+locEN+'.'+chtype+'N')
                        for respfname in respnlst:
                            if keepfname != respfname:
                                os.remove(respfname)
                # save data to SAC
                if isZ:
                    sactrZ = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=chtype+'Z')[0])
                    sactrZ.write(fnameZ)
                if isEN:
                    sactrE = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=chtype+'E')[0])
                    sactrE.write(fnameE)
                    sactrN = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=chtype+'N')[0])
                    sactrN.write(fnameN)
                Ndata  += 1
            # End loop over stations
            curtime  += 86400
            if verbose:
                print ('[%s] [TARMSEED2SAC] %d/%d (data/no_data) groups of traces extracted!' %(datetime.now().isoformat().split('.')[0], Ndata, Nnodata))
            # delete raw data
            if delete_extract:
                shutil.rmtree(datedir)
            if delete_tar:
                os.remove(tarlst[0])
        # End loop over dates
        print ('[%s] [TARMSEED2SAC] Extracted %d/%d (days_with)data/total_days) days of data' %(datetime.now().isoformat().split('.')[0], Nday - Nnodataday, Nday))

        return

    def mseed2sac_EQ(self, indir, outdir, start_date=None, end_date=None, unit_nm=True, sps=1., 
        rmresp=True, rotate=True,
        ninterp=2,   vmin=1.0, vmax=6.0, channels='Z',
        perl=5, perh=300., units='DISP', pfx='EQL_'):

        """
        To untar mseed.tar file, and preprocess of SAC
        parameter
        ----
        outdir :: for pyAFANT
        eq_window :: window of event records
        units :: output seismogram units ('DISP', 'VEL', 'ACC'. [Default 'DISP'])


        2021-10-20

        """
        if channels != 'EN' and channels != 'Z':
            raise ValueError('Unexpected channels = '+channels)


        stime = UTCDateTime(start_date)
        etime = UTCDateTime(end_date)

        # check station.inventory
        if len(self.inv) == 0:
            raise ValueError('No station inventory loaded')

        # check Eq catalog
        if len(self.catalog) == 0:
            raise ValueError('No event inventory loaded')
        
        #  setting of filter
        f2 = 1./(perh * 1.2)
        f1 = f2 * 0.8
        f3 = 1./(perl*0.8)
        f4 = f3* 1.2
        Nevent = 0
        for id, event in tqdm(enumerate(self.catalog)):
            pmag = event.preferred_magnitude()
            magnitude = pmag.mag 
            Mtype = pmag.magnitude_type 
            porigin = event.preferred_origin()
            otime = porigin.time
            evlo = porigin.longitude
            evla = porigin.latitude
            # evdp = porigin.depth/1e3
            try:
                evdp = porigin.depth/1000.
            except:
                logger.info('Event evdp missing!!!')
                continue


            if otime < stime or otime > etime:
                continue

            olabel = self.seedlabel(otime)
            labelpy = self.seedlabel_py(otime)


            
            
            tarwd = join(indir, f'{pfx}{olabel}.*.tar.mseed')
            tarlst = glob.glob(tarwd)

            if len(tarlst) == 0: 
                # logger.info(f'===== No mseed file: {olabel}')
                continue
            elif len(tarlst) > 1:
                # raise ValueError(f'===== More than one mseed file: {olabel}')
                logger.error(f'===== More than one mseed file: {olabel}')

            logger.info(f'[Event] {olabel}')


            tartemp = tarlst[0]
            tardir = join(indir, (tarlst[0].split('/')[-1])[:-10])
            

            # extract tar 
            tmp = tarfile.open(tartemp)
            tmp.extractall(path=indir)
            tmp.close()

            Ndata = 0
            Nnodata = 0
            Nerror = 0

            for station in self.stations:
                net = station.net
                sta = station.sta 
                staid = station.staid

                sta_start = station.start_date 
                sta_end = station.end_date
                stlo = station.lon
                stla = station.lat
                staxml = self.stationXML[station.staid]
                

                mseedfnm = join(tardir, f'{sta}.{net}.mseed')
                xmlfnm = join(tardir, f'IRISDMC-{sta}.{net}.xml')

                if not exists(mseedfnm):
                    Nnodata  += 1
                    continue

                # load data
                st = obspy.read(mseedfnm)  
                # -----
                # Get response info from XML
                # -----
                if not exists(xmlfnm):
                    logger.info(f'==== No RESP file: {staid}')
                    resp_inv = staxml.copy()
                    try:
                        for tr in st:
                            seed_id = f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}'
                            resp_inv.get_response(seed_id=seed_id, datatime=otime)
                    except:
                        logger.info(f"==== No RESP from station XML: {staid}")
                        Nerror += 1
                        continue
                else:
                    resp_inv = obspy.read_inventory(xmlfnm)

                dist, az, baz = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist = dist/1000.
                # land station
                starttime = otime + dist/vmax
                endtime = otime + dist/vmin                
                # starttime = otime 
                # endtime = otime + eq_window

                # merge data
                try:
                    st.merge(method=1, interpolation_samples=ninterp, fill_value='interpolate')
                except:
                    logger.info(f"==== Not same sampling rate: {staid}")
                    Nerror += 1
                    continue

                # choose channel type
                chantype = self.find_channel(trace=st, channels=channels)
                if chantype is None:
                    logger.info(f'==== No channel: {staid}')
                    Nerror += 1
                    continue

                # steam, trim     
                stream = obspy.Stream()
                for chan in channels:
                    temp = st.select(channel=f'{chantype}{chan}')
                    # 
                    temp.trim(starttime=starttime, endtime=endtime, pad=False)

                    if len(temp) > 1:
                        logger.info(f'==== More than one locs: {staid}')
                        tNpts = (temp[0].stats.npts)
                        outtr = temp[0].copy()
                        for tmptr in temp:
                            tmptr_N = tmptr.stats.npts
                            if tmptr_N > tNpts:
                                tNpts = tmptr_N
                                outtr = tmptr.copy()
                        stream.append(outtr)
                    else:
                        stream.append(temp[0])

                #-----
                # remove response
                #-----

                # From: atarc_download_event.py
                # stream.detrend('demean')
                # stream.detrend('linear')

                stream.detrend()

                # problem happend in resample, float divsion by zero. (Feng)
                if abs(stream[0].stats.delta - 1./sps) > (1./sps / 1e3):
                    # print(f'=== Resample: {staid} sps: {stream[0].stats.delta}')
                    stream.filter(type='lowpass', freq=0.5 * sps,  corners=2, zerophase=True) # prefilter
                    stream.resample(sampling_rate=sps, no_filter=True)

                # Check stream 
                is_ok, stream = utils.QC_streams(starttime, endtime, stream)
                if not is_ok:
                    Nerror += 1
                    continue

                # need trim ?
                # temp.trim(starttime=starttime, endtime=endtime, pad=False)        
                # if not np.all([abs(tr.stats.starttime - starttime) > 1*sps for tr in stream]):
                #     raise ValueError(f'=== Error in starttime: {staid}') 

                # remove responses
                try:
                    stream.remove_response(inventory=resp_inv, pre_filt=[f1, f2, f3, f4])
                    # stream.remove_response(inventory=resp_inv, pre_filt=[f1, f2, f3, f4], output=units)
                except:
                    logger.error(f'=== Error in response remove: {staid}')
                    Nerror  += 1
                    continue

                # used in noise day        
                if unit_nm:
                    for i in range(len(stream)):
                        stream[i].data *= 1e9

                # ------
                # EN -- rotate --> RT
                # ------
                if len(channels) >= 2:
                    if channels[:2] == 'EN' and rotate:
                        try:
                            stream.rotate('NE->RT', back_azimuth = baz)
                        except:
                            logger.error('!!! ERROR in rotation, station %s' %staid)
                            Nnodata     += 1
                            continue
                        out_channels = 'RT'+channels[2:]
                    else:
                        out_channels = channels
                else:
                    out_channels = channels
                
                #-------------
                # save to SAC
                #-------------
                # out_evtdir = join(outdir, olabel)
                # if not isdir(out_evtdir):
                #     os.makedirs(out_evtdir)        
        
                outpy_evtdir = join(outdir, labelpy)
                if not isdir(outpy_evtdir):
                    os.makedirs(outpy_evtdir)

                if out_channels == 'Z':
                    chan = 'Z' 
                elif out_channels == 'RT':
                    chan = 'T'
                else:
                    pdb.set_trace('Error!')

                # for output
                channel = f'{chantype}{chan}'
                # outfnm = join(out_evtdir, f'{staid}_{channel}.SAC')
                # outfnm = join(outpy_evtdir, 'EQ_{:s}_{:s}.SAC'.format(labelpy, staid))
                outfnm = join(outpy_evtdir, 'EQ_{:s}_{:s}.SAC'.format(labelpy, sta))
                sactr = obspy.io.sac.SACTrace.from_obspy_trace(stream.select(channel=channel)[0])
                sactr.o = 0.
                sactr.evlo = evlo
                sactr.evla = evla
                sactr.evdp = evdp
                sactr.stlo = stlo
                sactr.stla = stla  
                sactr.mag = magnitude
                sactr.dist = dist #(km)
                sactr.az = az
                sactr.baz = baz
                sactr.write(outfnm)

                Ndata += 1
                
            timestr = datetime.now().isoformat().split('.')[0]

            logger.info(f'[EVENT] {Ndata}/{Nerror}/{Nnodata} (data/error/Nodata) groups of traces extracted!')
            shutil.rmtree(tardir)
            if Ndata > 0:
                Nevent += 1
            logger.info(f'[EVENT] extracted event: {Nevent}/{id}')     
        return


    def find_channel(self, trace, chanrank=['LH', 'BH', 'HH'], channels='ENZ'):
        chantype = None
        for tmpchtype in chanrank:
            ich = 0
            for chan in channels:
                if len(trace.select(channel=f'{tmpchtype}{chan}')) > 0:
                    ich += 1
            if ich == len(channels):
                chantype   = tmpchtype
                break
        return chantype
