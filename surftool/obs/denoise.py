import numpy as np
import obspy 
from os.path import join, exists, isdir
import os 
from tqdm import tqdm 

from surftool.meta import Inventory, Eq, Array
import surftool.meta.station as Std
import surftool.obs.atacr_plotting as plotting

from obspy.core import UTCDateTime
from datetime import datetime
import logging.config
import shutil
from obstools.atacr import StaNoise, DayNoise, TFNoise, EventStream, utils
import pdb
import stdb

import multiprocessing as mp


# setting of logger
logger = logging.getLogger(__name__)

# log_format = '[%(asctime)-12s] %(levelname)-8s - %(message)s'
# log_date_format = '%m-%d %H:%M'
# logging.basicConfig(filename='tomo_{:s}.log'.format(method), filemode='a', format=log_format, datefmt=log_date_format,  level=logging.INFO)
# console                 = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter               = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logging.getLogger().addHandler(console)
# logger                  = logging.getLogger(__name__)

MONTH = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


#---- class ----#
class obs(object):

    def __init__(self):
        self.event = Eq()
        self.array = Array()
    
    def atacr_EQ_denoise(self, indir, outdir, noisedir, start_date='2018-01-01', end_date='2019-12-01', 
            sps=1., overlap=0.3, chan_rank=['L', 'B', 'H'], out_dtype=None,
            method='days', outformat=1,
            saveplot=False, plotpath='.',
            parallel=False, ncore=10):
        """
        Uses obstools to do denoise
        parameter
        ----- 
        indir :: dir of input event sac file
        outdir :: outdir of denoised event sac file 
        start_date :: '2018-05-20' starttime of catalog 
        end_date :: '2019-10-01' endtime of catalog
        sps :: sampling rate
        overlap :: 
        method :： method for calculation of transfer function

        """
        kwargs = {'saveplot': saveplot, 'plotpath': plotpath}
        # for checking
        if not exists(indir): raise ValueError('Wrong EQ SAC fold')
        if not exists(noisedir): raise ValueError('Wrong Noise fold')
        if not exists(outdir): os.mkdir(outdir)

        stime = UTCDateTime(start_date)
        etime = UTCDateTime(end_date)

        cat = self.event.catalog.filter(f"time > {stime}", f"time < {etime}")

        Nnodataev = 0
        Nevent = 0
        Nerror_all = 0

        if len(cat)==0: raise ValueError('Catalog is missing')

        # loop over events
        for event in tqdm(cat):
            pmag = event.preferred_magnitude()
            magnitude  = pmag.mag
            Mtype = pmag.magnitude_type
            event_descrip = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin = event.preferred_origin()
            otime = porigin.time
            timestr = otime.isoformat()
            # evlo = porigin.longitude
            # evla = porigin.latitude
            # evdp = porigin.depth/1000.
            event_id = event.resource_id.id.split('=')[-1]
            if otime < stime or otime > etime:
                continue
            descrip = event_descrip+', '+Mtype+' = '+str(magnitude)
            oyear = otime.year
            omonth = otime.month
            oday = otime.day
            ohour = otime.hour
            omin = otime.minute
            osec = otime.second
            label = f'{oyear}_{MONTH[omonth]}_{oday}_{ohour}_{omin}_{osec}'
            # label = '%d_%s_%d_%d_%d_%d' %(oyear, monthdict[omonth], oday, ohour, omin, osec)
            # lable = 'E{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(oyear, omonth, oday, ohour, omin, osec)
            
            eventdir = join(indir, label)
            if not isdir(eventdir):
                logging.error(f'!!----!! NO DATA: {label} {descrip}')
                Nnodataev   += 1
                continue
            time = datetime.now().isoformat().split('.')[0]
            logging.info(f'[{time}] [ATACR] computing: {timestr} {descrip}')
            #-------------------------
            # Loop over station 
            #-------------------------
            atacr_lst = []
            for sta in self.array.stations:
                # determine if the range of the station 1 matches current month
                # # # if staid != 'XO.LD35':
                # # #     continue

                _atacr = atacr_interface(staid=sta.staid, slon=sta.lon, slat=sta.lat, elevation=sta.elev, 
                datadir=indir, outdir=outdir, noisedir=noisedir, method=method, outformat=outformat,
                otime=otime, overlap=overlap, chan_rank=chan_rank, sps=sps, **kwargs)
                atacr_lst.append(_atacr)
                

            if parallel:     ## cannot control processes     
                with mp.Pool(processes=ncore) as pool:   
                    pool.imap(atacr_map, atacr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on 

            else:
                for _obs in atacr_lst:
                    if not _obs.load_EQ_SAC():
                        continue
                    if not _obs.transfer_func():
                        continue
                    _obs.correct(out_dtype=out_dtype)

        return
           

def atacr_map(atacr_in):
    """
    """
    if not atacr_in.load_EQ_SAC():
        return
    if not atacr_in.transfer_func():
        return
    atacr_in.correct()
    return 


#---- class ----#
class atacr_interface(object):
    """
    Obs denoise for single station

    """
    def __init__(self, staid, slon, slat, elevation, datadir, outdir, noisedir, otime, 
        overlap=0.3, chan_rank=['L', 'H', 'B'], sps=1., method='days', outformat=1, overwrite=True, **kwargs):

        """
        parameter
        ----
        outformat :: 1: datadir/2019_MAY_4_21_2_10/XO.LD44_HHZ.SAC
                  2: datadir/E20190504210210/EQ_E20190504210210_LD44.SAC
        """
        network = staid.split(".")[0]
        station = staid.split(".")[1]
        channel = ""
        location = ['']
        self.saveplot = kwargs.get('saveplot', False)
        self.plotpath = kwargs.get('plotpath', './pic_denoise')

        if self.saveplot and not exists(self.plotpath): os.mkdir(self.plotpath)

        self.stdb_inv = stdb.StDbElement(network=network, station=station, channel=channel,
            location=location, latitude=slat, longitude=slon, elevation=elevation,
            polarity=1., azcorr=0.)
            # startdate, enddate
        self.staid = staid
        self.stlo = slon 
        self.stla = slat

        self.datadir = datadir
        self.outdir = outdir
        self.noisedir = noisedir
        self.otime = otime
        self.overlap = overlap
        
        self.chan_rank = chan_rank
        monthdir = join(noisedir, f'{otime.year:04d}.{MONTH[otime.month]}')
        self.daydir = join(monthdir, f'{otime.year}.{MONTH[otime.month]}.{otime.day}')
        self.sps = sps
        
        self.method = method
        self.format = outformat
        self.stanoise  = StaNoise()
        self.overwrite = overwrite   
        return


    def load_EQ_SAC(self):
        """
        load single EQ event file

        format :: 1: datadir/2019_MAY_4_21_2_10/XO.LD44_HHZ.SAC
                  2: datadir/E20190504210210/EQ_E20190504210210_LD44.SAC        
        """
        targetdt = 1./self.sps

        oyear = self.otime.year
        omonth = self.otime.month
        oday = self.otime.day
        ohour = self.otime.hour
        omin = self.otime.minute
        osec = self.otime.second

        inlabel = f'{oyear}_{MONTH[omonth]}_{oday}_{ohour}_{omin}_{osec}'
        eventdir = join(self.datadir, inlabel)


        if self.format == 1:      
            self.outeventdir = join(self.outdir, inlabel)

        if self.format == 2:
            outlabel = 'E{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(oyear, omonth, oday, ohour, omin, osec)
            self.outeventdir = join(self.outdir, outlabel)            


        chan_type = None

        # load event SAC data
        for cht in self.chan_rank:
            fnm1 = join(eventdir, f'{self.staid}_{cht}H1.SAC')
            fnm2 = join(eventdir, f'{self.staid}_{cht}H2.SAC')
            fnmz = join(eventdir, f'{self.staid}_{cht}HZ.SAC')
            fnmp = join(eventdir, f'{self.staid}_{cht}DH.SAC')
            if exists(fnm1) and exists(fnm2) and exists(fnmz) and exists(fnmp):
                chan_type = cht
                break

        if chan_type is None:
            logger.info(f'Missing Evt: {inlabel}: {self.staid}')
            return False

        self.chan_type = chan_type
        self.sth = obspy.read(fnm1)
        self.sth += obspy.read(fnm2)
        self.sth += obspy.read(fnmz)
        self.stp = obspy.read(fnmp)


        # check fs
        if (abs(self.sth[0].stats.delta - targetdt) > 1e-3) or (abs(self.sth[1].stats.delta - targetdt) > 1e-3) or \
            (abs(self.sth[2].stats.delta - targetdt) > 1e-3) or (abs(self.stp[0].stats.delta - targetdt) > 1e-3):
            raise ValueError(f'!!! CHECK fs : {self.staid}')
        else:
            self.sth[0].stats.delta = targetdt
            self.sth[1].stats.delta = targetdt
            self.sth[2].stats.delta = targetdt
            self.stp[0].stats.delta = targetdt        

        # trim data    
        stime_event = self.sth[-1].stats.starttime
        etime_event = self.sth[-1].stats.endtime
        self.sth.trim(starttime=stime_event, endtime=etime_event, pad=True, nearest_sample=True, fill_value=0.)
        self.stp.trim(starttime=stime_event, endtime=etime_event, pad=True, nearest_sample=True, fill_value=0.)
        
        # setting of the window
        self.eventlength = self.sth[-1].stats.npts / self.sps

        return True

    def transfer_func(self, nday=3):
        """
        """
        if self.method == 'days':
            return self.transfer_func_days(nday=nday)

        elif self.method == 'oneday':
            return self.transfer_func_daily()

    def transfer_func_daily(self):
        """compute daily transfer function
        """
        targetdt = 1./self.sps
        chan_type = self.chan_type
        # window length of transfer function same as event length
        window = self.eventlength
        self.window = window

        # load daily noise data
        daystr = f'{self.otime.year}.{MONTH[self.otime.month]}.{self.otime.day}.{self.staid}'
        dfnm1 = join(self.daydir, f'ft_{daystr}.{chan_type}H1.SAC')
        dfnm2 = join(self.daydir, f'ft_{daystr}.{chan_type}H2.SAC')
        dfnmz = join(self.daydir, f'ft_{daystr}.{chan_type}HZ.SAC')
        dfnmp = join(self.daydir, f'ft_{daystr}.{chan_type}DH.SAC')

        if not(exists(dfnm1) and exists(dfnm2) and exists(dfnmz) and exists(dfnmp)):
            logger.info(f'Missing DayNoise: {daystr}')
            return False

        tr1 = obspy.read(dfnm1)[0]
        tr2 = obspy.read(dfnm2)[0]
        trZ = obspy.read(dfnmz)[0]
        trP = obspy.read(dfnmp)[0]
        
        if abs(tr1.stats.delta - targetdt) > 1e-3 or abs(tr2.stats.delta - targetdt) > 1e-3 or \
                abs(trZ.stats.delta - targetdt) > 1e-3 or abs(trP.stats.delta - targetdt) > 1e-3:
                raise ValueError('!!! CHECK fs :'+ self.staid)
        else:
            tr1.stats.delta = targetdt
            tr2.stats.delta = targetdt
            trP.stats.delta = targetdt
            trZ.stats.delta = targetdt
                
        # trim Noise data
        slidind_wlength = window - int(self.overlap* window)*tr1.stats.delta
        stime_noise = tr1.stats.starttime
        newtime = np.floor((tr1.stats.endtime - stime_noise)/slidind_wlength) * slidind_wlength
        tr1.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        tr2.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        trZ.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        trP.trim(starttime = stime_noise, endtime = stime_noise + newtime)

        if np.all(trP.data == 0.) and not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)):
            self.daynoise = DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=obspy.Trace(), overlap=self.overlap, window=window)
            # self.out_dtype = 'Z2-1'
        elif (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)) and (not np.all(trP.data == 0.)):
            self.daynoise = DayNoise(tr1=obspy.Trace(), tr2=obspy.Trace(), trZ=trZ, trP=trP, overlap=self.overlap, window=window)
            # self.out_dtype = 'ZP'
        elif (not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.))) and (not np.all(trP.data == 0.)):
            self.daynoise = DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=trP, overlap=self.overlap, window = window)
            # self.out_dtype = 'ZP-21'
        else:
            return False

        #-------#
        try:
            self.daynoise.QC_daily_spectra()
            self.daynoise.average_daily_spectra()
            self.tfnoise = TFNoise(self.daynoise)
            self.tfnoise.transfer_func()
            self.ncomp = self.daynoise.ncomp
        except:
            return False
        return True

    def transfer_func_days(self, window=8400., nday=3.):
        """
        compute transfer function based on the noise day for two or more days
        parameter
        ----
        window :: window (sec) for the event recording
        nday :: days of noise used in calculation of transfer 


        """
        # window length 
        # window = 8400.
        self.window = window
        daysec = 86400.
        targetdt = 1./self.sps
        chan_type = self.chan_type
        evtlabel = f'{self.otime.year}.{MONTH[self.otime.month]}.{self.otime.day}.{self.staid}'
        etime = UTCDateTime(f'{self.otime.year}-{self.otime.month:02d}-{self.otime.day:02d}') 
        stime = etime - daysec*nday
        ctime = stime

        Nday = 0
        while ctime < etime: 
            # load daily noise data
            monthdir = join(self.noisedir, f'{ctime.year:04d}.{MONTH[ctime.month]}')
            daystr = f'{ctime.year}.{MONTH[ctime.month]}.{ctime.day}.{self.staid}'
            daydir = f'{ctime.year}.{MONTH[ctime.month]}.{ctime.day}'
            dfnm1 = join(monthdir, daydir, f'ft_{daystr}.{chan_type}H1.SAC')
            dfnm2 = join(monthdir, daydir, f'ft_{daystr}.{chan_type}H2.SAC')
            dfnmz = join(monthdir, daydir, f'ft_{daystr}.{chan_type}HZ.SAC')
            dfnmp = join(monthdir, daydir, f'ft_{daystr}.{chan_type}DH.SAC')


            if not(exists(dfnm1) and exists(dfnm2) and exists(dfnmz) and exists(dfnmp)):
                logger.info(f'Missing DayNoise: {daystr}')
                ctime   += daysec
                continue 

            tr1 = obspy.read(dfnm1)[0]
            tr2 = obspy.read(dfnm2)[0]
            trZ = obspy.read(dfnmz)[0]
            trP = obspy.read(dfnmp)[0]
        
            if abs(tr1.stats.delta - targetdt) > 1e-3 or abs(tr2.stats.delta - targetdt) > 1e-3 or \
                    abs(trZ.stats.delta - targetdt) > 1e-3 or abs(trP.stats.delta - targetdt) > 1e-3:
                    raise ValueError('!!! CHECK fs :'+ self.staid)
            else:
                tr1.stats.delta = targetdt
                tr2.stats.delta = targetdt
                trP.stats.delta = targetdt
                trZ.stats.delta = targetdt
                
            # trim Noise data

            # for event length window
            # for 8400 event length
            delta = 8400. * 10 -1
            stime_noise = tr1.stats.starttime
            tr1.trim(starttime=stime_noise, endtime=stime_noise + delta)
            tr2.trim(starttime=stime_noise, endtime=stime_noise + delta)
            trZ.trim(starttime=stime_noise, endtime=stime_noise + delta)
            trP.trim(starttime=stime_noise, endtime=stime_noise + delta)



            if np.all(trP.data == 0.) and not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)):
                self.stanoise += DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=obspy.Trace(), overlap=self.overlap, window=window)

            elif (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)) and (not np.all(trP.data == 0.)):
                self.stanoise += DayNoise(tr1=obspy.Trace(), tr2=obspy.Trace(), trZ=trZ, trP=trP, overlap=self.overlap, window=window)


            elif (not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.))) and (not np.all(trP.data == 0.)):
                self.stanoise += DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=trP, overlap=self.overlap, window = window)

            else:
                ctime   += 86400.
                continue

            ctime += 86400.
            Nday += 1
        
        if Nday == 0:
            logger.error(f'Not enough noise: {Nday} days, {evtlabel}')
            return False
            
        #-------#
        try:
            self.stanoise.QC_sta_spectra()
            self.stanoise.average_sta_spectra()
            self.tfnoise = TFNoise(self.stanoise)
            self.tfnoise.transfer_func()
            self.ncomp = self.stanoise.ncomp

        except:
            logger.error(f'Error in Cal of transfer function, {evtlabel}')
            return False
        return True

    def correct(self, out_dtype=None):
        """
        compute monthly transfer function
        -----
        out_dtype :: 'ZP' only removes compliance noise; "Z2-1" only removes tilt noise; "ZP-21" removes for all;
                     None: decide by noise components
        """
        if not exists(self.outeventdir):
            os.makedirs(self.outeventdir)    

        tmptime = self.sth[-1].stats.starttime

        tstamp = str(tmptime.year).zfill(4)+'.' + str(tmptime.julday).zfill(3)+'.'
        tstamp = tstamp + str(tmptime.hour).zfill(2) + '.'+str(tmptime.minute).zfill(2)


        if out_dtype is None: 
            if self.ncomp == 2:
                out_dtype = 'ZP'
            elif self.ncomp == 3:
                out_dtype = 'Z2-1'
            elif self.ncomp == 4:
                out_dtype = 'ZP-21'

        
        ncomp = self.ncomp

        eventstream = EventStream(sta=self.stdb_inv, sth=self.sth, stp=self.stp,
                    tstamp=tstamp, lat=self.stla, lon=self.stlo, time=tmptime,
                    window=self.window, sampling_rate=1., ncomp=ncomp)

        eventstream.correct_data(self.tfnoise)


        # save data
        outTrZ = (self.sth.select(channel = '??Z')[0]).copy()
        try: 
            outTrZ.data = eventstream.correct[out_dtype].copy() # .correct is dic
        except:
            logger.error(f'No {out_dtype} result: for {self.staid}')
            return

        if self.format == 1:
            outfnameZ = join(self.outeventdir, f'{self.staid}_{self.chan_type}HZ.SAC')

        elif self.format == 2:
            outlabel = 'E{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(self.otime.year, self.otime.month, self.otime.day, self.otime.hour, self.otime.minute, self.otime.second)
            outfnameZ = join(self.outeventdir, 'EQ_{:s}_{:s}.SAC'.format(outlabel, self.staid.split('.')[1]))
        
        if os.path.isfile(outfnameZ):
            if self.overwrite:
                logging.info(f'Overwrite: {outfnameZ}')
                outTrZ.write(outfnameZ, format='SAC')
        else:
            outTrZ.write(outfnameZ, format='SAC')

        # if os.path.isfile(outfnameZ):
        #     # shutil.copyfile(src = outfnameZ, dst = outfnameZ+'_old')
        #     # os.remove(outfnameZ)
        #     print(f'Skip: {outfnameZ}')
        # else:
        #     outTrZ.write(outfnameZ, format='SAC')

        # plot corrected event
        if self.saveplot:
            label = f'{self.otime.year}_{self.otime.month}_{self.otime.day}_{self.otime.hour}_{self.otime.minute}_{self.otime.second}'
            fnm = join(self.plotpath, f'EQ_{label}_{self.staid}.pdf')
            plot = plotting.fig_event_corrected(eventstream, self.tfnoise.tf_list)
            plot.savefig(fnm, bbox_inches='tight', dpi=300)

        return 