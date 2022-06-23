import numpy as np
import obspy 
import pdb
from obspy.core import UTCDateTime
from obspy.geodetics import kilometer2degrees as k2d
from surftool.meta import Eq, Inventory
import smtplib
from tqdm import tqdm

mondict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


class Request(Inventory, Eq):

    def __init__(self):
        super().__init__()

        # self.inventory = Inventory()
        # self.event = Eq()
        return 

    def req_iris_noise(self, start_date=None, end_date=None, skipinv=True, chanrank=['LH', 'BH', 'HH'], channels='ENZ',
        quality='B', name = 'ChuanmingLiu', label='AKHH', email_address='chuanmingliu.cu@gmail.com', 
        send_email=False, iris_email='breq_fast@iris.washington.edu', verbose=True):
        """
        request continuous data for noise analysis
        """

        start_date = obspy.UTCDateTime(start_date)
        end_date = obspy.UTCDateTime(end_date)
        header_str1 = '.NAME %s\n' %name + '.INST CU\n'+'.MAIL University of Colorado Boulder\n'
        header_str1 += '.EMAIL %s\n' %email_address+'.PHONE\n'+'.FAX\n'+'.MEDIA: Electronic (FTP)\n'
        header_str1 += '.ALTERNATE MEDIA: Electronic (FTP)\n'
        FROM = 'no_reply@surfpy.com'
        TO = iris_email
        title = 'Subject: Requesting Data\n\n'
        ctime = start_date
        while(ctime <= end_date):
            year = ctime.year
            month = ctime.month
            day = ctime.day
            ctime += 86400
            year2 = ctime.year
            month2 = ctime.month
            day2 = ctime.day
            header_str2 = header_str1 +'.LABEL %s_%d.%s.%d\n' %(label, year, mondict[month], day)
            header_str2 += '.QUALITY %s\n' %quality +'.END\n'
            day_str = '%d %d %d 0 0 0 %d %d %d 0 0 0' %(year, month, day, year2, month2, day2)
            out_str = ''
            Nsta = 0
            for network in self.inv:
                for station in network:
                    netcode = network.code
                    stacode = station.code
                    staid = netcode+'.'+stacode
                    st_date = station.start_date
                    ed_date = station.end_date
                    if skipinv and (ctime < st_date or (ctime - 86400) > ed_date):
                        continue
                    # determine channel type (only single channel will be selected)
                    channel_type = None
                    for chantype in chanrank:
                        tmpch = station.select(channel = chantype+'?')
                        if len(tmpch) >= len(channels):
                            channel_type = chantype
                            break
                    if channel_type is None:
                        if verbose:
                            print('!!! NO selected channel types: '+ staid)
                            pdb.set_trace()
                        continue

                    # for HHZ
                    # if channel_type != 'HH':
                    #     pdb.set_trace()

                    Nsta += 1
                    for tmpch in channels:
                        chan = channel_type + tmpch
                        chan_str = '1 %s' %chan
                        sta_str = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                        out_str += sta_str

            out_str = header_str2 + out_str
            
            if Nsta == 0:
                print ('--- [NOISE DATA REQUEST] No data available in inventory, Date: %s' %(ctime - 86400).isoformat().split('T')[0])
                continue
            #========================
            # send email to IRIS
            #========================

            if send_email:
                server = smtplib.SMTP('localhost')
                MSG = title + out_str
                server.sendmail(FROM, TO, MSG)
                server.quit()
            print ('--- [NOISE DATA REQUEST] email sent to IRIS, Date: %s' %(ctime - 86400).isoformat().split('T')[0])
        return

    def req_iris_event_obs(self,  mindist=10., maxdist=120., window=8400., chanrank=['L', 'B', 'H'], obs_channels = ['H1', 'H2', 'HZ', 'DH'],
            start_date=None, end_date=None, label='OBS', quality='B', name = 'ChuanmingLiu',
            send_email=False, email_address='chuanmingliu.cu@gmail.com', iris_email='breq_fast@iris.washington.edu'):
        """
        request Rayleigh wave data from IRIS server
        parameters
        -----
        min/maxdist :: minimum/maximum epicentral distance, in degree
        obs_channels :: Channel code, need four channels [H1,, H2, HZ, DH] to denoise
        window :: request event length (sec), consist with noise segement 
        =====================================================================================================================
        """
        header_str1 = '.NAME %s\n' %name + '.INST CU\n'+'.MAIL University of Colorado Boulder\n'
        header_str1 += '.EMAIL %s\n' %email_address+'.PHONE\n'+'.FAX\n'+'.MEDIA: Electronic (FTP)\n'
        header_str1 += '.ALTERNATE MEDIA: Electronic (FTP)\n'
        FROM = 'no_reply@surfpy.com'
        TO = iris_email
        title = 'Subject: Requesting Data\n\n'

        print(f'Total: {len(self.catalog)} events')

        stime4down = UTCDateTime(start_date)
        etime4down = UTCDateTime(end_date)
            
        print('Start to check out the catalog for email!')
        Nevt = 0
        for event in tqdm(self.catalog):
        
            pmag = event.preferred_magnitude()
            magnitude = pmag.mag
            Mtype = pmag.magnitude_type
            event_descrip = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin = event.preferred_origin()
            otime = porigin.time
            timestr = otime.isoformat()
            evlo = porigin.longitude
            evla = porigin.latitude
            evdp = porigin.depth/1000.

            if otime < stime4down or otime > etime4down:
                print('Out of time range')
                continue
            oyear = otime.year
            omonth = otime.month
            oday = otime.day
            ohour = otime.hour
            omin = otime.minute
            osec = otime.second
            olabel = f'{otime.year}_{otime.month:02d}_{otime.day:02d}'
            header_str2 = header_str1 +'.LABEL %s_%d_%s_%d_%d_%d_%d\n' %(label, oyear, mondict[omonth], oday, ohour, omin, osec)
            header_str2 += '.QUALITY %s\n' %quality +'.END\n'
            out_str = ''
            # loop over stations
            Nsta = 0

            for network in self.inv:
                for station in network:
                    netcode = network.code
                    stacode = station.code
                    staid = netcode+'.'+stacode
                    st_date = station.start_date
                    ed_date = station.end_date
                    if (otime < st_date or otime > ed_date):
                        # print(f'Skip for station time: {olabel}')
                        continue
                    channel_type = None
                    for chantype in chanrank:
                        tmpch = station.select(channel=chantype+'??')
                        if len(tmpch) >= len(obs_channels):
                            channel_type = chantype
                            break
                    if channel_type is None:
                        print(f'--- NO selected channel types: {staid}, SKIP')
                        continue

                    stlo = station.longitude
                    stla = station.latitude
        

                    dist, az, baz = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    dist = dist/1000.
                    dist = k2d(dist)
                    if (dist < mindist) or (dist > maxdist): 
                        print('Skip for dist')
                        continue 

                    # starttime       = otime+dist/vmax
                    # endtime         = otime+dist/vmin

                    starttime = otime 
                    endtime = otime + window

                    # start time stampe
                    year = starttime.year
                    month = starttime.month
                    day = starttime.day
                    hour = starttime.hour
                    minute = starttime.minute
                    second = starttime.second
                    # end time stampe
                    year2 = endtime.year
                    month2 = endtime.month
                    day2 = endtime.day
                    hour2 = endtime.hour
                    minute2 = endtime.minute
                    second2 = endtime.second
                    day_str = '%d %d %d %d %d %d %d %d %d %d %d %d'%(year, month, day, hour, minute, second, year2, month2, day2, hour2, minute2, second2)
                    
                    for tmpch in obs_channels:
                        chan = channel_type + tmpch
                        chan_str = '1 %s' %chan
                        sta_str = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                        out_str += sta_str
                    Nsta    += 1
            out_str = header_str2 + out_str
            
            if Nsta == 0:
                # print(f'--- [EQ DATA REQUEST] No data available in inventory, Event: {otime.isoformat} {event_descrip}')
                continue
            #========================
            # send email to IRIS
            #========================
            Nevt += 1
            print(f'--- [EQ DATA REQUEST] email sent to IRIS, Event: {olabel} dist: {dist:.1f} deg {Mtype}:{magnitude}')
            if send_email:
                server  = smtplib.SMTP('localhost')
                MSG     = title + out_str
                server.sendmail(FROM, TO, MSG)
                server.quit()

        print(f'{Nevt} in total')
        return    

    def req_iris_event(self,  mindist=10., maxdist=120., chanrank=['LH', 'BH', 'HH'], channels ='ENZ',
            start_date=None, end_date=None, vmax=6.0, vmin=1.0,
            label='LCM', quality='B', name = 'ChuanmingLiu',
            send_email=False, email_address='chuanmingliu.cu@gmail.com', iris_email='breq_fast@iris.washington.edu'):
        """
        request Rayleigh wave data from IRIS server
        parameters
        -----
        channel :: Chanel coode, e.g. 'Z', 'ENZ'
        vmin, vmax :: minimum/maximum velocity for surface wave event window (km/s)

        =====================================================================================================================
        """
        header_str1 = '.NAME %s\n' %name + '.INST CU\n'+'.MAIL University of Colorado Boulder\n'
        header_str1 += '.EMAIL %s\n' %email_address+'.PHONE\n'+'.FAX\n'+'.MEDIA: Electronic (FTP)\n'
        header_str1 += '.ALTERNATE MEDIA: Electronic (FTP)\n'
        FROM = 'no_reply@surfpy.com'
        TO = iris_email
        title = 'Subject: Requesting Data\n\n'

        print(f'Total: {len(self.catalog)} events')

        stime4down = UTCDateTime(start_date)
        etime4down = UTCDateTime(end_date)
            
        print('Start to check out the catalog for email!')
        Nevt = 0
        Ndist = 0
        for event in tqdm(self.catalog):
        
            pmag = event.preferred_magnitude()
            magnitude = pmag.mag
            Mtype = pmag.magnitude_type
            event_descrip = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin = event.preferred_origin()
            otime = porigin.time
            timestr = otime.isoformat()
            evlo = porigin.longitude
            evla = porigin.latitude
            # evdp = porigin.depth/1000.

            if otime < stime4down or otime > etime4down:
                print('Out of time range')
                continue
            oyear = otime.year
            omonth = otime.month
            oday = otime.day
            ohour = otime.hour
            omin = otime.minute
            osec = otime.second
            olabel = f'{otime.year}_{otime.month:02d}_{otime.day:02d}'
            header_str2 = header_str1 +'.LABEL %s_%d_%s_%d_%d_%d_%d\n' %(label, oyear, mondict[omonth], oday, ohour, omin, osec)
            header_str2 += '.QUALITY %s\n' %quality +'.END\n'
            out_str = ''
            # loop over stations
            Nsta = 0

            for network in self.inv:
                for station in network:
                    netcode = network.code
                    stacode = station.code
                    staid = netcode+'.'+stacode
                    st_date = station.start_date
                    ed_date = station.end_date
                    if (otime < st_date or otime > ed_date):
                        # print(f'Skip for station time: {olabel}')
                        continue
                    channel_type = None
                    for chantype in chanrank:
                        tmpch = station.select(channel=chantype+'?')
                        if len(tmpch) >= len(channels):
                            channel_type = chantype
                            break
                    if channel_type is None:
                        print(f'--- NO selected channel types: {staid}, SKIP')
                        pdb.set_trace()
                        continue

                    stlo = station.longitude
                    stla = station.latitude
        

                    dist, az, baz = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    dist_km = dist/1000.
                    dist_rad = k2d(dist_km)
                    if (dist_rad < mindist) or (dist_rad > maxdist): 
                        print('Skip for dist')
                        continue 

                    starttime = otime+dist_km/vmax
                    endtime = otime+dist_km/vmin

                    # starttime = otime 
                    # endtime = otime + window

                    # start time stampe
                    year = starttime.year
                    month = starttime.month
                    day = starttime.day
                    hour = starttime.hour
                    minute = starttime.minute
                    second = starttime.second
                    # end time stampe
                    year2 = endtime.year
                    month2 = endtime.month
                    day2 = endtime.day
                    hour2 = endtime.hour
                    minute2 = endtime.minute
                    second2 = endtime.second
                    day_str = '%d %d %d %d %d %d %d %d %d %d %d %d'%(year, month, day, hour, minute, second, year2, month2, day2, hour2, minute2, second2)
                    
                    for tmpch in channels:
                        chan = channel_type + tmpch
                        chan_str = '1 %s' %chan
                        sta_str = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                        out_str += sta_str
                    Nsta    += 1
            out_str = header_str2 + out_str
            
            if Nsta == 0:
                # print(f'--- [EQ DATA REQUEST] No data available in inventory, Event: {otime.isoformat} {event_descrip}')
                continue
            #========================
            # send email to IRIS
            #========================
            Nevt += 1
            print(f'--- [EQ DATA REQUEST] email sent to IRIS, Event: {olabel} dist: {dist_rad:.1f} deg {Mtype}:{magnitude}')
            if send_email:
                server  = smtplib.SMTP('localhost')
                MSG     = title + out_str
                server.sendmail(FROM, TO, MSG)
                server.quit()

        print(f'{Nevt} in total')
        return    


    def seedlabel_py(self, time):
        return 'E{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour, time.minute, time.second)

    def hist_distdistribution(self, clon, clat, fout='long_dist_evt.txt'):
        """
        plot histgram of epicentral distance 
        parameter
        -----
        clon :: central longitude
        clat :: central latitude
        """
        dist_array = []
        levt = []
        fid = open(fout, 'w')
        for event in tqdm(self.catalog):
            porigin = event.preferred_origin()
            otime = porigin.time
            evlo = porigin.longitude
            evla = porigin.latitude
            label = self.seedlabel_py(otime)
            dist, az, baz = obspy.geodetics.gps2dist_azimuth(evla, evlo, clat, clon) # distance is in m
            dist = dist/1000.
            dist_array.append(dist)
            distdeg = k2d(dist)
            tcmin = dist/2. # 1km/s
            if tcmin > 8400.:
                levt.append(label)
                print(f'{label} dist: {dist:.0f} km')
                fid.write(f'{label}\n')
        fid.close()
        print(f'longer dist: {np.max(dist_array):.1f} km')
         
        return 