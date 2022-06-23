import numpy as np
import obspy 
from tqdm import tqdm
import pdb
from obspy.clients.fdsn.client import Client
import obspy.clients.iris
from obspy.core import UTCDateTime
from os.path import join, exists




class Eq(object):

    def __init__(self):
        self.catalog = obspy.core.event.Catalog()
        return 

    def get_events(self, startdate, enddate, gcmt=False, Mmin=5.5, Mmax=None,
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, latitude=None, longitude=None,
            minradius=None, maxradius=None, mindepth=None, maxdepth=None, magnitudetype=None):
        """
        Get earthquake catalog from IRIS server
        parameters
        -----
        startdate, enddate :: start/end date for searching
        Mmin, Mmax :: minimum/maximum magnitude for searching                
        minlatitude :: Limit to events with a latitude larger than the specified minimum.
        maxlatitude :: Limit to events with a latitude smaller than the specified maximum.
        minlongitude :: Limit to events with a longitude larger than the specified minimum.
        maxlongitude :: Limit to events with a longitude smaller than the specified maximum.
        latitude :: Specify the latitude to be used for a radius search.
        longitude :: Specify the longitude to the used for a radius search.
        minradius :: Limit to events within the specified minimum number of degrees from the
                    geographic point defined by the latitude and longitude parameters.
        maxradius :: Limit to events within the specified maximum number of degrees from the
                    geographic point defined by the latitude and longitude parameters.
        mindepth :: Limit to events with depth, in kilometers, larger than the specified minimum.
        maxdepth :: Limit to events with depth, in kilometers, smaller than the specified maximum.
        magnitudetype :: Specify a magnitude type to use for testing the minimum and maximum limits.
        =======================================================================================================
        """
        starttime = UTCDateTime(startdate)
        endtime = UTCDateTime(enddate)
        print('=== Start searching of catalog')
        if not gcmt:
            client = Client('IRIS')
            try:
                catISC = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='ISC',
                minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                maxdepth=maxdepth, magnitudetype=magnitudetype)
                endtimeISC = catISC[0].origins[0].time
            except:
                catISC = obspy.core.event.Catalog()
                endtimeISC = starttime

            if endtime.julday-endtimeISC.julday >1:
                try:
                    catPDE = client.get_events(starttime=endtimeISC, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='NEIC PDE',
                    minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                    latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                    maxdepth=maxdepth, magnitudetype=magnitudetype)
                    catalog = catISC+catPDE
                except:
                    catalog = catISC
            else:
                catalog = catISC
            outcatalog = obspy.core.event.Catalog()
            # check magnitude
            for event in catalog:
                if event.magnitudes[0].mag < Mmin:
                    continue
                outcatalog.append(event)
        else:
            # Use of GCMT catalog
            # Updated the URL on Jul 25th, 2020
            gcmt_url_old = 'http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec17.ndk'
            gcmt_new = 'http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY'
            if starttime.year < 2005:
                print('--- Loading catalog: '+gcmt_url_old)
                cat_old = obspy.read_events(gcmt_url_old)
                if Mmax != None:
                    cat_old = cat_old.filter("magnitude <= %g" %Mmax)
                if maxlongitude != None:
                    cat_old = cat_old.filter("longitude <= %g" %maxlongitude)
                if minlongitude != None:
                    cat_old = cat_old.filter("longitude >= %g" %minlongitude)
                if maxlatitude != None:
                    cat_old = cat_old.filter("latitude <= %g" %maxlatitude)
                if minlatitude != None:
                    cat_old = cat_old.filter("latitude >= %g" %minlatitude)
                if maxdepth != None:
                    cat_old = cat_old.filter("depth <= %g" %(maxdepth*1000.))
                if mindepth != None:
                    cat_old = cat_old.filter("depth >= %g" %(mindepth*1000.))
                temp_stime  = obspy.core.utcdatetime.UTCDateTime('2018-01-01')
                outcatalog  = cat_old.filter("magnitude >= %g" %Mmin, "time >= %s" %str(starttime), "time <= %s" %str(endtime) )
            else:
                outcatalog  = obspy.core.event.Catalog()
                temp_stime  = copy.deepcopy(starttime)
                temp_stime.day  = 1
            while (temp_stime < endtime):
                year = temp_stime.year
                month = temp_stime.month
                yearstr = str(int(year))[2:]
                monstr = monthdict[month]
                monstr = monstr.lower()
                if year==2005 and month==6:
                    monstr = 'june'
                if year==2005 and month==7:
                    monstr = 'july'
                if year==2005 and month==9:
                    monstr = 'sept'
                gcmt_url_new = gcmt_new+'/'+str(int(year))+'/'+monstr+yearstr+'.ndk'
                try:
                    cat_new = obspy.read_events(gcmt_url_new, format='ndk')
                    print('--- Loading catalog: '+gcmt_url_new)
                except:
                    print('--- Link not found: '+gcmt_url_new)
                    break
                cat_new = cat_new.filter("magnitude >= %g" %Mmin, "time >= %s" %str(starttime), "time <= %s" %str(endtime) )
                if Mmax != None:
                    cat_new = cat_new.filter("magnitude <= %g" %Mmax)
                if maxlongitude != None:
                    cat_new = cat_new.filter("longitude <= %g" %maxlongitude)
                if minlongitude!=None:
                    cat_new = cat_new.filter("longitude >= %g" %minlongitude)
                if maxlatitude!=None:
                    cat_new = cat_new.filter("latitude <= %g" %maxlatitude)
                if minlatitude!=None:
                    cat_new = cat_new.filter("latitude >= %g" %minlatitude)
                if maxdepth != None:
                    cat_new = cat_new.filter("depth <= %g" %(maxdepth*1000.))
                if mindepth != None:
                    cat_new = cat_new.filter("depth >= %g" %(mindepth*1000.))
                outcatalog  += cat_new
                try:
                    temp_stime.month +=1
                except:
                    temp_stime.year +=1
                    temp_stime.month = 1
        
        print('=== End searching of catalog')
        try:
            self.catalog += outcatalog
        except:
            self.catalog = outcatalog
        return

    def write_catalog(self, fnm='catalog.xml', format='QUAKEML'):
        """
        Saves catalog into a file
        """
        try:
            self.catalog.write(fnm, format=format)
        except:
            raise AttributeError('No self.catalog')
    
    def read_catalog(self, fnm):
        self.catalog += obspy.core.event.read_events(fnm, format='QUAKEML')
        return 

    def get_event_meta(self, outdir='.', outfnm='src.meta'):
        """
        output event.pickle for eikonal
        pd.DataFrame:
        orid lon lat
        """
        Nevent              = len(self.catalog)
        orid_lst            = []
        outlst              = np.array([])
        ievent              = 0
        for event in tqdm(self.catalog):
            ievent          += 1
            Ndata           = 0
            outstr          = ''
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            timestr         = otime.isoformat()
            evlo            = porigin.longitude
            evla            = porigin.latitude
            try:
                evdp        = porigin.depth/1000.
            except:
                continue
            event_id        = event.resource_id.id.split('=')[-1]
            oyear           = otime.year
            omonth          = otime.month
            oday            = otime.day
            ohour           = otime.hour
            omin            = otime.minute
            osec            = otime.second
            
            label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
            event_tag       = 'surf_'+label

            evid_sta        = 'E{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(oyear, omonth, oday, ohour, omin, osec)  

            # if not event_tag in taglst: continue
                
            if magnitude is None: continue

            outlst          = np.append(outlst, 'EQ')
            outlst          = np.append(outlst, evid_sta)
            outlst          = np.append(outlst, '{:15.6f}'.format(evlo))
            outlst          = np.append(outlst, '{:15.6f}'.format(evla))

        outlst              = outlst.reshape((-1, 4))
        # output
        np.savetxt(join(outdir, 'src.meta'), outlst, fmt='%s')
        return     