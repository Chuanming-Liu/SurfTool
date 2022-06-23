import numpy as np
from os.path import join, exists
import os 
import pdb 
from obspy.clients.fdsn import Client
import obspy
import pyasdf
from obspy.core import UTCDateTime
import copy


#--- functions
def loadcols(fnm, cols, separator=None, comment='#'):
    """
    load station txt file

    fnm : input file name
    cols : column index of output
    separator : separator in the line
    """
    try: 
        cols = list(cols)
    except TypeError:
        cols = [cols] 
    ncol = len(cols)
    out = [ [] for i in range(ncol) ]
    fid = open(fnm, 'r')
    for line in fid:
        _var = line.rstrip().split(separator)
        if _var[0] == comment:
            continue
        for i in range(ncol):
            out[i].append(_var[cols[i]])
    fid.close()
    return out

def txt2Array(fnm, order=[0, 1, 2, 3]):
    """
    load txt station list into a Array object
    """
    if not exists(fnm):
        print('Missing station list')
    array = Array()
    _net, _sta, _lon, _lat = loadcols(fnm, order)
    for k in range(len(_net)):
        array += Station(net=_net[k], sta=_sta[k], lon=float(_lon[k]), lat=float(_lat[k]))
    return array

# def txt2Inventory(fnm, order=[0, 1, 2, 3]):
#     """
#     load txt station list into a Array object
#     """
#     inv = Inventory()
#     _net, _sta, _lon, _lat = loadcols(fnm, order)
#     for k in range(len(_net)):
#         inv += Station(net=_net[k], sta=_sta[k], lon=float(_lon[k]), lat=float(_lat[k]))
#     return inv

def obstxt2Array(fnm, order=[0, 1, 2, 3, 4]):
    """
    elevation: (m)
    """
    array = Array()
    _net, _sta, _lon, _lat,  _elev = loadcols(fnm, order)
    for k in range(len(_sta)):
        array += Station(net=_net[k], sta=_sta[k], lon=float(_lon[k]), lat=float(_lat[k]), elev=float(_elev[k]) )
    return array 


#--- ---#
class Station(object):
    """
    single station class object
    """
    def __init__(self, lat, lon, net='', sta='', elev=0., start_date=None, end_date=None, **kwargs):

        if lon > 180: lon -=360. 

        self.sta = sta
        self.net = net 
        self.staid = f'{net}.{sta}'
        self.lon = lon
        self.lat = lat
        self.elev = elev
        self.start_date = start_date
        self.end_date = end_date


#--- ---#
class Array(object):
    """
    object of multiple stations
    """

    def __init__(self, stations=None):

        self.stations = []
        if isinstance(stations, Station):
            stations = [stations]
        if stations:
            self.stations.extend(stations)

        self.staidic = {f'{s.net}.{s.sta}': s for s in self.stations}

        self.stadic = {f'{s.sta}': s for s in self.stations}

        self.staidlst = [s.staid for s in self.stations]

        self.stalst = [s.sta for s in self.stations]

        self.sta2net = {f'{s.sta}': s.net for s in self.stations  }
        
        return 
    
    def __add__(self, other):
        if isinstance(other, Station):
            other = Array([other])
        else:
            raise TypeError 
        stations = self.stations + other.stations
        return self.__class__(stations)
    
    def append(self, station):
        if isinstance(station, Station):
            self.stations.append(station)
        else:
            raise TypeError('Class Array only accept Station object')
        
        return self


    def setdic(self):
        """
        """
        self.dic = {}
        for sta in self.stations:
            self.dic[sta.staid] = sta
        return
    
    # @property
    # def stalst(self):
    #     """
    #     getting the station value
    #     """
    #     return self._stalst

    # @stalst.setter
    # def stalst(self):
    #     """
    #     setting the station value
    #     """
    #     self._stalst = {f'{s.net}.{s.sta}': s for s in self.stations}

    def setlst(self):
        self.list = []
        for s in self.stations:
            self.list.append(s.staid)
    

    def loadtxt(self, fnm, order=[0, 1, 2, 3]):
        """
        load txt station list into a Array object
        """
        _net, _sta, _lon, _lat = loadcols(fnm, order)
        for k in range(len(_net)):
            sta_inv = Station(net=_net[k], sta=_sta[k], lon=float(_lon[k]), lat=float(_lat[k]))
            self.append(sta_inv)
        return self

    def loadobs(self, fnm, order=[0, 1, 2, 3, 4]):
        """
        elevation: (m)
        """
        _net, _sta, _lon, _lat,  _elev = loadcols(fnm, order)
        for k in range(len(_sta)):
            sta_inv = Station(net=_net[k], sta=_sta[k], lon=float(_lon[k]), lat=float(_lat[k]), elev=float(_elev[k]))
            self.append(sta_inv)
        return self

    def output_meta(self):
        pass


CLIENT = ['GFZ', 'ICGC', 'INGV', 'IPGP','IRIS', 'ODC', 'ORFEUS', 'RESIF']

class Inventory(Array):
    def __init__(self, stations=None):
        """
        self.waveforms[staid].StationXML
        """
        super().__init__(stations=stations)
        self.inv = obspy.Inventory()
        # self.array = Array()
        self.list = []
        self.stationXML = {}
        self.stalst = []
        return

    def get_station_inventory(self, start_date='2000-01-01', end_date='2020-12-31', onlyobs=False, onlyland=False,
         network_reject=None):
        """
        Get obspy inventory
        https://docs.obspy.org/packages/obspy.core.html
        """
        client = Client('IRIS')
        starttime = obspy.UTCDateTime(start_date) 
        endtime = obspy.UTCDateTime(end_date)   
        inv = obspy.Inventory()     
        for _sta in self.stations: 
            net = _sta.net 
            sta = _sta.sta
            inv += client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime, channel='*', 
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None,
            latitude=None, longitude=None, minradius=None, maxradius=None, level='channel', includerestricted=True)


        if network_reject is not None:
            inv = inv.remove(network = network_reject)

        self.inv = inv
        
        if onlyobs:
            self.inv = self.filter_obs()
        elif onlyland:
            self.inv = self.filter_land(inv)

        self.setlst()
        self.setArray()
        self.setwaveforms()
        print(f"Total: {len(self.list)} stations")

        return self.inv

    def get_stations(self, onlyobs=False, onlyland=False, client_name='IRIS', startdate=None, enddate=None, network=None, station=None, location=None, channel=None,
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, latitude=None, longitude=None, minradius=None, maxradius=None,
            network_reject=None,  includerestricted=False):

        """
        Request station inventory based on Obspy

        Parameters
        ---------
        saveobs :: only keep obs stations
        startdate, enddata  - start/end date for searching (e.g. "2010-10-01")
        network :: network codes 
                  Multiple codes are comma-separated (e.g. "IU,TA").
        station :: SEED station codes.
                  Multiple codes are comma-separated (e.g. "ANMO,PFO").
        location :: SEED location identifiers.
                   Multiple identifiers are comma-separated (e.g. "00,01").
                   As a special case ?--? (two dashes) will be translated to a string of two space
                   characters to match blank location IDs.
        channel :: SEED channel codes.
                  Multiple codes are comma-separated (e.g. "BHZ,HHZ").
        includerestricted :: default is False (e.g. C9)

        For box range:
        minlatitude :: Limit to stations with a latitude larger than the specified minimum.
        maxlatitude :: Limit to stations with a latitude smaller than the specified maximum.
        minlongitude :: Limit to stations with a longitude larger than the specified minimum.
        maxlongitude - Limit to stations with a longitude smaller than the specified maximum.

        For circle range:
        latitude :: Specify the latitude to be used for a radius search.
        longitude :: Specify the longitude to the used for a radius search.
        minradius :: Limit to events within the specified minimum number of degrees from the
                     geographic point defined by the latitude and longitude parameters.
        maxradius :: Limit to events within the specified maximum number of degrees from the
                    geographic point defined by the latitude and longitude parameters.

        Returns
        -------
        inventory with requested staiton information

        Ref: https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html
        """
        
        starttime = UTCDateTime(startdate)
        endtime = UTCDateTime(enddate)

        if client_name == 'IRIS':
            client = Client("IRIS")
            inv = client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=None, startafter=None,
            endbefore=None, endafter=None, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, 
            minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, 
            maxradius=maxradius, level='channel', includerestricted=includerestricted)

        else:
            for base_url in CLIENT:
                inv = obspy.Inventory()
                client = Client(base_url)
                inv += client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=None, startafter=None,
                endbefore=None, endafter=None, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, 
                minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, 
                maxradius=maxradius, level='channel', includerestricted=includerestricted)

        if network_reject is not None:
            inv = inv.remove(network = network_reject)

        
        if onlyobs:
            inv = self.filter_obs(inv)
        elif onlyland:
            inv = self.filter_land(inv)
        else:
            pdb.set_trace('Error!')

        self.inv += inv
        self.setlst()
        self.setArray()
        self.setwaveforms()
        # pdb.set_trace()
        # [i for i, e in enumerate(self.list) if e == 'AK.TRF']
        print(f"Total: {len(self.list)} stations")
        # # # return inv
        # self.add_stationxml(inv)
        # self.update_inv_info()
        return
    
    def setlst(self):
        """
        setting the station value
        """
        self.list = []
        self.stalst = []
        for network in self.inv: 
            for station in network: 
                staid = f'{network.code}.{station.code}'
                # multiple station under same name (few stations)
                if staid in self.list:
                    continue

                self.list.append(staid)
                self.stalst.append(f'{station.code}')

                    
        return 

    def setArray(self):
        """
        Save the station information in a Array class
        """
        # array = Array()
        for network in self.inv:
            for sta in network:
                station = Station(net=network.code, sta=sta.code, lon=sta.longitude, lat=sta.latitude, elev=sta.elevation,
                start_date=sta.start_date, end_date=sta.end_date)
                self.append(station)
        return 

    def filter_obs(self, inv):
        """
        keep only obs station
        """        
        remove = []
        # inv = copy.deepcopy(self.inv)
        for network in inv:
            for station in network:
                if station.elevation > -1.:
                    remove.append(f'{network.code}.{station.code}')
                    print(f"remove: {network.code}.{station.code}")
                    inv = inv.remove(network=network.code, station=station.code)
        return inv

    def filter_land(self, inv):
        """
        keep only land station
        """
        remove = []
        # inv = copy.deepcopy(inv)
        for network in inv:
            for station in network:
                if station.elevation < 0.:
                    remove.append(f'{network.code}.{station.code}')
                    print(f"remove: {network.code}.{station.code}")
                    inv = inv.remove(network=network.code, station=station.code)
        return inv
        
    def get_limits_lonlat(self):
        """
        get the geographical limits of the stations
        lon :: obspy [-180, 180]
        """
        staLst = self.stations
        minlat = staLst[0].lat
        maxlat = staLst[0].lat
        minlon = staLst[0].lon
        maxlon = staLst[0].lon

        for sta in staLst:
            lon = sta.lon
            lat = sta.lat 

            minlat = min(lat, minlat)
            maxlat = max(lat, maxlat)
            minlon = min(lon, minlon)
            maxlon = max(lon, maxlon)
  
        print ('longitude range:', minlon, '~', maxlon, 'latitude range: ', minlat, '~', maxlat)

        return minlat, maxlat, minlon, maxlon
    
    def get_center(self):
        
        """
        Get the center of station box
        """
        minlat, maxlat, minlon, maxlon = self.get_limits_lonlat()
        clon = (minlon+maxlon)/2.
        clat = (minlat+maxlat)/2.

        print (f'Center of box: ({clon:.1f}, {clat:.1f})')
        return clon, clat

    def setwaveforms(self):
        """
        similar to pyasdf.utils.StationAccessor waveforms[staid].StationXML
        """
        self.stationXML = {}
        for network in self.inv:
            for station in network:
                net = obspy.core.inventory.Network(code=network.code, stations=[station], description=network.description)
                inv = obspy.Inventory(networks=[net], source=self.inv.source)
                staid = f'{network.code}.{station.code}'
                if staid in self.stationXML:
                    continue 
                self.stationXML[f'{network.code}.{station.code}'] = inv
        return 