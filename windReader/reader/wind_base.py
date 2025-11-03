"""Base reader for Satellite Wind Data"""

import h5py
import netCDF4
import numpy as np
from functools import lru_cache

class WIND_BASE(object):

    def __init__(self, fname, engine='h5py'):
        if engine == 'h5py':
            self._datasets = h5py.File(fname, "r")
        elif engine == 'netcdf4':
            self._datasets = netCDF4.Dataset(fname, "r")
        else:
            raise ValueError("Engine name not matched.")

        self.wvc_time = None
        self.latitude = None
        self.longitude = None

        self.wind_spd = None
        self.wind_dir = {'v': None, 'h': None}

        self.WIND_DATASETS_ID = None

    @staticmethod
    def _autodecode():
        return NotImplemented

    @lru_cache(maxsize=2)
    def _get_indices(self, georange):
        latmin, latmax, lonmin, lonmax = georange
        barr = (
            (self.latitude >= latmin)
            & (self.latitude <= latmax)
            & (self.longitude >= lonmin)
            & (self.longitude <= lonmax)
        )
        barrind_y, barrind_x = np.where(barr)
        yi, yj = np.amin(barrind_y), np.amax(barrind_y)
        xi, xj = np.amin(barrind_x), np.amax(barrind_x)
        return yi, yj, xi, xj

    def all_available_datasets(self):
        return self.WIND_DATASETS_ID

    def load(self):
        return NotImplemented

    @property
    def attrs(self):
        return NotImplemented

    @property
    def platform_name(self):
        return NotImplemented

    @property
    def resolution(self):
        return NotImplemented

    @property
    def start_time(self):
        return NotImplemented

    @property
    def end_time(self):
        return NotImplemented

    def crop(self, ll_box):
        if self.longitude is None or self.latitude is None or self.wind_spd is None:
            raise ValueError(
                "Longitude or Latitude or data is not empty. "
                "You should run `load` first."
            )
        if not ll_box:
            raise ValueError("crop must be given ll_box value.")
        yi, yj, xi, xj = self._get_indices(ll_box)
        self.wind_spd = self.wind_spd[yi:yj, xi:xj]
        self.wind_dir['v'] = self.wind_dir['v'][yi:yj, xi:xj]
        self.wind_dir['h'] = self.wind_dir['h'][yi:yj, xi:xj]
        self.latitude = self.latitude[yi:yj, xi:xj]
        self.longitude = self.longitude[yi:yj, xi:xj]
        if len(self.wvc_time.shape) == 2:
            self.wvc_time = self.wvc_time[yi:yj, xi:xj]
        else:
            self.wvc_time = self.wvc_time[yi:yj]

    def nearest_time(self, ll_box):
        if not ll_box:
            raise ValueError("nearest_time must be given ll_box value.")
        latmin, latmax, lonmin, lonmax = ll_box
        lon_mean = (lonmin + lonmax) / 2
        lat_mean = (latmin + latmax) / 2
        loc, min_dist = np.nan, 1000
        locs = []
        for (ilon, lon), (_, lat) in zip(
            np.ndenumerate(self.longitude),
            np.ndenumerate(self.latitude)
        ):
            if lon >= lonmin and lon <= lonmax and lat >= latmin and lat <= latmax:
                locs.append((ilon, lon, lat))
        for loc_meta in locs:
            loc_, lon_, lat_ = loc_meta
            lon_diff = abs(lon_ - lon_mean)
            lat_diff = abs(lat_ - lat_mean)
            lonlat_dist = np.sqrt(lon_diff**2 + lat_diff**2)
            if lonlat_dist <= min_dist:
                min_dist = lonlat_dist
                loc = loc_
        try:
            if isinstance(self.wvc_time, (list, tuple, np.ndarray)):
                if len(self.wvc_time.shape) == 1:
                    nearest_datetime = self.wvc_time[loc[0]]
                else:
                    nearest_datetime = self.wvc_time[*loc]
            else:
                nearest_datetime = self.wvc_time[loc]
        except Exception:
            nearest_datetime = None
        return nearest_datetime

    def get_lonlats(self):
        return self.longitude, self.latitude
    
    def get_values(self):
        return self.wind_spd, self.wind_dir
