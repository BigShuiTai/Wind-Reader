import numpy as np
from datetime import datetime, timedelta
from windReader.reader.wind_base import WIND_BASE

class ASCAT(WIND_BASE):

    def __init__(self, fname):
        super(ASCAT, self).__init__(fname, engine='netcdf4')
        if not self.attrs['title_short_name'].startswith("ASCAT"):
            raise ValueError("Satellite not matched")

    @staticmethod
    def _autodecode(string, encoding="utf-8"):
        return string.decode(encoding) if isinstance(string, bytes) else string

    def _calc_wvc_time(self, seconds):
        # calculate wvc time from 1990-01-01 00:00 UTC
        t0 = datetime(1990, 1, 1, 0, 0, 0)
        out = np.empty(seconds.shape, dtype=object)
        for idx, s in np.ndenumerate(seconds):
            out[*idx] = t0 + timedelta(seconds=s)
        return out

    def _calc_wind_spd(self, spd):
        # m*s^-1 to knots
        return spd / 0.514

    def _calc_wind_dir(self, spd, dir):
        v = spd * np.sin(np.deg2rad(dir))
        h = spd * np.cos(np.deg2rad(dir))
        return {'v': v, 'h': h}

    def load(self):
        self.wvc_time = self._calc_wvc_time(
            self._datasets.variables["time"][:]
        )
        self.wind_spd = self._calc_wind_spd(
            self._datasets.variables["wind_speed"][:]
        )
        self.wind_dir = self._calc_wind_dir(
            self.wind_spd,
            self._datasets.variables["wind_dir"][:]
        )
        self.latitude = self._datasets.variables["lat"][:]
        self.longitude = self._datasets.variables["lon"][:]

    @property
    def attrs(self):
        return {k: self._autodecode(v) for k, v in self._datasets.__dict__.items()}

    @property
    def platform_name(self):
        return self.attrs['source'] + " Level 2"

    @property
    def resolution(self):
        return self.attrs['pixel_size_on_horizontal'].upper()

    @property
    def start_time(self):
        time = self.attrs['start_date']
        time += " " + self.attrs['start_time']
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    @property
    def end_time(self):
        time = self.attrs['stop_date']
        time += " " + self.attrs['stop_time']
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
