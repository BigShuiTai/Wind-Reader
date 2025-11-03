import numpy as np
from datetime import datetime
from windReader.reader.wind_base import WIND_BASE

class CSCAT(WIND_BASE):

    def __init__(self, fname):
        super(CSCAT, self).__init__(fname, engine='netcdf4')
        if not self.platform_name.startswith("CFOSAT"):
            raise ValueError("Satellite not matched")

    @staticmethod
    def _autodecode(string, encoding="utf-8"):
        return string.decode(encoding) if isinstance(string, bytes) else string

    def _calc_wvc_time(self, times):
        out = np.empty(times.shape, dtype=object)
        for idx, t in np.ndenumerate(times):
            time_str = self._autodecode(t).strip()
            try:
                out[*idx] = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                out[*idx] = None
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
            self._datasets.variables["row_time"][:]
        )
        self.wind_spd = self._calc_wind_spd(
            self._datasets.variables["wind_speed_selection"][:],
        )
        self.wind_dir = self._calc_wind_dir(
            self.wind_spd,
            self._datasets.variables["wind_dir_selection"][:],
        )
        self.latitude = self._datasets.variables["wvc_lat"][:]
        self.longitude = self._datasets.variables["wvc_lon"][:]

    @property
    def attrs(self):
        return {k: self._autodecode(v) for k, v in self._datasets.__dict__.items()}

    @property
    def platform_name(self):
        return self.attrs['platform'] + " " + self.attrs['sensor'] + " Level 2B"

    @property
    def resolution(self):
        res = float(self.attrs['geospatial_lon_resolution'])
        return "%.1f" % round(res * 100) + " KM"

    @property
    def start_time(self):
        time = self.attrs['time_coverage_start']
        return datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")

    @property
    def end_time(self):
        time = self.attrs['time_coverage_end']
        return datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
