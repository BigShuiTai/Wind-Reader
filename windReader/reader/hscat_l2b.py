import numpy as np
from datetime import datetime
from windReader.reader.wind_base import WIND_BASE

class HSCAT(WIND_BASE):

    def __init__(self, fname):
        super(HSCAT, self).__init__(fname, engine='h5py')
        if not self.attrs['Instrument_ShorName'].startswith("HSCAT"):
            raise ValueError("Satellite not matched")

    @staticmethod
    def _autodecode(string, encoding="utf-8"):
        return string.decode(encoding) if isinstance(string, bytes) else string

    def _quality_control(self, data, qc_flag):
        bitmask = (1 << 31) - 1
        truncated = qc_flag & bitmask
        allowed_codes = np.array([1 << 14, 1 << 15], dtype=np.int64)
        allowed_mask = int(np.bitwise_or.reduce(allowed_codes))
        keep = (truncated & allowed_mask) == truncated
        return np.ma.array(
            data,
            mask=(~keep),
            dtype=data.dtype,
            fill_value=data.fill_value
        )

    def _calc_wvc_time(self, times):
        out = np.empty(times.shape, dtype=object)
        for idx, t in np.ndenumerate(times):
            time_str = self._autodecode(t).strip()
            try:
                out[*idx] = datetime.strptime(time_str, "%Y%m%dT%H:%M:%S")
            except ValueError:
                out[*idx] = None
        return out

    def _calc_wind_spd(self, spd, slope, intercept):
        # mask invalid values
        spd = np.ma.array(spd, mask=(spd==-32767), fill_value=-32767)
        # 0 slope is invalid. Note: slope can be a scalar or array.
        slope = np.where(slope == 0, 1, slope)
        spd = spd * slope + intercept
        # m*s^-1 to knots
        return spd / 0.514

    def _calc_wind_dir(self, spd, dir, slope, intercept):
        # mask invalid values
        dir = np.ma.array(dir, mask=(dir==-32767), fill_value=-32767)
        # 0 slope is invalid. Note: slope can be a scalar or array.
        slope = np.where(slope == 0, 1, slope)
        dir = dir * slope + intercept
        v = spd * np.sin(np.deg2rad(dir))
        h = spd * np.cos(np.deg2rad(dir))
        return {'v': v, 'h': h}

    def load(self, qc=True):
        self.wvc_time = self._calc_wvc_time(
            self._datasets["wvc_row_time"][:]
        )
        self.wind_spd = self._calc_wind_spd(
            self._datasets["wind_speed_selection"][:],
            self._datasets["wind_speed_selection"].attrs["scale_factor"],
            self._datasets["wind_speed_selection"].attrs["add_offset"]
        )
        self.wind_dir = self._calc_wind_dir(
            self.wind_spd,
            self._datasets["wind_dir_selection"][:],
            self._datasets["wind_dir_selection"].attrs["scale_factor"],
            self._datasets["wind_dir_selection"].attrs["add_offset"]
        )
        self.latitude = self._datasets["wvc_lat"][:]
        self.longitude = self._datasets["wvc_lon"][:]
        if qc:
            # quality control by qc flags
            qc_flag = self._datasets["wvc_quality_flag"][:]
            self.wind_spd = self._quality_control(self.wind_spd, qc_flag)
            self.wind_dir["v"] = self._quality_control(self.wind_dir["v"], qc_flag)
            self.wind_dir["h"] = self._quality_control(self.wind_dir["h"], qc_flag)

    @property
    def attrs(self):
        return {k: self._autodecode(v[-1]) for k, v in self._datasets.attrs.items()}

    @property
    def platform_name(self):
        return self.attrs['Platform_ShortName'] + " " + self.attrs['Instrument_ShorName'] + " Level 2B"

    @property
    def resolution(self):
        return "25.0 KM"

    @property
    def start_time(self):
        time = self.attrs['Range_Beginning_Time']
        return datetime.strptime(time, "%Y%m%dT%H:%M:%S")

    @property
    def end_time(self):
        time = self.attrs['Range_Ending_Time']
        return datetime.strptime(time, "%Y%m%dT%H:%M:%S")
