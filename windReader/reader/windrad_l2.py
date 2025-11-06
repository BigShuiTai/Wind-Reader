import numpy as np
from datetime import datetime, timedelta
from windReader.reader.wind_base import WIND_BASE

class WindRAD(WIND_BASE):

    def __init__(self, fname):
        super(WindRAD, self).__init__(fname, engine='h5py')
        if not self.attrs["Sensor Name"] == "WindRAD":
            raise ValueError("Satellite not matched")
        self.dataset_id = None
        self.dataset_type = None
        self.WIND_DATASETS_ID = ['C_band', 'Dual_band', 'Ku_band', 'Ku_band_10km']
        self.WIND_DATASETS_NAME = ['C Band', 'Dual Band', 'Ku Band', 'Ku Band']

    @staticmethod
    def _autodecode(string, encoding="utf-8"):
        return string.decode(encoding) if isinstance(string, bytes) else string

    def _quality_control(self, data, qc_flag):
        bitmask = (1 << 17) - 1
        truncated = qc_flag & bitmask
        return np.ma.array(
            data,
            mask=truncated,
            dtype=data.dtype,
            fill_value=data.fill_value
        )
        # Bit 2 and Bit 3 may be falsely reported as QC flags,
        # use the code below if you need.
        # allowed_codes = np.array([1 << 2, 1 << 3], dtype=np.int64)
        # allowed_mask = int(np.bitwise_or.reduce(allowed_codes))
        # keep = (truncated & allowed_mask) == truncated
        # return np.ma.array(
        #     data,
        #     mask=(~keep),
        #     dtype=data.dtype,
        #     fill_value=data.fill_value
        # )

    def _calc_wvc_time(self, day_count, day_slope, day_intercept,
                             ms_count, ms_slope, ms_intercept):
        # mask invalid values
        day_count = np.ma.array(day_count, mask=(day_count==65535), fill_value=-32767)
        ms_count = np.ma.array(ms_count, mask=(ms_count==4294967295), fill_value=-32767)
        # 0 slope is invalid. Note: slope can be a scalar or array.
        day_slope = np.where(day_slope == 0, 1, day_slope)
        ms_slope = np.where(ms_slope == 0, 1, ms_slope)
        # force day_slope to 1 due to wrong value in POAD
        day_slope = 1.
        # calculate date from 2000-01-01 12:00:00 UTC
        t0 = datetime(2000, 1, 1, 12, 0, 0)
        dc = np.asarray(day_count, dtype=np.float64)
        ms = np.asarray(ms_count, dtype=np.float64)
        if dc.shape != ms.shape:
            raise ValueError(f"Shape mismatch: {dc.shape=} vs {ms.shape=}")
        d = dc * day_slope + day_intercept
        s = (ms * ms_slope + ms_intercept) * 1e-3
        out = np.empty(day_count.shape, dtype=object)
        for (idx, d_), (_, s_) in zip(np.ndenumerate(d), np.ndenumerate(s)):
            out[*idx] = t0 + timedelta(days=float(d_), seconds=float(s_))
        return out

    def _calc_wind_spd(self, spd, slope, intercept):
        # mask invalid values
        spd = np.ma.array(spd, mask=(spd==32767), fill_value=-32767)
        # 0 slope is invalid. Note: slope can be a scalar or array.
        slope = np.where(slope == 0, 1, slope)
        spd = spd * slope + intercept
        # m*s^-1 to knots
        return spd / 0.514

    def _calc_wind_dir(self, spd, dir, slope, intercept):
        # mask invalid values
        dir = np.ma.array(dir, mask=(dir==32767), fill_value=-32767)
        # 0 slope is invalid. Note: slope can be a scalar or array.
        slope = np.where(slope == 0, 1, slope)
        dir = dir * slope + intercept
        v = spd * np.sin(np.deg2rad(dir))
        h = spd * np.cos(np.deg2rad(dir))
        return {'v': v, 'h': h}

    def load(self, band_id, qc=True):
        if band_id not in self.WIND_DATASETS_ID:
            raise ValueError("Band ID not matched")
        self.dataset_id = band_id
        self.dataset_type = self.attrs["Projection Type"]
        self.wvc_time = self._calc_wvc_time(
            self._datasets[self.dataset_id]["day_count"][:],
            self._datasets[self.dataset_id]["day_count"].attrs["Slope"],
            self._datasets[self.dataset_id]["day_count"].attrs["Intercept"],
            self._datasets[self.dataset_id]["millisecond_count"][:],
            self._datasets[self.dataset_id]["millisecond_count"].attrs["Slope"],
            self._datasets[self.dataset_id]["millisecond_count"].attrs["Intercept"],
        )
        self.wind_spd = self._calc_wind_spd(
            self._datasets[self.dataset_id]["wind_speed_selected"][:],
            self._datasets[self.dataset_id]["wind_speed_selected"].attrs["Slope"],
            self._datasets[self.dataset_id]["wind_speed_selected"].attrs["Intercept"]
        )
        self.wind_dir = self._calc_wind_dir(
            self.wind_spd,
            self._datasets[self.dataset_id]["wind_dir_selected"][:],
            self._datasets[self.dataset_id]["wind_dir_selected"].attrs["Slope"],
            self._datasets[self.dataset_id]["wind_dir_selected"].attrs["Intercept"],
        )
        if self.dataset_type == "GLL":
            # WindRAD daily data (POAD)
            self.latitude = self._datasets[self.dataset_id]["grid_lat"][:]
            self.longitude = self._datasets[self.dataset_id]["grid_lon"][:]
        else:
            self.latitude = self._datasets[self.dataset_id]["wvc_lat"][:]
            self.longitude = self._datasets[self.dataset_id]["wvc_lon"][:]
        if qc and self.dataset_type != "GLL":
            # quality control by qc flags
            qc_flag = self._datasets[self.dataset_id]["wvc_quality_flag"][:]
            self.wind_spd = self._quality_control(self.wind_spd, qc_flag)
            self.wind_dir["v"] = self._quality_control(self.wind_dir["v"], qc_flag)
            self.wind_dir["h"] = self._quality_control(self.wind_dir["h"], qc_flag)

    @property
    def attrs(self):
        return {k: self._autodecode(v) for k, v in self._datasets.attrs.items()}

    @property
    def platform_name(self):
        platform = self.attrs['Satellite Name'] + " " + self.attrs['Sensor Name'] + " Level 2"
        if self.dataset_id:
            _dataset_name = self.WIND_DATASETS_NAME[
                self.WIND_DATASETS_ID.index(self.dataset_id)
            ]
            return platform + " " + _dataset_name
        else:
            return platform

    @property
    def resolution(self):
        if self.dataset_type and self.dataset_type == "GLL":
            return "25.0 KM (Daily)"
        data_shape = self._datasets[self.dataset_id]["day_count"].shape
        return "10.0 KM" if data_shape[0] == 2201 else "20.0 KM"

    @property
    def start_time(self):
        time = self.attrs['Observing Beginning Date'] + " " + self.attrs['Observing Beginning Time']
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")

    @property
    def end_time(self):
        time = self.attrs['Observing Ending Date'] + " " + self.attrs['Observing Ending Time']
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
