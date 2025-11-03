from .ascat_l2 import ASCAT
from .oscat_l2 import OSCAT
from .hscat_l2b import HSCAT
from .cscat_l2b import CSCAT
from .windrad_l2 import WindRAD

_WIND_READERS = {
    "ascat_nc": ASCAT,
    "oscat_nc": OSCAT,
    "hscat_hdf": HSCAT,
    "cscat_nc": CSCAT,
    "windrad_hdf": WindRAD
}

_reader_list = ["ascat_nc", "oscat_nc", "hscat_hdf", "cscat_nc", "windrad_hdf"]

def find_reader(fname, reader=None):
    """Find a correct reader to read the file given"""
    if not reader or reader == "auto":
        _test_readers = _reader_list.copy()
    elif isinstance(reader, str):
        _test_readers = [reader]
    else:
        _test_readers = list(reader)
    for _reader in _test_readers:
        try:
            if _reader not in _reader_list:
                raise ValueError(f"Reader {reader} not found.")
            print(f"Trying reader {_reader} to load...")
            _WIND_READERS[_reader](fname)
        except ValueError:
            continue
        except Exception as e:
            print(f"Exception for reader {_reader}:", e)
            continue
        else:
            return {
                'name': _reader,
                'class': _WIND_READERS[_reader]
            }
    return None
