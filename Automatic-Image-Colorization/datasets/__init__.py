from .rgb_data_loader import RGBDataLoad
from .rgb_data_loader_L import RGB2LDataLoad

__datasets__ = {
    "dataload": RGBDataLoad,
    "dataload_L": RGB2LDataLoad
}
