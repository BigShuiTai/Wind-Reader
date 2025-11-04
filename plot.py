import json

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
plt.rcParams['axes.unicode_minus'] = False

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from windReader.reader import find_reader
from windReader.colormap import colormap as cm

DEFAULT_WIDTH = 5

def calc_figsize(georange):
    latmin, latmax, lonmin, lonmax = georange
    ratio = (latmax - latmin) / (lonmax - lonmin)
    figsize = (DEFAULT_WIDTH, DEFAULT_WIDTH * ratio)
    return figsize


def main(config):
    """read configs"""
    # reader parameters
    reader = config.get("reader", None)
    route = config.get("source", None)
    fname = config.get("filename", None)
    band = config.get("wind_band", None)
    crop_area = config.get("crop_area", False)
    georange = tuple(config.get("georange", (-90, 90, 0, 360)))
    # plot parameters
    proj_name = config.get("projection", "PlateCarree")
    proj_para = config.get("projection_parameters", {"central_longitude": 0})
    lonlat_step = config.get("lon_lat_step", 2)
    # save parameters
    spath = config.get("save_path", None)
    sfname = config.get("save_filename", None)

    """search reader"""
    reader = "auto" if not reader else reader

    load_file = f"{route}/{fname}"
    reader_config = find_reader(load_file, reader=reader)

    if reader_config is None:
        raise ValueError("No reader matched for this file.")

    reader_name = reader_config['name']
    reader_class = reader_config['class']

    """load wind data"""
    reader = reader_class(load_file)

    if reader.WIND_DATASETS_ID:
        reader.load(band)
    else:
        reader.load()

    # add 360 deg for longitude that lower than 0
    reader.longitude[reader.longitude < 0] += 360

    if crop_area:
        reader.crop(georange)

    time = reader.nearest_time(georange)
    if not time:
        print(
            "Cannot find nearest time for given area, "
            "will try to use start time."
        )
        time = reader.start_time

    resolution = reader.resolution # ends with KM

    lons, lats = reader.get_lonlats()
    wind_speed, wind_dir = reader.get_values()
    wind_dir_v, wind_dir_h = wind_dir['v'], wind_dir['h']

    """get max wind"""
    # get max wind in given area
    if len(wind_speed) == 0 or isinstance(wind_speed.max(), np.ma.core.MaskedConstant):
        print("Empty data for given area.")
    else:
        damax = "%.01f" % wind_speed.max()

    """get satellite info"""
    # transfroming resolution string
    sat_title = reader.platform_name + " " + resolution

    """plot data to figure"""
    print("...PLOTING...")

    # set figure-dpi
    dpi = 1200 / DEFAULT_WIDTH
    
    # set figsize
    figsize = calc_figsize(georange)

    # set projection
    proj = getattr(ccrs, proj_name)

    # normal style
    plot_style = {
        'axes_facecolor': '#FFFFFF',
        'colormap': 'wind',
        'barbs_alpha': 1.,
        'coastline_color': 'k',
        'gridlines_color': 'k'
    }

    # for overlaying infrared imagery
    # plot_style = {
    #     'axes_facecolor': '#333333',
    #     'colormap': 'wind_fnmoc',
    #     'barbs_alpha': 0.7,
    #     'coastline_color': 'k',
    #     'gridlines_color': 'w'
    # }

    # set figure and axis
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=proj(**proj_para)))
    ax.patch.set_facecolor(plot_style["axes_facecolor"])

    # let spines invisible
    for spine in ax.spines.values():
        spine.set_visible(False)

    # set extent
    latmin, latmax, lonmin, lonmax = georange
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    # plot brabs
    cmap, vmin, vmax = cm.get_colormap(plot_style["colormap"])
    nh = lats > 0
    bb = ax.barbs(
        lons,
        lats,
        wind_dir_v,
        wind_dir_h,
        wind_speed,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        flip_barb=(~nh),
        pivot='middle',
        length=3.5,
        linewidth=0.25,
        alpha=plot_style["barbs_alpha"],
        transform=ccrs.PlateCarree(),
    )

    # plot colorbar
    cb = plt.colorbar(
        bb,
        ax=ax,
        orientation='vertical',
        pad=0.01,
        aspect=35,
        fraction=0.03,
        extend='both',
    )
    # set color-bar params
    cb.set_ticks(np.arange(0, 70, 5).tolist())
    cb.ax.tick_params(labelsize=4, length=0)
    cb.outline.set_linewidth(0.3)
    cb.set_alpha(1)
    cb.draw_all()

    # add coastlines
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        facecolor="None",
        edgecolor=plot_style["coastline_color"],
        lw=0.5,
    )

    # add gridlines
    xticks = np.arange(-180, 181, lonlat_step)
    yticks = np.arange(-90, 91, lonlat_step)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        linestyle=':',
        color=plot_style["gridlines_color"],
        xlocs=xticks,
        ylocs=yticks,
    )
    gl.rotate_labels = False
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.left_labels = True
    gl.geo_labels = False
    gl.xpadding = 2.5
    gl.ypadding = 2.5
    gl.xlabel_style = {'size': 3.5, 'color': 'k', 'ha': 'center'}
    gl.ylabel_style = {'size': 3.5, 'color': 'k', 'va': 'center'}

    # add title at the left top of figure
    text = f'{sat_title} Wind (barbs) [kt]'
    text += f' (Generated by @Shuitai)\n'
    text += f'Valid Time: {time.strftime("%Y/%m/%d %H%MZ")}'
    ax.set_title(text, loc='left', fontsize=5)

    # add max wind title at the right top of figure
    try:
        text = f'Max. Wind: {damax}kt'
        ax.set_title(text, loc='right', fontsize=4)
    except NameError:
        pass

    # save figure
    fig.savefig(
        f"{spath}/{sfname}",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close("all")


# main codes
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='wind_plotter')
    parser.add_argument('-c','--config_path', default='config.json')
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    main(config)
