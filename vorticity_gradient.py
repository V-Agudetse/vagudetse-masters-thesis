#!/usr/bin/env python3

# *** Computes climatological meridional vorticity gradient *** 
# Uses A.J. Dawson's windspharm package: https://ajdawson.github.io/windspharm/latest/index.html

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pylab import text
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
import argparse
import sys
from scipy import signal
from scipy import stats

from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='ctl, no_mt')
    return parser.parse_args()

args = parse_args()
data = args.data

if data=='ctl':
    fname = "b40.1850.track1.2deg.003.cam2.h0.Z3.000101-100012.nc"
    fname2 = "b40.1850.track1.2deg.003.cam2.h0.V.000101-100012.nc"
    identifier = "control"
    tag = "ctl"
    lli, uli = 6000, 12000 #lower and upper time limit
    level = 12
elif data=='no_mt':
    fname = "pl_PSL_PS_TS_PREC_U_V_Z3_CAM4POP_NoMT_f19.cam.0001-0302.nc"
    fname2 = "pl_PSL_PS_TS_PREC_U_V_Z3_CAM4POP_NoMT_f19.cam.0001-0302.nc"
    identifier = "No M.T."
    tag = "no_mt_long"
    lli, uli = 0, 3600
    level = 8
else:
    print("Error: not a valid file option.")
    sys.exit()

#Function to get area of interest
def getareabounds(var,lat_lb,lat_ub,lon_lb,lon_ub):
    #l == lower bounds, u == upper bounds
    latli = np.argmin(np.abs(lat-lat_lb))
    latui = np.argmin(np.abs(lat-lat_ub))
    lonli = np.argmin(np.abs(lon-lon_lb))
    lonui = np.argmin(np.abs(lon-lon_ub))
    varlats = lat[latli:latui]
    varlons = lon[lonli:lonui]
    areavar = var[:,latli:latui,lonli:lonui]
    return areavar, varlats, varlons

#function to get seasonal mean
def getseasonmean(var,offset):
    var1 = var[11+offset::12]
    var2 = var[12+offset::12]
    var3 = var[13+offset::12]
    yrmax, nlon, nlat = np.shape(var3)
    seasonmean = (var1[:yrmax]+var2[:yrmax]+var3[:yrmax])/3
    return seasonmean, yrmax

filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units,time.calendar)
ts = f.variables['TS'][lli:uli]
f.close()

filepath = "./data/{}".format(fname2)
f = nc.Dataset(filepath,'r')
uwnd = f.variables['U'][lli:uli,level,:,:]
vwnd = f.variables['V'][lli:uli,level,:,:]
f.close()

tsmean = np.mean(ts, axis=0)
ts = signal.detrend(ts, axis=0, type='linear')

uwnd, dummy1 = getseasonmean(uwnd,0)
vwnd, dummy2 = getseasonmean(vwnd,0)

uwnd, uwnd_info = prep_data(uwnd, 'tyx')
vwnd, vwnd_info = prep_data(vwnd, 'tyx')
lats, uwnd, vwnd = order_latdim(lat, uwnd, vwnd)
w = VectorWind(uwnd, vwnd)

eta = w.absolutevorticity() 
etaanom = np.ma.anom(eta,axis=2)
eta = np.mean(eta,axis=2)[:,:,np.newaxis]

etaanomx, etaanomy = w.gradient(etaanom)
etax, etay = w.gradient(eta)

plot_map, plons_c = add_cyclic_point(np.moveaxis(etay,-1,0), coord=lon) #r_map, corr_map_masked
plons_c[-1] = 359.9999 

#Plots
levels = np.arange(-1.2e-10,1.3e-10,0.2e-10)
ax = plt.axes(projection=ccrs.PlateCarree(180)) 
plt.contourf(plons_c,-lat,plot_map.squeeze(),levels,transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend='both')
ax.coastlines()
#ax.set_extent([-179.99999, 179.999999, 20, 90], crs=ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

gl2 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl2.xlabels_top = False
gl2.ylabels_left = True
gl2.xlabels_bottom = True
gl2.ylabels_right = False
#gl.xlines = True
gl2.xlocator = mticker.FixedLocator([-135, -90, -45, 0, 45, 90, 135, 180])
gl2.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl2.xformatter = LONGITUDE_FORMATTER
gl2.yformatter = LATITUDE_FORMATTER
#text(0.05, 0.9,"f)", ha='center', va='center',bbox=dict(facecolor='white', alpha=0.7), transform=ax.transAxes)

plt.colorbar(shrink=0.7) #orientation='horizontal',
plt.title(r"200 hPa meridional vort. gradient (m$^{{-1}}$s$^{{-1}}$) - {}".format(identifier)) 
plt.savefig("./figures/regression-maps-platecarree/SST_PC1+etay_reg_{}long.png".format(tag),bbox_inches='tight')
plt.show()
