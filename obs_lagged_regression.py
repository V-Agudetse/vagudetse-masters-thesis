#!/usr/bin/env python3

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
from eofs.standard import Eof
from scipy import signal
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='COBE; ERSST')
    parser.add_argument('season', type=str, help='SON, DJF, MAM, JJA')
    parser.add_argument('year', type=int, help='-1, 0, 1')
    return parser.parse_args()

args = parse_args()
data = args.data
season = args.season
year = args.year

if data=='COBE':
    fname = "COBESST_sst_monthlymean.nc"
    identifier = "obs"
    tag = "obs_cobe"
    missingvalue = 1.e20
elif data=='ERSST':
    fname = "ERSST_sst_monthlymean.nc"
    identifier = "obs"
    tag = "obs_ersst"
    missingvalue = -9.96921e+36
else:
    print("Error: not a valid file option.")
    sys.exit()

if season=='DJF':
    lag = 0
    lag_tag = "D({})JF({})".format(year,year+1)
elif season=='SON':
    lag = -3
    lag_tag = "SON({})".format(year)
elif season=='JJA':
    lag = -6 
    lag_tag = "JJA({})".format(year)
elif season=='MAM':
    lag = -9 
    lag_tag = "MAM({})".format(year)
else:
    print("Error: not a valid season.")
    sys.exit()

filepath = "./data/{}".format(fname)
f = nc.Dataset(filepath,'r')

lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
time = f.variables['time']
dates = nc.num2date(time[:],time.units)
ts = f.variables['sst'][:]
ts = ts.astype(float)
ts = np.ma.masked_values(ts,missingvalue)
ts = signal.detrend(ts, axis=0, type='linear')
ts = np.ma.masked_outside(ts,-100,100)
f.close()

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

tsDJF, dummy = getseasonmean(ts,0)
ts_lagged, yrmax = getseasonmean(ts,lag)

tsDJF, eoflats, eoflons = getareabounds(tsDJF,30,-30,140,290)
ts_lagged, lats, lons = getareabounds(ts_lagged,30,-30,80,290) #30,-30,80,290

#EOF Analysis
coslat = np.cos(np.deg2rad(eoflats))
wgts = np.sqrt(coslat)[...,np.newaxis]
solver = Eof(tsDJF, weights=wgts)
eofs = solver.eofs(neofs=5)
pcs = solver.pcs(npcs=5,pcscaling=1) #DIMENSIONS (t,n)
eigenvalues = solver.eigenvalues()
variances = solver.varianceFraction()
eigen_errors = solver.northTest() #returns errors in eigenvalues according to North, 1982
pc1 = -pcs[:,0]

#Computing lag correlation map
nsamples, nx, ny = np.shape(ts_lagged)
a = ts_lagged.reshape(nsamples,(nx*ny))
if year==0:
    a = a[:dummy]
elif year==-1:
    a = a[:dummy-1]
    pc1 = pc1[1:]
elif year==1:
    a = a[1:]
    if season=='MAM' or season=='JJA':
        pc1 = pc1[:dummy]
    else:
        pc1 = pc1[:-1]
corr = np.corrcoef(a,pc1,rowvar=False)
corr_map = corr[-1,0:-1]
corr_map = corr_map.reshape(nx,ny)

#Statistical significance of correlations
t_map = corr_map*np.sqrt(nsamples-2)/np.sqrt(1-corr_map**2) #Student t-score
p_map = stats.t.sf(np.abs(t_map),nsamples) #P-value
p_map = np.ma.masked_greater(p_map,0.05)
mask = np.ma.getmask(p_map)
corr_map_masked = np.ma.masked_array(corr_map, mask=mask)
std_ts = np.ma.std(ts_lagged,axis=0)
std_pc = np.std(pcs[:,0])
r_map = corr_map*std_ts/(std_pc)
r_map_masked = np.ma.masked_array(r_map, mask=mask)

#map to be plotted
plotmap = r_map
ax = plt.axes(projection=ccrs.PlateCarree(180))
vmax, vmin = [3.0,-3.0] #upper and lower colorbar values
levels = np.linspace(vmin,vmax,13) #colorbar levels
plt.contourf(lons,lats,plotmap,levels,transform=ccrs.PlateCarree(),cmap=plt.cm.coolwarm,extend='both')
ax.coastlines()
plt.colorbar(shrink=0.5) 
plt.title("{}".format(lag_tag),loc='left',fontsize='xx-large')
plt.savefig("./figures/lagged_regressions-{}/{}_lagged_reg_{}.png".format(tag,tag,lag+year*12),bbox_inches='tight')
plt.show()
