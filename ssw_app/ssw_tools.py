#!/usr/bin/env python
# coding: utf-8
''' useful functions for Sudden Stratospheric Warmings (SSWs) and Strong Vortex Events (SVEs).'''
''' Scripts produced for the EU/HORIZON-funded MSCA-IF-GF SD4SP project (GA 101065820)'''

import numpy as np
import matplotlib as mpl
from netCDF4 import num2date, date2num, Dataset
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.fftpack as fftp
import scipy.stats as stats
import matplotlib.path as mpath
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def interp25(u10, lat):
    '''
    Arranges latitudes in increasing order and interpolates a variable 
    from 55N to 70N every 2.5 degrees. Preparation for ssw_go.
    -- Input -- two arrays:
        'u10' (array): nt x nlatitudes array containing the variable
                       (e.g. zonal-mean zonal wind)
        'lat' (array): nlatitudes array
    -- Output -- two arrays:
        'dint'   (array): nt x 7 variable interpolated from 55N to 70N every 2.5 degrees
        'lat_lr' (array): low-resolution latitudes from 55N to 70N every 2.5 degrees
    '''
    nt = u10.shape[0]

    # --- Ensure latitudes are in increasing order ---
    if lat[0] > lat[1]:
        u10 = u10[:, ::-1]
        lat_n = np.sort(lat)
    else:
        lat_n = np.copy(lat)

    # --- Restrict to Northern Hemisphere (0N–90N) ---
    l0  = np.abs(lat_n -  0).argmin()
    l90 = np.abs(lat_n - 90).argmin()
    lat_n = lat_n[l0:l90 + 1]
    u10   = u10[:, l0:l90 + 1]

    # --- Interpolate to 2.5-degree grid from 55N to 70N ---
    lat_lr = np.arange(7) * 2.5 + 55
    dint   = np.array([np.interp(lat_lr, lat_n, u10[it, :]) for it in range(nt)])

    return dint, lat_lr

def ssw_go(uu, lat, N, MM):
    '''
    23/08/2023: added MM option to include early March final warmings (before 10 March).
    
    Similar to Palmeiro et al. (2015, 2020). Finds sudden stratospheric warmings (SSW) 
    and final warmings (SFW) in a range of latitudes in increasing order. SSWs are selected 
    as zonal-mean zonal wind reversals at any given latitude. Also works with one latitude. 
    The earliest event is always selected, and must be separated with at least N consecutive 
    days of westerlies. Note that the choice of N is crucial for the selection of the SFW.
    Better use with an interpolation of 2.5 degrees using interp25.
    -- Input --
        'uu'  (array): nt x nlatitudes array containing zonal-mean zonal wind
        'lat' (array): nlatitudes array containing latitudes to be evaluated
        'N'   (int):   number of days between events, 21 is optimal
        'MM'  (int):   1 if early March final warmings are included in SSW count, 0 if not
    -- Output -- four arrays
        'ssw212, sfw212': Positions of the events in time starting 1 Nov (212 days x nyr)
        'ssw365, sfw365': Positions of the events in time starting 1 Jan (365 days x nyr)
    '''
    ny_lr = len(lat)
    nyr   = int(len(uu[:, 0]) / 212)
    uu    = np.reshape(uu, (nyr, 212, ny_lr))

    ssw212 = np.array([])
    ssw365 = np.array([])
    sfw212 = np.array([])
    sfw365 = np.array([])

    for iyr in range(nyr):
        st_sort = np.array([])
        st_all  = np.array([])

        # --- Collect all easterly days across latitudes ---
        for iy in range(ny_lr):
            st = np.where(uu[iyr, :, iy] < 0)[0]
            st_all = np.concatenate((st_all, st), axis=0)

        # --- Filter: keep earliest event per cluster, separated by at least N days ---
        if len(st_all) != 0:
            st_sort = np.sort(st_all)
            diff    = st_sort - np.roll(st_sort, +1) + 1
            res     = np.where(np.abs(diff) >= N)[0]
            st_sort = st_sort[res]

        # --- Identify SFW and separate from SSWs ---
        if MM == 1:
            # Early March final warmings (before 10 Mar) count as SFW, not SSW
            if len(st_sort) >= 1:
                if st_sort[-1] > (30 + 31 + 31 + 28 + 10):  # Nov+Dec+Jan+Feb+10Mar
                    sfw     = st_sort[-1]
                    st_sort = st_sort[:-1]
                else:
                    sfw = st_sort[-1]   # include as final warming anyway
                    # st_sort unchanged (early March SFW stays as SSW in count)
            else:
                sfw     = 212
                st_sort = np.array([])
        else:
            # Keep March SFWs: last event is always the SFW
            if len(st_sort) >= 2:
                sfw     = st_sort[-1]
                st_sort = st_sort[:-1]
            elif len(st_sort) >= 1:      # ← bug fix: era len(st_sort>=1)
                sfw     = st_sort[0]
                st_sort = np.array([])
            else:
                sfw     = 212
                st_sort = np.array([])

        # --- Store events in both timelines ---
        if len(st_sort) != 0:
            st212  = st_sort + (iyr * 212)
            st365  = st_sort + 304 + (iyr * 365)
            ssw212 = np.concatenate((ssw212, st212), axis=0)
            ssw365 = np.concatenate((ssw365, st365), axis=0)

        sfw212 = np.concatenate((sfw212, [sfw + iyr * 212]),           axis=0)
        sfw365 = np.concatenate((sfw365, [sfw + 304 + (iyr * 365)]),   axis=0)

    print(f'A total of {len(ssw365)} SSWs were found in {nyr} years.')
    print(f'{np.round(len(ssw365) / nyr, 2)} SSW/yr  |  '
          f'{np.round(len(ssw365) * 10 / nyr, 2)} SSW/decade')

    return ssw212, ssw365, sfw212, sfw365

def find_ese(dat,th,sep):
    nyr = dat.shape[0]
    ese365=[]
    for iyr in range(nyr):
        #print('processing year: ',iyr+year0) 
        st_sort=np.array([])
        st365=np.array([])
        if th > 0:
            st_all = np.where(dat[iyr,:] > th)[0]
        else:
            st_all = np.where(dat[iyr,:] < th)[0]
        if len(st_all)!= 0: # DEFINE SSWs if THEY EXIST
            st_sort = np.sort(st_all)
        #locate events in the timeline    
        if len(st_sort) != 0:
            st365=st_sort+304+(iyr*365)
            if sep:
                st_filt = filter_events(st365, separation=21)
            else:
                st_filt = np.copy(st365)
            ese365=np.concatenate((ese365,st_filt[0::]),axis=0)


    return ese365
def filter_events(st, separation=21):
    """
    Filter to remove consecutive events separated less than 21 days.
    """
    if len(st) == 0:
        return np.array([])

    # first one first
    filtered_ev = [st[0]]
    
    last_valid = st[0]

    for i in range(1, len(st)):
        if st[i] - last_valid >= separation:
            filtered_ev.append(st[i])
            last_valid = st[i] # update
            
    return np.array(filtered_ev)





def my_smooth(x,win_len):
    nx=len(x)
    y=np.zeros(nx)
    y.fill(np.nan)
    nx2=nx+win_len*2
    temp=np.zeros(nx2)
    temp.fill(np.nan)
    temp[win_len:nx2-win_len]=x
    for ii in range(nx):
        ran=temp[ii:ii+win_len*2+1]
        y[ii]=np.nanmean(ran)
    ran=0
    temp=0
    return y


def ssw_intra(st, year0, nyr, labn, colorn):
    '''
    Plots the intra-seasonal distribution of SSW events for a single case.
    -- Input --
        'st'     (array): SSW event positions in 365-day timeline (1 Jan start)
        'year0'  (int):   first year of the dataset
        'nyr'    (int):   number of years
        'labn'   (str):   legend label
        'colorn' (str):   line color
    -- Output --
        fig, ax1: matplotlib Figure and Axes objects
    '''
    nst = len(st)

    # --- Convert 365-day positions to winter-day index (Nov1=0 ... Mar31~151) ---
    dia   = (st % 365).astype(float)
    intra = np.zeros(151)

    for i in range(nst):
        d = int(dia[i])
        if d >= 304:
            d = d - 304       # Nov 1 → 0, Dec 31 → 57
        else:
            d = d + 61        # Jan 1 → 61, Mar 31 → 150
        if 0 <= d < 151:
            intra[d] += 1

    # --- 21-day centered running sum, normalized to events/decade ---
    intra1 = np.zeros(171)
    intra1[10:161] = intra
    runm = np.zeros(151)
    for i in range(10, 161):
        runm[i - 10] = np.sum(intra1[i - 10:i + 11])
    runm = runm * 10. / nyr

    # --- Smooth and plot ---
    r  = my_smooth(runm, 5)
    ff = np.round(nst * 10. / nyr, 1)

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("SSW per decade", fontsize='x-large')
    ax1.plot(np.arange(151), r,
             label=f'{labn} [{ff}]',
             color=colorn, linewidth=3)

    # --- Formatting ---
    ax1.set_xlim(0, 151)
    ax1.set_ylim(0, 4)
    ax1.set_xticks([0, 31, 61, 92, 121])
    ax1.set_xticklabels(['Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
                        fontsize=14, rotation=0)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.legend()
    ax1.grid(True)

    return fig, ax1

def ssw_intra_comp(ssw, years0, nyrs, labs, colors):
    '''
    Plots the intra-seasonal distribution of SSW events per decade.
    -- Input --
        'ssw'    (list of arrays): SSW event positions in 365-day timeline (1 Jan start)
        'years0' (list of int):    first year for each case
        'nyrs'   (list of int):    number of years for each case
        'labs'   (list of str):    legend labels
        'colors' (list of str):    line colors
    -- Output --
        fig, ax1: matplotlib Figure and Axes objects
    '''
    # --- Cumulative day-of-year boundaries (Jan-based) ---
    ndm  = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    lmes = np.zeros(13)
    for l in range(12):
        lmes[l + 1] = np.sum(ndm[:l + 1])

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("SSW per decade", fontsize='x-large')

    for icase in range(len(ssw)):
        st     = ssw[icase]
        labn   = labs[icase]
        colorn = colors[icase]
        nyr    = nyrs[icase]
        nst    = len(st)

        # --- Convert 365-day positions to winter-day index (Nov1=0 ... Mar31~151) ---
        dia   = (st % 365).astype(float)
        intra = np.zeros(151)

        for i in range(nst):
            d = int(dia[i])
            if d >= 304:
                d = d - 304           # Nov 1 → 0, Dec 31 → 57
            else:
                d = d + 61            # Jan 1 → 61, Mar 31 → 150
            if 0 <= d < 151:
                intra[d] += 1

        # --- 21-day centered running sum, normalized to events/decade ---
        intra1 = np.zeros(171)
        intra1[10:161] = intra
        runm = np.zeros(151)
        for i in range(10, 161):
            runm[i - 10] = np.sum(intra1[i - 10:i + 11])
        runm = runm * 10. / nyr

        # --- Smooth and plot ---
        r    = my_smooth(runm, 5)
        ff   = np.round(nst * 10. / nyr, 1)
        ax1.plot(np.arange(151), r,
                 label=f'{labn} [{ff}]',
                 color=colorn, linewidth=2)

    # --- Formatting ---
    ax1.set_xlim(0, 151)
    ax1.set_ylim(0, 4)
    ax1.set_xticks([0, 31, 61, 92, 121])
    ax1.set_xticklabels(['Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
                        fontsize=14, rotation=0)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.legend()
    ax1.grid(True)

    return fig, ax1

def detect_ese(root, criterion, year0):
    """
    Loads and prepares zonal wind data from a NetCDF dataset to detect SSW or ESE events.
    Automatically detects the u-wind variable name, latitude/level coordinate names,
    handles Pa→hPa conversion, takes the zonal mean if needed, extracts the 10 hPa level,
    and returns event positions by calling ssw_go or find_ese based on the criterion.
    Parameters
    ----------
    root : netCDF4.Dataset
        Open NetCDF dataset containing zonal wind.
    criterion : str
        Detection criterion. One of:
            '_U60'       → SSW at 60N (single latitude)
            '_U65'       → SSW at 65N (single latitude)
            '_U5570'     → SSW at 55–70N, SFW excluded from SSW count
            '_U5570_11M' → SSW at 55–70N, early March final warmings counted as SSW
            '_ese_wpv'       → Extremely weak stratospheric events at 65N (2σ anomaly threshold) - similar to SSWs
            '_ese_spv'       → Extreme strong stratospheric events at 65N (2σ anomaly threshold)
            
    year0 : int
        First calendar year of the dataset (assumes data starts on Jan 1 of year0).
    Returns
    -------
    ssw212 : ndarray or None
        SSW positions in 212-day timeline. None for '_ese'.
    ssw365 : ndarray
        SSW/ESE positions in 365-day timeline (1 Jan start).
    sfw212 : ndarray or None
        SFW positions in 212-day timeline. None for '_ese'.
    sfw365 : ndarray or None
        SFW positions in 365-day timeline. None for '_ese'.
    nyr : int
        Number of complete winter seasons processed.
    perc : str
        Label suffix describing the criterion configuration.
    """

    U_NAMES = ['ua', 'u', 'uwnd', 'uwind', 'u_wind', 'U', 'U010', 'UWND', 'u-wind']
    u_var = next((n for n in U_NAMES if n in root.variables), None)
    if u_var is None:
        raise KeyError(
            f"No u-wind variable found. Tried: {U_NAMES}.\n"
            f"Available variables: {list(root.variables.keys())}"
        )
    print(f"[detect_ese] u-wind variable  : '{u_var}'")

    LAT_NAMES = ['lat', 'latitude', 'LAT', 'LATITUDE', 'nav_lat', 'rlat', 'y']
    lat_var = next((n for n in LAT_NAMES if n in root.variables), None)
    if lat_var is None:
        raise KeyError(
            f"No latitude variable found. Tried: {LAT_NAMES}.\n"
            f"Available variables: {list(root.variables.keys())}"
        )
    lat = root.variables[lat_var][:]
    print(f"[detect_ese] latitude variable : '{lat_var}' "
          f"({lat[0]:.1f}°→{lat[-1]:.1f}°, n={len(lat)})")


    LEV_NAMES = ['level', 'lev', 'plev', 'pressure', 'LEV', 'LEVEL',
                 'plevel', 'isobaricInhPa', 'pres', 'depth']
    lev_var = next((n for n in LEV_NAMES if n in root.variables), None)

    u_dims  = list(root.variables[u_var].dimensions)
    has_lev = (lev_var is not None) and any(
        d in u_dims for d in LEV_NAMES
    )

    print(f"[detect_ese] u dimensions      : {u_dims}")
    print(f"[detect_ese] Has level axis    : {has_lev}  |  "
          )


    iz10 = None
    if has_lev:
        lev = root.variables[lev_var][:].astype(float)
        lev_units = getattr(root.variables[lev_var], 'units', '').strip()
        nz = len(lev) 
        print(nz)
        # If values are in Pa (max > 1200 Pa ~ 12 hPa), convert
        if np.max(np.abs(lev)) > 1200:
            lev = lev / 100.
            print(f"[detect_ese] Level units '{lev_units}' → "
                  f"converted Pa → hPa")
        else:
            print(f"[detect_ese] Level units '{lev_units}' (hPa, no conversion)")

        iz10 = np.argmin(np.abs(lev - 10.))
        print(f"[detect_ese] 10 hPa → index {iz10} "
              f"(nearest: {lev[iz10]:.2f} hPa)")

    u = root.variables[u_var][:].astype(float)
    u = np.squeeze(u)   # remove any size-1 dimensions


    # u is now (time, lev, lat) or (time, lat)
    if nz>1:
        print(u.shape)
        u = u[:, iz10, :]           # → (time, lat)
    else:
        u = np.squeeze(u)
    u = u[304:, :]
    nyr = int(u.shape[0] // 365)

    if nyr == 0:
        raise ValueError(
            f"Not enough data after Nov 1 slicing: "
            f"only {u.shape[0]} days available (need ≥ 365)."
        )
    print(f"[detect_ese] Processing {nyr} winters: "
          f"{year0} – {year0 + nyr - 1}")

    ny = u.shape[1]

    u = u[:nyr * 365, :]
    u = np.reshape(u, (nyr, 365, ny))
    u = u[:, :212, :]                       # (nyr, 212, nlat)
    u = np.reshape(u, (nyr * 212, ny))      # (nyr*212, nlat)
    uu, lat_lr = interp25(u, lat)
    uu = np.reshape(uu, (nyr, 212, 7))      # (nyr, 212, 7)

    #    Build u10: pad days 185–212 with -1 (easterlies)                #
    #   Forces the SFW detection to terminate within the season          #
    u10 = np.full((nyr, 212, 7), -1.)
    u10[:, :212, :] = uu
    u10 = np.reshape(u10, (nyr * 212, 7))   # (nyr*212, 7)

    # lat_lr indices: 0=55N, 1=57.5N, 2=60N, 3=62.5N, 4=65N, 5=67.5N, 6=70N
    perc = ''

    if criterion == '_U60':
        ssw212, ssw365, sfw212, sfw365 = ssw_go(
            u10[:, 2, np.newaxis], np.array([60.]), 21, 0)

    elif criterion == '_U65':
        ssw212, ssw365, sfw212, sfw365 = ssw_go(
            u10[:, 4, np.newaxis], np.array([65.]), 21, 0)

    elif criterion == '_U5570_11M':
        ssw212, ssw365, sfw212, sfw365 = ssw_go(u10, lat_lr, 21, 1)

    elif criterion == '_U5570':
        ssw212, ssw365, sfw212, sfw365 = ssw_go(u10, lat_lr, 21, 0)

    elif 'ese' in criterion :
        # Anomaly at 65N (index 4) relative to the daily climatology
        u10_2d = np.reshape(u10, (nyr, 212, 7))[:, :, 4]   # (nyr, 212)
        clim   = np.nanmean(u10_2d, axis=0, keepdims=True)  # (1, 212)
        au10   = u10_2d - clim                               # (nyr, 212)
        if 'spv' in criterion:
            sign = 1
        else:
            sign = -1
        ths    = np.nanstd(au10, ddof=1)*(sign)

        print(f"[detect_ese] ESE: σ = {ths:.3f} m/s  |  "
              f"threshold (2σ) = {2*ths:.3f} m/s")
        ssw365 = find_ese(au10, ths * 2, sep=True)
        ssw212, sfw212, sfw365 = None, None, None
        if 'spv' in criterion:
            perc   = '2DS_U65_21d_'
        else:
            perc   = '_2DS_U65_21d_'

    else:
        raise ValueError(
            f"Unknown criterion: '{criterion}'. "
            f"Valid options: '_U60', '_U65', '_U5570', '_U5570_11M', '_ese_spv', 'ese_wpv'."
        )

    return ssw212, ssw365, sfw212, sfw365, nyr, perc    

def plt_prop_comp(tit, var, ndays, lev, year0, save=False):
    wind = np.transpose(var)
    time = np.arange(ndays)
    nt = len(wind[0, :])
    
    clev = np.array([-3.  , -2.75, -2.5 , -2.25, -2.  , -1.75, -1.5 , -1.25, -1.  ,
       -0.75, -0.5 , -0.25, 0, 0.25,  0.5 ,  0.75,  1.  ,  1.25,
        1.5 ,  1.75,  2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    cmap = plt.cm.seismic
    
    fig, ax1 = plt.subplots(figsize=(7, 3.5), layout="constrained")
    
    ax1.set_yscale('log')
    ax1.axis([0, nt, 100, 10]) # De 100 a 10 hPa
    
    cf = ax1.contourf(time, lev, wind, levels=clev, cmap=cmap, extend='both')
    
    cs = ax1.contour(time, lev, wind, levels=clev, colors="#45484c", linewidths=0.5)
    #ax1.clabel(cs, fmt='%0.1f', inline=True, fontsize=12)
    ax1.set_title(tit, fontsize=14, pad=20)
    ax1.set_ylabel('Pressure (hPa)', fontsize=12)
    ax1.set_xlabel('days', fontsize=12)
    
    
    plt.yticks([850, 200, 100, 50, 30, 10], ['850', '200', '100', '50', '30', '10'])
    plt.xticks(np.arange(0, 210, 30), ['-90', '-60', '-30', '0', '30', '60', '90'])
    
    ax1.axhline(y=200, linewidth=2, color='navy', linestyle=':')
    cbar = fig.colorbar(cf, ax=ax1, orientation='vertical', pad=0.02, aspect=20)
    cbar.set_label('Polar Cap Z-Index', fontsize=12)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3]) # Solo los valores principales

    if save:
        fname = f'fig_prop_{year0}.png'
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f'Figura guardada como: {fname}')
    
    return fig
def plot_propagation(date_sel,composite=True):
    
    ''''
    Plots the "dripping paint" similar to Baldwin and Dunkerton (Science-2001) but using the pcindex* 
    (*Area-averaged polar geopotential height anomalies, standarized at each pressure level)
    
    Parameters
    ----------
    date_sel : datetime with SSWs dates
            
    year0 : int
        First calendar year of the dataset (assumes data starts on Jan 1 of year0).
    Returns
    -------
    fig : matplotlib.figure.Figure  
    
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'NAM_ERA5_std_new.nc')
    
    root =Dataset('file_path')
    year0=1950
    iyear = year0-1959
    if iyear < 0:
        iyear = 0
    dat     = np.squeeze(root.variables['nam'][(iyear*365)+122::])*(1)#data starts the 1st of sep 1959
    lev     = root.variables['lev'][:]
    nz = len(lev)
    ## NOW let's do a composite
    if len(date_sel) > 1:
        mask = np.array([((f.year > 1960) & (f.year < 2022)) for f in date_sel])
        date_sel = np.array(date_sel)
        mask = np.array(mask).astype(bool)  # Forzamos a que sea booleano

        st1 = date2num(date_sel,units ="days since 1950-01-01",calendar ='noleap') #SSWs arranged as nam data
        st1 = st1[mask]
        date_sel  = num2date(st1,units ="days since 1950-01-01",calendar ='noleap') #SSWs arranged as nam data
        st1 = date2num(date_sel,units ="days since 1960-01-01",calendar ='noleap') #SSWs arranged as nam data
    else:
        yy1 = int(date_sel[1:5])
        if yy1 <1960:
            print('Not available data to plot, please select events after year 1960')
            exit
        else:
            tit = date_sel

    iz10 = np.argmin(abs(lev-10))
    nt = dat.shape[0]
    res = np.where(st1<nt-90)
    st1 =st1[res]
    nst = len(st1)
    comp = np.zeros((nst,180,nz))

    for c in range(nst):
        pos = int(st1[c])
        res = np.where(dat[pos-10:pos+10,iz10] == np.nanmax(dat[pos-10:pos+10,iz10]))[0]
        posn = int(res+pos-10)
        comp[c,:,:] = dat[pos-90:pos+90,:] 

    comp = np.nanmean(comp,axis=0)
    wintplot = comp
    print(wintplot.shape)
    if composite:
        tit = 'Composite'
    else:
        if len(date_sel)>1:       
            tit= date_sel[0].strftime("%d %b %Y")
        else:
            print('Not available data to plot, please select events after year 1960')
    print(tit)
    fig = plt_prop_comp(tit,wintplot,180,lev,year0,save=False)
    return fig
def plt_stereo(tit, var, sig, lat, lon, clevs, colormap, colbar, un):
    """
    Northern Hemisphere polar-stereographic contour map (Cartopy).
    var : plotted as contour lines
    sig : plotted as filled contours (significance or same as var)
    """
    if lat[0] > lat[-1]:   # ensure ascending latitudes
        lat, var, sig = lat[::-1], var[::-1, :], sig[::-1, :]

    var_cyc, lon_cyc = add_cyclic_point(var, coord=lon)
    sig_cyc, _       = add_cyclic_point(sig, coord=lon)

    fig = plt.figure(figsize=(5, 5))
    ax  = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())

    # Circular boundary
    theta  = np.linspace(0, 2*np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.LAND, facecolor='#6d6a69', alpha=0.15)
    ax.coastlines(linewidth=0.7, color='#6d6a69')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    cs = ax.contourf(lon_cyc, lat, sig_cyc, transform=ccrs.PlateCarree(),
                     levels=clevs, cmap=colormap, extend='both')
    ax.contour(lon_cyc, lat, var_cyc,  transform=ccrs.PlateCarree(),
               levels=clevs, colors='k', linewidths=1)

    if colbar == 1:
        cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, shrink=0.7, ticks=clevs)
        cbar.set_label(un, rotation=90)

    plt.title(tit, y=0.99)
    return fig, ax

def montecarlo_lonlat(na, nb, nst, st, n, nd, nyr):
    """
    Random onset dates preserving day-of-year seasonality for surface composites.
    Returns pos : (nst, n) integer array of sampled dates.
    """
    pos = np.zeros((nst, n))
    for c in range(nst):
        dy = st[c] % nd
        iyears = np.arange(nyr)
        if dy + na > 365: iyears = iyears[:-1]
        if dy + nb < 0:   iyears = iyears[1:]
        rand_years = np.random.choice(iyears, size=n, replace=True)
        pos1 = rand_years * nd + dy
        valid = pos1 > 11
        pos[c, valid] = pos1[valid]
    return pos


def composite(data, ndays, nb, na, st, nyr, lag=0, signif=False):
    """
    Lag-composite and Monte Carlo significance for surface fields.
    Parameters
    ----------
    data  : (nt, lat, lon) or (nt,) or (nt, lat)
    nb,na : lag window in days [event+nb : event+na]
    st    : event dates (continuous day numbering)
    lag   : calendar offset (e.g. -304 to align Nov=0)
    signif: compute Monte Carlo mask
    Returns comp_long, comp, matsig
    """
    if data.ndim == 1:
        data = data[:, np.newaxis, np.newaxis]
    elif data.ndim == 2:
        data = data[:, :, np.newaxis]
        
    nt, ndim1, ndim2 = data.shape
    st = st[st + na < nt] # Remove events too close to the end
    nst = len(st)
    ntim = abs(na) - nb
    
    print(f'data shape: {data.shape} | events: {nst} | window: {ntim} days')

    # 2. Extract Composites
    comp_long = np.full((nst, ntim, ndim1, ndim2), np.nan)
    for c in range(nst):
    
        comp_long[c, :, :, :] = data[int(st[c]+lag) + nb : int(st[c]+lag) + na, :, :]
        
    comp = np.nanmean(comp_long, axis=1) # Mean over time window
    compm = np.nanmean(comp, axis=0)     # Mean over all events (ensemble mean)

    matsig = None
    
    # 3. Monte Carlo Significance Test
    if signif:
        n = 500
        # Instead of a huge 5D array that kills memory, I'll store just the final means
        comp_aleas = np.zeros((n, ndim1, ndim2))
        pos = montecarlo_lonlat(na, nb, nst, st, n, ndays, nyr)
        
        for m in range(n):
            # Temporary array for this specific bootstrap iteration
            temp_comp = np.zeros((nst, ndim1, ndim2))
            for c in range(nst):
                p = int(pos[c, m]+lag)
                if p + na <= data.shape[0] and p + nb >= 0:
                    # Getting the time mean for each random event
                    temp_comp[c, :, :] = np.nanmean(data[p + nb : p + na, :, :], axis=0)
            
            # Storing the ensemble mean of this random iteration
            comp_aleas[m, :, :] = np.nanmean(temp_comp, axis=0)

        # Vectorized percentile calculation (no more slow nested loops over lat/lon!)
        pos95 = np.nanpercentile(comp_aleas, 97.5, axis=0)
        pos5 = np.nanpercentile(comp_aleas, 2.5, axis=0)
        
        # Masking: keep the value if it's significant, otherwise set to NaN
        matsig = np.where((compm < pos5) | (compm > pos95), compm, np.nan)

    return comp_long, comp, matsig

def plot_impacts(st1,date_sel,var,composite=True):
    ''''
    Plots SSW impact at the surface as a stereographic map of geopotential height anomalies at 1000 hPa
    
    Parameters
    ----------
    st1 : np.array with the positions of events from 1Jan and multiple of 365. Can use ssw365 directly.
            
    year0 : int
        First calendar year of the dataset (assumes data starts on Jan 1 of year0).
    Returns
    -------
    fig : matplotlib.figure.Figure  
    
    '''    
    root    = Dataset('ssw_app/nl_lr_geopot1000_day_ERA5_1950-2021.nc')
    st1 = date2num(date_sel,units ="days since 1950-01-01",calendar ='noleap') 
    dat     = np.squeeze(root.variables['geopot'][:]) / 9.81
    dat = np.squeeze(dat)
    lat_rea = root.variables['lat'][:]
    lon_rea = root.variables['lon'][:]
    ndays_s = 365
    nyr_s   = dat.shape[0]//ndays_s 
    dat = dat[0:nyr_s*ndays_s,:,:]
    dat     = np.reshape(dat, (nyr_s, ndays_s, len(lat_rea), len(lon_rea)))
    anoms_s = np.zeros_like(dat)
    anoms_s.fill(np.nan)
    for iyr in range(nyr_s):
        for iday in range(ndays_s):
            anoms_s[iyr, iday,:,:] = dat[iyr, iday,:,:] - np.nanmean(dat[:,iday,:,:], axis=0)          
    anoms_s = np.reshape(anoms_s, (nyr_s*ndays_s, len(lat_rea), len(lon_rea)))
    gc.collect()
    print(f"[plot_impacts] anoms calculated,      : {anoms_s.shape}")
    clevs_surf = np.array([-30,-25,-20,-15,-10,-5, 5,10,15,20,25,30])
    if composite:
        st1 = st1
    else:
        st1 = np.array([int(st1)])

    comp_long, comp, matsig = composite(anoms_s, ndays_s, 5, 50, st1, nyr_s,lag=0, signif=True)
    fig = plt_stereo('SSW Z1000  [+5,+50] days', np.nanmean(comp, axis=0), matsig,
            lat_rea, lon_rea, clevs_surf, 'RdBu_r', 1, 'm')
    return fig
