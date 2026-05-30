#!/usr/bin/env python
# coding: utf-8
#Some useful tools for skill assessment
#This code is part of the EU MSCA postdoctoral fellowships project SD4SP 
# (Stratospheric Dynamics for Seasonal Prediction), GA: 101065820 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats as spstats
import gc
import matplotlib as mpl
def bilinear_interp(data, src_lat, src_lon, dst_lat, dst_lon):
    """
    Bilinear interpolation from src → dst grid.
    data    : [..., nlat_src, nlon_src]
    Returns : [..., nlat_dst, nlon_dst]
    Assumes src_lat and src_lon are monotonically increasing.
    Preserves physical gradients; no smoothing applied.
    """
    nlat_d, nlon_d = len(dst_lat), len(dst_lon)
    shape_out = data.shape[:-2] + (nlat_d, nlon_d)
    out = np.full(shape_out, np.nan, dtype=data.dtype)

    for i, lat in enumerate(dst_lat):
        j0 = int(np.searchsorted(src_lat, lat)) - 1
        j0 = np.clip(j0, 0, len(src_lat) - 2)
        j1 = j0 + 1
        dlat = (lat - src_lat[j0]) / (src_lat[j1] - src_lat[j0] + 1e-12)

        for j, lon in enumerate(dst_lon):
            k0 = int(np.searchsorted(src_lon, lon)) - 1
            k0 = np.clip(k0, 0, len(src_lon) - 2)
            k1 = k0 + 1
            dlon = (lon - src_lon[k0]) / (src_lon[k1] - src_lon[k0] + 1e-12)

            out[..., i, j] = (
                (1-dlat)*(1-dlon) * data[..., j0, k0] +
                   dlat *(1-dlon) * data[..., j1, k0] +
                (1-dlat)*   dlon  * data[..., j0, k1] +
                   dlat *   dlon  * data[..., j1, k1]
            )
    return out

def era5_flat_index(years, era5_start_year, lead):
    """
    ERA5 flat time index for a given forecast lead.
    ERA5 time axis: Jan[era5_start_year], Feb, …, Dec (month 0 = Jan)
    
    Lead 0 = Nov(year)  → ERA5 month index 10 of year
    Lead 1 = Dec(year)  → ERA5 month index 11 of year
    Lead 2 = Jan(year+1)→ ERA5 month index  0 of year+1
    …
    Lead 5 = Apr(year+1)→ ERA5 month index  3 of year+1
    """
    lead_to_cal = {0: (0, 11), 1: (0, 12),   # (year_offset, 1-based month)
                   2: (1, 1),  3: (1, 2), 4: (1, 3), 5: (1, 4)}
    yr_off, mon1 = lead_to_cal[lead]
    cal_years = np.array(years) + yr_off
    return (cal_years - era5_start_year) * 12 + (mon1 - 1)


def model_flat_index(years, model_start_year, lead):
    """
    Model flat time index. Model time axis starts in Nov of model_start_year.
    lead 0 = Nov (offset 0), lead 1 = Dec (offset 1), …, lead 5 = Apr (offset 5)
    """
    year_offsets = np.array(years) - model_start_year
    return year_offsets * 12 + lead


def nearest_grid_point(lat_arr, lon_arr, lat_pt, lon_pt):
    """Return (i_lat, i_lon) of nearest grid point."""
    i = np.argmin(np.abs(lat_arr - lat_pt))
    j = np.argmin(np.abs(lon_arr - lon_pt))
    return int(i), int(j)


# ──────────────────────────────────────────────────────────────────────────────
# SKILL METRICS  (all expect axis 0 = years)
# ──────────────────────────────────────────────────────────────────────────────
def acc(fcst_anom, obs_anom):
    """Anomaly Correlation Coefficient [nlat, nlon] over year axis."""
    num = np.nansum(fcst_anom * obs_anom, axis=0)
    den = np.sqrt(np.nansum(fcst_anom**2, axis=0) * np.nansum(obs_anom**2, axis=0))
    return np.where(den == 0, np.nan, num / den)
# ── Significance tests ────────────────────────────────────────────────────────
def acc_pvalue(r_map, n):
    """Two-tailed p-value for ACC via t-transform. df = n-2."""
    r = np.clip(r_map, -1 + 1e-9, 1 - 1e-9)
    t = r * np.sqrt((n - 2) / (1.0 - r**2))
    return 2.0 * spstats.t.sf(np.abs(t), df=n - 2)

def bias_pvalue(diff_series):
    """Two-tailed p-value for mean bias via paired t-test. diff = fcst - obs [nyr, nlat, nlon]."""
    n    = diff_series.shape[0]
    mean = np.nanmean(diff_series, axis=0)
    std  = np.nanstd(diff_series, axis=0, ddof=1)
    t    = mean / (std / np.sqrt(n) + 1e-12)
    return 2.0 * spstats.t.sf(np.abs(t), df=n - 1)

def bss_pvalue(bss_map, n):
    """One-tailed p-value for BSS > 0. SE ~ sqrt(2/n)."""
    se = np.sqrt(2.0 / n)
    t  = bss_map / (se + 1e-12)
    return spstats.t.sf(t, df=n - 1)

def sig_mask(pval_map, level=SIG_LEVEL):
    """Boolean mask: True where significant."""
    return pval_map < level


def rmse(fcst_anom, obs_anom):
    """Root Mean Squared Error [nlat, nlon]."""
    return np.sqrt(np.nanmean((fcst_anom - obs_anom)**2, axis=0))


def msss(fcst_anom, obs_anom):
    """
    Mean Square Skill Score = 1 - MSE_fcst / Var_obs
    Positive → better than climatology.
    """
    mse_f = np.nanmean((fcst_anom - obs_anom)**2, axis=0)
    var_o = np.nanvar(obs_anom, axis=0, ddof=1)
    return 1.0 - np.where(var_o == 0, np.nan, mse_f / var_o)


def bias(fcst_abs, obs_abs):
    """Mean bias [nlat, nlon] (no anomalies needed)."""
    return np.nanmean(fcst_abs - obs_abs, axis=0)


def bss_tercile(fcst_ens, obs_abs, fcst_emean=None, fcst_spread=None):
    """
    BSS para terciles below/above normal.
    - members mode : fcst_ens [nyr, nmem, nlat, nlon]
    - emean mode   : fcst_ens=None, fcst_emean + fcst_spread [nyr, nlat, nlon]
                     asume distribución Gaussiana N(mu, sigma)
    """
    from scipy.special import ndtr as _Phi

    t33 = np.nanpercentile(obs_abs, 100/3, axis=0)
    t67 = np.nanpercentile(obs_abs, 200/3, axis=0)

    obs_b = (obs_abs < t33).astype(np.float32)
    obs_a = (obs_abs > t67).astype(np.float32)

    if fcst_ens is not None:
        p_b = np.nanmean(fcst_ens < t33[np.newaxis, np.newaxis], axis=1)
        p_a = np.nanmean(fcst_ens > t67[np.newaxis, np.newaxis], axis=1)
    else:
        sigma = np.where(fcst_spread > 1e-6, fcst_spread, 1e-6)
        p_b = _Phi((t33[np.newaxis] - fcst_emean) / sigma)
        p_a = 1.0 - _Phi((t67[np.newaxis] - fcst_emean) / sigma)

    n    = obs_abs.shape[0]
    bs_b = np.nanmean((p_b - obs_b)**2, axis=0)
    bs_a = np.nanmean((p_a - obs_a)**2, axis=0)
    ref  = (1/3) * (2/3)

    bss_b = 1.0 - bs_b / ref
    bss_a = 1.0 - bs_a / ref

    return {
        "bss_below": bss_b,
        "bss_above": bss_a,
        "p_below"  : bss_pvalue(bss_b, n),
        "p_above"  : bss_pvalue(bss_a, n),
    }

# ──────────────────────────────────────────────────────────────────────────────
# SPATIAL UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def make_region_mask(lat, lon, lat0, lat1, lon0, lon1):
    """Boolean mask for a region [nlat, nlon]."""
    lat_mask = (lat >= lat0) & (lat <= lat1)
    lon_mask = (lon >= lon0) & (lon <= lon1)
    return lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]


def area_mean(data, lat, mask=None):
    """
    Cosine-weighted spatial mean.
    data : [nlat, nlon]  →  scalar
    mask : [nlat, nlon] booleana, opcional. None = dominio completo.
    """
    w    = np.cos(np.deg2rad(lat))
    w_2d = np.broadcast_to(w[:, np.newaxis], data.shape)

    if mask is not None:
        data = np.where(mask, data, np.nan)

    return np.nansum(data * w_2d) / np.nansum(w_2d * np.isfinite(data))

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

#for all globe:

PROJ = ccrs.LambertConformal(central_longitude=-30) 
LAND = cfeature.NaturalEarthFeature("physical", "land", "50m",
                                    edgecolor="0.4", facecolor="none", linewidth=0.5)

# for NH:
PROJ_NH = ccrs.Orthographic(central_longitude=0.0, central_latitude=70)  # centrada en el Atlántico Norte
LAND = cfeature.NaturalEarthFeature("physical", "land", "50m",
                                    edgecolor="0.4", facecolor="none", linewidth=0.5)

import matplotlib.path as mpath

def decorate_ax_nh(ax):
    """Specific for NorthPolarStereo."""
    ax.add_feature(LAND)
    ax.coastlines(resolution="50m", linewidth=0.6, color="0.3")
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())

    # Recorte circular
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.4,
                      color="gray", alpha=0.6, linestyle="--")
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(20, 91, 20))


def make_map_axes(nrows, ncols, figsize=None, height_per_row=1.5, title=None, proj=None):
    if proj is None:
        proj = PROJ
    if figsize is None:
        figsize = (16, height_per_row * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             subplot_kw={"projection": proj},
                             constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig, np.atleast_2d(axes)


def decorate_ax(ax, proj, lat=None, lon=None):
    ax.add_feature(LAND)
    ax.coastlines(resolution="50m", linewidth=0.1, color="0.3")
    if isinstance(proj, ccrs.NorthPolarStereo):
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
        
from cartopy.util import add_cyclic_point

def skill_map(ax, data, lat, lon, cmap, vmin, vmax, title="", sig=None, proj=None):
    if proj is None:
        proj = PROJ
    data_c, lon_c = add_cyclic_point(data, coord=lon)
    Lon, Lat = np.meshgrid(lon_c, lat)

    im = ax.pcolormesh(Lon, Lat, data_c, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading='auto', rasterized=True)

    if isinstance(proj, ccrs.NorthPolarStereo):
        decorate_ax_nh(ax)
    else:
        decorate_ax(ax, proj)

    if sig is not None:
        sig_c, _ = add_cyclic_point(sig.astype(float), coord=lon)
        ax.contourf(Lon, Lat, (~sig_c.astype(bool)).astype(float),
                    levels=[0.5, 1.5], hatches=["...."], colors="none",
                    transform=ccrs.PlateCarree())

    ax.set_title(title)
    return im


def add_colorbar(fig, im, ax, label="", orientation="horizontal", fraction=0.04):
    cb = fig.colorbar(im, ax=ax, orientation=orientation,
                      fraction=fraction, pad=0.04)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def skill_map_nae(ax, data, lat, lon, cmap, vmin, vmax, title="", sig=None, proj=None):
    if proj is None:
        proj = PROJ
    data_c, lon_c = add_cyclic_point(data, coord=lon)
    Lon, Lat = np.meshgrid(lon_c, lat)

    im = ax.pcolormesh(Lon, Lat, data_c, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading='auto', rasterized=True)
   # Plot boundaries:
    xmin = -90
    xmax = 30
    ymin = 20
    ymax = 75
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())  
    vertices = [(lon, ymin) for lon in range(xmin, xmax+1, 1)] + \
               [(lon, ymax) for lon in range(xmax, xmin-1, -1)]
    boundary = mpath.Path(vertices)
    ax.set_boundary(boundary, transform=ccrs.PlateCarree())
    decorate_ax(ax, proj)
    # Draw parallel and meridians
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='#515151', alpha=0.7, linestyle=(0, (1, 1)))
    gl.ylocator = mpl.ticker.FixedLocator([0, 40, 60,80])
    gl.xlocator = mpl.ticker.FixedLocator([-60, -30, 0,30,60,80])
    fs = 10
    # Manual labelling of parallels and meridians (dirty trick, it is not supported by LambertConformal)
    ax.text(-91.5, 40,"40$^{\\circ}$N", transform=ccrs.PlateCarree(), fontsize=fs,color='#515151', ha='right',va="center",rotation=0)
    ax.text(-91.5, 60,"60$^{\\circ}$N", transform=ccrs.PlateCarree(), fontsize=fs,color='#515151', ha='right',va="center",rotation=0)
    ax.text(-30, 18,"30$^{\\circ}$W", transform=ccrs.PlateCarree(), fontsize=fs,color='#515151', ha='center',va="center",rotation=0)
    ax.text(-60, 18,"60$^{\\circ}$W", transform=ccrs.PlateCarree(), fontsize=fs,color='#515151', ha='center',va="center", rotation=-20)

    #ax.text(-90, 15,"30$^{\\circ}$E", transform=ccrs.PlateCarree(), fontsize=fs,color='#515151', ha='center',va="center", rotation=10)
    ax.text(0, 18,"0$^{\\circ}$", transform=ccrs.PlateCarree(), fontsize=fs,color='#515151', ha='center',va="center", rotation=20)

    if sig is not None:
        sig_c, _ = add_cyclic_point(sig.astype(float), coord=lon)
        ax.contourf(Lon, Lat, (~sig_c.astype(bool)).astype(float),
                    levels=[0.5, 1.5], hatches=["...."], colors="none",
                    transform=ccrs.PlateCarree())

    ax.set_title(title)
    return im


def skill_maps_regional(results, pvals, var, metric, lat, lon,
                        cmap, vmin, vmax, domain=DOMAINS["euro_atlantic"],
                        label="", sig_key=None):
    """
    Mapas de skill recortados a un dominio regional.
    sig_key : clave en pvals para el stippling, e.g. "acc". None = sin stippling.
    """
    lon0, lon1, lat0, lat1 = domain

    fig, axes = make_map_axes(
        len(EXPERIMENTS), N_LEADS,
        figsize=(16, 3.2 * len(EXPERIMENTS)),
        title=f"{metric.upper()} — {var.upper()} | NDJFMA | Euro-Atlantic"
              + (f"  (dots = not significant p<{SIG_LEVEL})" if sig_key else "")
    )

    for r, exp in enumerate(EXPERIMENTS):
        for c, lead in enumerate(LEAD_MONTHS):
            ax  = axes[r, c]
            sig = sig_mask(pvals[exp][var][sig_key][lead]) if sig_key else None
            im  = skill_map(ax, results[exp][var][metric][lead],
                            lat, lon, cmap, vmin, vmax,
                            title=f"{exp} | {LEAD_LABELS[c]}", sig=sig, proj = PROJ)
            ax.set_extent([lon0, lon1, lat0, lat1], crs=proj)
            ax.set_aspect('auto')   # ← esto
            if c == 0:
                ax.set_ylabel(exp, fontsize=9)

    add_colorbar(fig, im, axes, label=label)
    fname = f"{OUTDIR}/{metric}_{var}_euro_atlantic.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    gc.collect()
    return fig