import pymtrd_preprocess as preprocess
import numpy as np


# Precipitation concentration index
def stat_pci(ts_monthly, nyears, nodata_value=-9999):
    """
    Get precipitation concentration index of a monthly-scale time series
    ts_monthly, time series with the scale of month
    nyears, the study data length
    nodata_value: a data which means the result is valueless
    """
    pci = np.zeros(nyears)
    
    for i in range(nyears):
        x = 0.0
        y = 0.0
        for j in range(12):
            x += ts_monthly[i * 12 + j] * ts_monthly[i * 12 + j]
            y += ts_monthly[i * 12 + j]
        pci[i] = nodata_value
        if abs(y) > preprocess.DELTA:
            pci[i] = 100.0 * x / y / y
            
    return pci


def stat_pci_raster(sp_monthly, nyears, nrows, ncols, nodata_value=-9999):
    """
    Get precipitation concentration index months of spatial data
    ts_monthly, array with the scale of month
    nyears, the study data length
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    pci = np.zeros((nyears, nrows, ncols))
    array_nodata = np.zeros(nyears)
    
    for i in range(nyears):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        row_data = sp_monthly[:, i, :]
        for j in range(ncols):
            ts_monthly = row_data[:, j]
            if ts_monthly[0] == nodata_value or ts_monthly[0] < -1 or str(ts_monthly[0]) == '--':
                pci[:, i, j] = array_nodata
            else:
                pci[:, i, j] = stat_pci(ts_monthly, nyears, nodata_value)
                
    return pci


# dimensionless seasonality index
def stat_dsi(ts_monthly, nyears, value_max_rainfall=100, nodata_value=-9999):
    """
    Get dimensionless seasonality index of a monthly-scale time series
    ts_monthly, time series with the scale of month
    nyears, the study data length
    value_max_rainfall, observed maximum mean annual rainfall in the dataset
    """
    dsi = 0
    ts_monthly_avg = np.zeros(12)
    year_avg = 0
    
    for i in range(nyears):
        for j in range(12):
            ts_monthly_avg[j] += ts_monthly[i * 12 + j]
            
    for i in range(12):
        ts_monthly_avg[i] /= 12.0
        year_avg += ts_monthly_avg[i]
        
    for i in range(12):
        if year_avg <= 0 or year_avg <= preprocess.DELTA:
            return nodata_value
        pm = ts_monthly_avg[i] / year_avg
        value = 0
        if pm > 0:
            value = pm * np.log2(pm * 12)
        dsi += value
        
    # relative entropy to dimensionless seasonality index
    dsi = dsi * year_avg / value_max_rainfall
    
    return dsi


def stat_dsi_raster(sp_monthly, nyears, nrows, ncols, value_max_rainfall, nodata_value=-9999):
    """
    Get dimensionless seasonality index of spatial data
    ts_monthly, array with the scale of month
    nyears, the study data length
    nrows: rows in the study area
    ncols: cols in the study area
    value_max_rainfall, observed maximum mean annual rainfall in the dataset
    nodata_value: a data which means the result is valueless
    """
    dsi = np.zeros((nrows, ncols))
    
    for i in range(nrows):
        row_data = sp_monthly[:, i, :]
        for j in range(ncols):
            ts_monthly = row_data[:, j]
            if ts_monthly[0] == nodata_value or ts_monthly[0] < -1 or str(ts_monthly[0]) == '--':
                dsi[i, j] = nodata_value
            else:
                dsi[i, j] = stat_dsi(ts_monthly, nyears, value_max_rainfall, nodata_value)
                
    return dsi


# Seasonality Index
def stat_si(ts_monthly, nyears, nodata_value=-9999):
    """
    Get seasonality index of a monthly-scale time series
    ts_monthly, time series with the scale of month
    nyears, the study data length
    nodata_value: a data which means the result is valueless
    """
    r = np.zeros(nyears)
    si = np.zeros(nyears)
    
    for i in range(nyears):
        for j in range(12):
            r[i] += ts_monthly[i * 12 + j]
            
    for i in range(nyears):
        for j in range(12):
            si[i] += abs(ts_monthly[i * 12 + j] - r[i] / 12.0)
            
        if abs(r[i]) > preprocess.DELTA:
            si[i] /= r[i]
        else:
            si[i] = nodata_value
            
    return si


def stat_si_raster(sp_monthly, nyears, nrows, ncols, nodata_value=-9999):
    """
    Get seasonality index of spatial data
    ts_monthly, array with the scale of month
    nyears, the study data length
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    si = np.zeros((nyears, nrows, ncols))
    array_nodata = np.zeros(nyears)
    
    for i in range(nyears):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        row_data = sp_monthly[:, i, :]
        for j in range(ncols):
            ts_monthly = row_data[:, j]
            if ts_monthly[0] == nodata_value or ts_monthly[0] < -1 or str(ts_monthly[0]) == '--':
                si[:, i, j] = array_nodata
            else:
                si[:, i, j] = stat_si(ts_monthly, nyears, nodata_value)
                
    return si
