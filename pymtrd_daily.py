import pymtrd_preprocess as preprocess
import numpy as np
import heapq
import ctypes
import platform


if platform.system().lower() == 'windows':
    gini_c = ctypes.CDLL('pymtrd_gini.dll', winmode=0)
else:
    gini_c = ctypes.cdll.LoadLibrary('pymtrd_gini.so')


# Rainfall frequency and Consecutive dry days
def stat_interval(ts_daily, nyears, percentile=None, threshold=1, nodata_value=-9999):
    """
    Get annual rainfall times and consecutive dry days of a daily-scale time series rainfall data
    ts_daily: time series with the scale of day
    nyears: the study data length
    percentile: a list of percentile, which is used to calculate cdds
    threshold: the threshold to determine whether it is a rainfall day
    nodata_value: a data which means the result is valueless
    """
    if percentile is None:
        percentile = []
    n = len(ts_daily)
    interval_list = []
    cumu = 0
    index_former = -1
    
    for i in range(n):
        if ts_daily[i] >= threshold:
            interval_list.append(i - index_former - 1)
            index_former = i
        # if ts_daily[i] < threshold:
        #     # consider consecutive days with small rainfall
        #     if ts_daily[i] > 1:
        #         cumu += ts_daily[i]
        #         if cumu >= threshold:
        #             interval_list.append(i - index_former - 1)
        #             index_former = i
        #             cumu = 0
        #     # when the consecutive days end, set cumu to zero
        #     else:
        #         cumu = 0
        # else:
        #     interval_list.append(i - index_former - 1)
        #     index_former = i
            
    per_n = len(percentile)
    if len(interval_list) <= 10:
        if per_n == 0:
            return nodata_value, nodata_value
        else:
            array_nodata = np.zeros(per_n)
            for i in range(per_n):
                array_nodata[i] = nodata_value
            return nodata_value, array_nodata
        
    # annual rainfall times
    times = float(len(interval_list)) / nyears
    
    if per_n == 0:
        cdds = np.zeros(1)
        cdds[0] = np.mean(heapq.nlargest(nyears, interval_list))
        if cdds[0] > 365:
            cdds[0] = 365
    else:
        cdds = np.zeros(per_n)
        for i in range(per_n):
            cdds[i] = np.percentile(interval_list, percentile[i])
            if cdds[i] > 365:
                cdds[i] = 365
                
    return times, cdds


def stat_interval_raster(sp_daily, nyears, nrows, ncols, percentile=None, threshold=1, nodata_value=-9999):
    """
    get annual rainfall times and consecutive dry days of Spatial Data
    sp_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns
    nyears: the study data length
    nrows: rows in the study area
    ncols: cols in the study area
    percentile: a list of percentile, which is used to calculate cdds
    threshold: the threshold to determine whether it is the precipitation day
    nodata_value: a data which means the result is valueless
    """
    if percentile is None:
        percentile = []
    n = len(percentile)
    
    if n == 0:
        n = 1
        
    array_interval = np.zeros((n, nrows, ncols))
    array_times_mean = np.zeros((nrows, ncols))
    array_nodata = np.zeros(n)
    
    for i in range(n):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        row_data = sp_daily[:, i, :]
        for j in range(ncols):
            ts_daily = row_data[:, j]
            if ts_daily[0] == nodata_value or ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                array_times_mean[i, j] = nodata_value
                array_interval[:, i, j] = array_nodata
            else:
                array_times_mean[i, j], array_interval[:, i, j] = stat_interval(
                    ts_daily, nyears, percentile, threshold, nodata_value)
                
    return array_times_mean, array_interval


# Rainfall intensity
def stat_intensity(ts_daily, percentile=None, nodata_value=-9999):
    """
    Get the rainfall intensity of a daily-scale time series rainfall data
    ts_daily: time series with the scale of day
    percentile: the array of percentile, when percentile is empty, return average
    nodata_value: a data which means the result is valueless
    """
    if percentile is None:
        percentile = []
    intensity_list = []
    
    for item in ts_daily:
        if item > 1:
            intensity_list.append(item)
            
    n = len(percentile)
    if n == 0:
        if len(intensity_list) >= 10:
            avg = np.mean(intensity_list)
        else:
            avg = nodata_value
        if avg > preprocess.THRESHOLD_INTENSITY:
            return nodata_value
        return avg
    else:
        p = np.zeros(n)
        array_nodata = np.zeros(n)
        for i in range(n):
            array_nodata[i] = nodata_value
        if len(intensity_list) >= 10:
            for i in range(n):
                p[i] = np.percentile(intensity_list, percentile[i])
                if p[i] > preprocess.THRESHOLD_INTENSITY:
                    p[i] = nodata_value
        else:
            p = array_nodata
        return p


def stat_intensity_raster(sp_daily, nrows, ncols, percentile=None, nodata_value=-9999):
    """
    Get the rainfall intensity of spatial data
    nrows: rows in the study area
    ncols: cols in the study area
    sp_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns,
    percentile: the array of percentile, when percentile is empty, return average
    nodata_value: a data which means the result is valueless
    """
    if percentile is None:
        percentile = []
    n = len(percentile)
    
    if n == 0:
        array_intensity_mean = np.zeros((1, nrows, ncols))
        for i in range(nrows):
            row_data = sp_daily[:, i, :]
            for j in range(ncols):
                ts_daily = row_data[:, j]
                if ts_daily[0] == nodata_value or ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                    array_intensity_mean[0, i, j] = nodata_value
                else:
                    array_intensity_mean[0, i, j] = stat_intensity(ts_daily, percentile, nodata_value)
        return array_intensity_mean
    else:
        array_intensity_percentile = np.zeros((n, nrows, ncols))
        array_nodata = np.zeros(n)
        for i in range(n):
            array_nodata[i] = nodata_value
        for i in range(nrows):
            row_data = sp_daily[:, i, :]
            for j in range(ncols):
                ts_daily = row_data[:, j]
                if ts_daily[0] == nodata_value or ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                    array_intensity_percentile[:, i, j] = array_nodata
                else:
                    array_intensity_percentile[:, i, j] = stat_intensity(ts_daily, percentile, nodata_value)
        return array_intensity_percentile


# unranked Gini index
def get_ugini(ts_daily, nodata_value=-9999):
    """
    get unranked Gini index of a daily-scale time series rainfall data
    ts_daily: time series with the scale of day
    nodata_value: a data which means the result is valueless
    """
    ts_daily = [n for n in ts_daily if (n != nodata_value and n > -1 and str(n) != '--')]
    cum_wealths_u = np.cumsum(ts_daily)
    sum_wealths_u = cum_wealths_u[-1]
    
    if abs(sum_wealths_u) < preprocess.DELTA:
        Ugini = nodata_value
    else:
        xarray_u = np.array(range(0, len(cum_wealths_u))) / float(len(cum_wealths_u) - 1)
        yarray_u = cum_wealths_u / sum_wealths_u
        yarray_u_max = np.maximum(xarray_u, yarray_u)
        yarray_u_min = np.minimum(xarray_u, yarray_u)
        u_max = np.trapz(yarray_u_max, x=xarray_u)
        u_min = np.trapz(yarray_u_min, x=xarray_u)
        Ugini = u_max - u_min
        Ugini *= 2
        
    return Ugini


def stat_ugini(ts_daily, nyears, year_start, ts_long, nodata_value=-9999):
    """
    Take 365/366 as a cycle, take it from the beginning to the end, and output the corresponding minimum
    and maximum values. The missing values in the last year are supplemented with the initial data
    ts_daily: time series with the scale of day
    year_start: the start time of the data
    """
    ugini_min_list = np.zeros(nyears)
    ugini_max_list = np.zeros(nyears)
    gini_c.stat_ugini.restype = ctypes.POINTER(ctypes.c_float)
    """
    if nyears > 1:
        data = np.hstack((ts_daily, ts_daily[:366]))
    else:
        days = preprocess.get_days_year(year_start)
        data = np.hstack((ts_daily, ts_daily[:days]))
    
    data_start0 = 0
    for i in range(nyears):
        days = preprocess.get_days_year(i + year_start)
        UGini = np.zeros(days)
        for j in range(days):
            data_start = data_start0 + j
            data_end = data_start + days
            prec_data = data[data_start:data_end]
            prec_data = np.insert(prec_data, 0, 0)
            prec_data_32 = prec_data.astype(np.float32)
            prec_data_c = prec_data_32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            # UGini[j] = dll.get_ugini(prec_data_c, days, nodata_value)
            UGini[j] = get_ugini(prec_data, nodata_value)
        data_start0 += days
        ugini_min_list[i] = min(UGini)
        ugini_max_list[i] = max(UGini)
    print(ugini_min_list, ugini_max_list)
    """
    data_32 = ts_daily.astype(np.float32)
    ts_daily_c = data_32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ugini_result = gini_c.stat_ugini(ts_daily_c, nyears, year_start, ts_long, nodata_value)

    ugini_list = [ugini_result[i] for i in range(nyears*2)]
    for i in range(nyears):
        ugini_min_list[i] = ugini_list[i]
        ugini_max_list[i] = ugini_list[i + nyears]
    return ugini_min_list, ugini_max_list


def stat_ugini_raster(sp_daily, nyears, nrows, ncols, year_start, ts_long, nodata_value=-9999):
    """
    Take 365/366 as a cycle, take it from the beginning to the end, and output the corresponding minimum
    and maximum values. The missing values in the last year are supplemented with the initial data
    sp_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns,
    nyears: the study data length,
    nrows: rows in the study area
    ncols: cols in the study area
    year_start: the start time of the data,
    nodata_value: a data which means the result is valueless
    """
    array_nodata = np.zeros(nyears)
    array_ugini_min = np.zeros((nyears, nrows, ncols))
    array_ugini_max = np.zeros((nyears, nrows, ncols))
    
    for i in range(nyears):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        for j in range(ncols):
            ts_daily = sp_daily[:, i, j]
            if ts_daily[0] == nodata_value or ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                array_ugini_min[:, i, j] = array_nodata
                array_ugini_max[:, i, j] = array_nodata
            else:
                array_ugini_min[:, i, j], array_ugini_max[:, i, j] = stat_ugini(
                    ts_daily, nyears, year_start, ts_long, nodata_value)
                
    return array_ugini_min, array_ugini_max


# Gini index
def get_gini(ts_daily, nodata_value=-9999):
    """
    get Gini index of a daily-scale time series rainfall data
    ts_daily: time series with the scale of day
    nodata_value: a data which means the result is valueless
    """
    ts_daily = [n for n in ts_daily if (n != nodata_value and n > -1 and str(n) != '--')]
    ts_daily.sort()
    cum_wealths = np.cumsum(ts_daily)
    sum_wealths = cum_wealths[-1]
    
    if abs(sum_wealths) < preprocess.DELTA:
        gini = nodata_value
    else:
        xarray = np.array(range(0, len(cum_wealths))) / float(len(cum_wealths) - 1)
        yarray = cum_wealths / sum_wealths
        B = np.trapz(yarray, x=xarray)
        A = 0.5 - B
        gini = A / (A + B)
        
    return gini


def stat_gini(ts_daily, nyears, year_start, ts_long, nodata_value=-9999):
    """
    Take 365/366 as a cycle, take it from the beginning to the end, and output the corresponding minimum
    and maximum values. The missing values in the last year are supplemented with the initial data
    ts_daily: time series with the scale of day
    year_start: the start time of the data
    """
    gini_min_list = np.zeros(nyears)
    gini_max_list = np.zeros(nyears)
    """
    if nyears > 1:
        data = np.hstack((ts_daily, ts_daily[:366]))
    else:
        days = preprocess.get_days_year(year_start)
        data = np.hstack((ts_daily, ts_daily[:days]))
    
    data_start0 = 0
    for i in range(nyears):
        days = preprocess.get_days_year(i + year_start)
        Gini = np.zeros(days)
        for j in range(days):
            data_start = data_start0 + j
            data_end = data_start + days
            prec_data = data[data_start:data_end]
            prec_data = np.insert(prec_data, 0, 0)
            Gini[j] = get_gini(prec_data, nodata_value)
        data_start0 += days
        gini_min_list[i] = min(Gini)
        gini_max_list[i] = max(Gini)
    """
    gini_c.stat_gini.restype = ctypes.POINTER(ctypes.c_float)
    data_32 = ts_daily.astype(np.float32)
    ts_daily_c = data_32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gini_result = gini_c.stat_gini(ts_daily_c, nyears, year_start, ts_long, nodata_value)

    gini_list = [gini_result[i] for i in range(nyears * 2)]
    for i in range(nyears):
        gini_min_list[i] = gini_list[i]
        gini_max_list[i] = gini_list[i + nyears]
    return gini_min_list, gini_max_list


def stat_gini_raster(sp_daily, nyears, nrows, ncols, year_start, ts_long, nodata_value=-9999):
    """
    Take 365/366 as a cycle, take it from the beginning to the end, and output the corresponding minimum
    and maximum values. The missing values in the last year are supplemented with the initial data
    sp_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns,
    nyears: the study data length,
    nrows: rows in the study area
    ncols: cols in the study area
    year_start: the start time of the data,
    nodata_value: a data which means the result is valueless
    """
    array_nodata = np.zeros(nyears)
    array_gini_min = np.zeros((nyears, nrows, ncols))
    array_gini_max = np.zeros((nyears, nrows, ncols))
    
    for i in range(nyears):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        for j in range(ncols):
            ts_daily = sp_daily[:, i, j]
            if ts_daily[0] == nodata_value or ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                array_gini_min[:, i, j] = array_nodata
                array_gini_max[:, i, j] = array_nodata
            else:
                array_gini_min[:, i, j], array_gini_max[:, i, j] = stat_gini(ts_daily, nyears, year_start, ts_long, nodata_value)
    return array_gini_min, array_gini_max


# wet-day Gini index
def get_wgini(ts_daily, threshold=1, nodata_value=-9999):
    """
    get Wet-day Gini index of a time series
    ts_daily: time series with the scale of day
    threshold: the threshold to determine whether it is the precipitation day
    nodata_value: a data which means the result is valueless
    """
    ts_daily = [n for n in ts_daily if (n != nodata_value and n > -1 and str(n) != '--')]
    ts_daily.sort()
    data_w = [n for n in ts_daily if n > threshold]
    data_w.insert(0, 0)
    cum_wealths_w = np.cumsum(data_w)
    sum_wealths_w = cum_wealths_w[-1]
    
    if abs(sum_wealths_w) < preprocess.DELTA:
        wgini = nodata_value
    else:
        xarray_w = np.array(range(0, len(cum_wealths_w))) / float(len(cum_wealths_w) - 1)
        yarray_w = cum_wealths_w / sum_wealths_w
        B_W = np.trapz(yarray_w, x=xarray_w)
        A_W = 0.5 - B_W
        wgini = A_W / (A_W + B_W)
        
    return wgini


def stat_wgini(ts_daily, nyears, year_start, ts_long, threshold=1, nodata_value=-9999):
    """
    Take 365/366 as a cycle, take it from the beginning to the end, and output the corresponding minimum
    and maximum values. The missing values in the last year are supplemented with the initial data
    ts_daily: time series with the scale of day
    year_start: the start time of the data
    """
    wgini_min_list = np.zeros(nyears)
    wgini_max_list = np.zeros(nyears)
    """
    if nyears > 1:
        data = np.hstack((ts_daily, ts_daily[:366]))
    else:
        days = preprocess.get_days_year(year_start)
        data = np.hstack((ts_daily, ts_daily[:days]))
    
    data_start0 = 0
    for i in range(nyears):
        days = preprocess.get_days_year(i + year_start)
        WGini = np.zeros(days)
        for j in range(days):
            data_start = data_start0 + j
            data_end = data_start + days
            prec_data = data[data_start:data_end]
            prec_data = np.insert(prec_data, 0, 0)
            WGini[j] = get_wgini(prec_data, threshold, nodata_value)
        data_start0 += days
        wgini_min_list[i] = min(WGini)
        wgini_max_list[i] = max(WGini)
    """
    gini_c.stat_wgini.restype = ctypes.POINTER(ctypes.c_float)
    data_32 = ts_daily.astype(np.float32)
    ts_daily_c = data_32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    wgini_result = gini_c.stat_wgini(ts_daily_c, nyears, year_start, ts_long, nodata_value, ctypes.c_float(threshold))

    wgini_list = [wgini_result[i] for i in range(nyears * 2)]
    for i in range(nyears):
        wgini_min_list[i] = wgini_list[i]
        wgini_max_list[i] = wgini_list[i + nyears]
    return wgini_min_list, wgini_max_list


def stat_wgini_raster(sp_daily, nyears, nrows, ncols, year_start, ts_long, threshold=1, nodata_value=-9999):
    """
    Take 365/366 as a cycle, take it from the beginning to the end, and output the corresponding minimum
    and maximum values. The missing values in the last year are supplemented with the initial data
    sp_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns,
    nyears: the study data length,
    nrows: rows in the study area
    ncols: cols in the study area
    year_start: the start time of the data,
    nodata_value: a data which means the result is valueless
    """
    array_nodata = np.zeros(nyears)
    array_wgini_min = np.zeros((nyears, nrows, ncols))
    array_wgini_max = np.zeros((nyears, nrows, ncols))
    
    for i in range(nyears):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        for j in range(ncols):
            ts_daily = sp_daily[:, i, j]
            if ts_daily[0] == nodata_value or ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                array_wgini_min[:, i, j] = array_nodata
                array_wgini_max[:, i, j] = array_nodata
            else:
                array_wgini_min[:, i, j], array_wgini_max[:, i, j] = stat_wgini(
                    ts_daily, nyears, year_start, ts_long, threshold, nodata_value)
                
    return array_wgini_min, array_wgini_max
