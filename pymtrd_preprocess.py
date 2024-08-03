import numpy as np
from netCDF4 import Dataset

DELTA = 0.000001
THRESHOLD_INTENSITY = 1000


def get_days_year(year):
    """
    Get the number of days in the year according to the input year
    """
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 366
    else:
        return 365


def get_start_end_days(year_start, year):
    """
    get the index range of the daily scale time series of the corrseponding year
    year_start: the start time of the time series
    year: the corresponding year
    """
    index_start = 0
    for i in range(year_start, year):
        index_start += get_days_year(i)
        
    index_end = index_start + get_days_year(year)
    
    return index_start, index_end


def get_days_month(year, month):
    """
    Get the number of days in the corresponding month
    """
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        if get_days_year(year) == 365:
            return 28
        else:
            return 29


def clear_invalid_data(ts_monthly, nyear, nodata_value=-9999):
    """
    Clear invalid data from monthly rainfall data
    ts_monthly: a one-dimensional array of monthly scale
    nyears: the study data length
    nodata_value: a data which means the result is valueless
    """
    for i in range(nyear * 12):
        if ts_monthly[i] < -1 or ts_monthly[i] == nodata_value:
            for j in range(nyear * 12):
                ts_monthly[j] = nodata_value
            return ts_monthly
    return ts_monthly


def daily_to_monthly_onemonth(data_daily, month_days, nrows, ncols, nodata_value=-9999):
    """
    Convert daily scale data to monthly scale data of a month
    data_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns,
    month_days: days of conversion month
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    data_monthly = np.zeros((nrows, ncols))

    for i in range(nrows):
        for j in range(ncols):
            data_monthly[i, j] = nodata_value
            nodata = True
            for k in range(month_days):
                if data_daily[k, i, j] > -1 or data_daily[k, i, j] != nodata_value:
                    if nodata:
                        nodata = False
                        data_monthly[i, j] = data_daily[k, i, j]
                    else:
                        data_monthly[i, j] += data_daily[k, i, j]

    return data_monthly


def daily_to_monthly_oneyear(data_daily, year, nrows, ncols, nodata_value=-9999):
    """
    Convert daily scale data to monthly scale data of a year
    data_daily: a three_dimensional array, the first dimension is time, the second dimension is rows, and the third dimension is columns,
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    days_in_month = []
    for i in range(1, 13):
        days_in_month.append(get_days_month(year, i))
    cumdays_in_month = np.cumsum(days_in_month)

    data_monthly = np.zeros((12, nrows, ncols))
    for mon in range(12):
        index_start = 0
        if mon > 0:
            index_start = cumdays_in_month[mon - 1]
        index_end = cumdays_in_month[mon]
        data_monthly[mon, :, :] = daily_to_monthly_onemonth(
            data_daily[index_start:index_end, :, :], (index_end - index_start), nrows, ncols, nodata_value)

    return data_monthly


def daily_to_monthly_oneyear_parallel(daily_path, name_var, output_path, year, nrows, ncols, nodata_value=-9999):
    """
    Convert daily scale data to monthly scale data of a year which is used for parallel computing
    daily_path: the path of input nc file
    name_var: the var name of precipitation
    output_path: the output path of monthly data
    year: the year of input nc file
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    try:
        print(f'Daily to monthly: {year}')
        ds = Dataset(daily_path)
        days_in_month = []
        for i in range(1, 13):
            days_in_month.append(get_days_month(year, i))
        cumdays_in_month = np.cumsum(days_in_month)
        data_monthly = np.zeros((12, nrows, ncols))
        for mon in range(12):
            index_start = 0
            if mon > 0:
                index_start = cumdays_in_month[mon - 1]
            index_end = cumdays_in_month[mon]
            days_monthly = get_days_month(year, mon+1)
            data_daily = np.zeros((days_monthly, nrows, ncols))
            data_daily[:, :, :] = np.array(ds.variables[name_var][index_start:index_end, :, :])
            data_monthly[mon, :, :] = daily_to_monthly_onemonth(
                data_daily, (index_end - index_start), nrows, ncols, nodata_value)
        np.save(output_path, data_monthly)
    except Exception as e:
        print(f"Error processing year {year}: {e}")


def daily_to_monthly(data_daily, start_year, nyears, nrows, ncols, nodata_value=-9999):
    """
    Convert daily scale data to monthly scale data
    data_daily: a three_dimensional array, the first dimension is time, the second dimension is rows,
    and the third dimension is columns,
    start_year: the start time of the data
    nyears: nyears: the length of data
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    data_monthly = np.zeros((nyears * 12, nrows, ncols))
    index_start = 0
    index_end = 0
    index = 0

    for i in range(start_year, start_year + nyears):
        if get_days_year(i) == 365:
            index_end += 365
        else:
            index_end += 366
        print(f'Daily to monthly: {i}')
        data_monthly[index:index + 12, :, :] = daily_to_monthly_oneyear(
            data_daily[index_start:index_end, :, :], i, nrows, ncols, nodata_value)
        index_start = index_end
        index += 12

    return data_monthly


def get_max_mean_annual_rainfall(sp_monthly, nyears, nrows, ncols, nodata_value=-9999):
    """
    Get the observed maximum mean annual rainfall in the dataset
    sp_monthly: array with the scale of month, 3-dim (time, row, col)
    nyears, the study data length
    nrows: rows in the study area
    ncols: cols in the study area
    nodata_value: a data which means the result is valueless
    """
    max_rainfall = 0.0
    
    for i in range(nrows):
        for j in range(ncols):
            ts_monthly = sp_monthly[:, i, j]
            if ts_monthly[0] < -1 or str(ts_monthly[0]) == '--' or ts_monthly[0] == nodata_value:
                continue
            else:
                all_rainfall = 0.0
                for t in range(nyears * 12):
                    all_rainfall += ts_monthly[t]
                if all_rainfall > max_rainfall:
                    max_rainfall = all_rainfall
    max_rainfall /= nyears
    
    return max_rainfall


def get_max_mean_annual_rainfall_fun2(sp_monthly_avg, nrows, ncols, nodata_value=-9999):

    max_rainfall = 0.0
    
    for i in range(nrows):
        for j in range(ncols):
            ts_monthly = sp_monthly_avg[:, i, j]
            if ts_monthly[0] < -1 or str(ts_monthly[0]) == '--' or ts_monthly[0] == nodata_value:
                continue
            else:
                all_rainfall = 0.0
                for t in range(12):
                    all_rainfall += ts_monthly[t]
                if all_rainfall > max_rainfall:
                    max_rainfall = all_rainfall
    
    return max_rainfall


def get_average_monthly_raster(sp_monthly, nyears, nrows, ncols, nodata_value=-9999):
    """
    Get the average of the input 3D data on the time scale
    sp_monthly: a three dimensional array of monthly precipitation, the first dimension is time,
    the second dimension is rows, and the third dimension is columns
    nyears: length of time of input data
    nrows: rows of input data
    ncols: cols of input data
    nodata_value: invalid value of input data
    """
    sp_monthly_avg = np.zeros((12, nrows, ncols))
    array_nodata = np.zeros(12)
    
    for i in range(12):
        array_nodata[i] = nodata_value
        
    for i in range(nrows):
        for j in range(ncols):
            ts_monthly = sp_monthly[:, i, j]
            if ts_monthly[0] == nodata_value or ts_monthly[0] < -1:
                sp_monthly_avg[:, i, j] = array_nodata
            else:
                for m in range(12):
                    for n in range(nyears):
                        sp_monthly_avg[m, i, j] += ts_monthly[n*12+m]
                    sp_monthly_avg[m, i, j] /= (nyears*1.0)
                    
    return sp_monthly_avg


def get_average_monthly(ts_monthly, nyears, nodata_value=-9999):
    """
    Get the average of the input monthly scale time series
    ts_monthly: a monthly scale time series
    nyears: length of time of input data
    nodata_value: invalid value of input data
    """
    list_monthly_avg = np.zeros(12)
    array_nodata = np.zeros(12)
    
    for i in range(12):
        array_nodata[i] = nodata_value
        
    if ts_monthly[0] < -1 or ts_monthly[0] == nodata_value:
        return array_nodata
    else:
        for i in range(12):
            for j in range(nyears):
                list_monthly_avg[i] += ts_monthly[j*12+i]
            list_monthly_avg[i] /= (nyears*1.0)
            
    return list_monthly_avg


def get_average_daily(ts_daily, nyears, year_start, nodata_value=-9999):
    """
    Get the average of the input daily scale time series
    ts_daily: a daily scale time series
    nyears: length of time of input data
    year_start: the start time of input data
    nodata_value: invalid value of input data
    """
    list_daily_avg = np.zeros(365)
    list_year_start = np.zeros(nyears)
    array_nodata = np.zeros(365)
    
    for i in range(365):
        array_nodata[i] = nodata_value
        
    if ts_daily[0] < -1 or ts_daily[0] == nodata_value:
        return array_nodata
    else:
        list_year_start[0] = 0
        for j in range(1, nyears):
            year_days = get_days_year(j+year_start)
            list_year_start[j] = list_year_start[j-1] + year_days
        for i in range(365):
            for j in range(nyears):
                if ts_daily[int(list_year_start[j]+i)] > -1 and ts_daily[int(list_year_start[j]+i)] != nodata_value:
                    list_daily_avg[i] += ts_daily[int(list_year_start[j]+i)]
            list_daily_avg[i] /= (nyears*1.0)
            
    return list_daily_avg


def get_average_variable(sp_data, nyears, nrows, ncols, nodata_value=-9999):
    """
    Get the average of the input 3D data
    sp_data: a three dimensional array, the first dimension is time,
    the second dimension is rows, and the third dimension is columns
    nyears: length of time of input data
    nrows: rows of input data
    ncols: cols of input data
    nodata_value: invalid value of input data
    """
    array_variable_avg = np.zeros((nrows, ncols))
    
    for i in range(nrows):
        for j in range(ncols):
            ts = sp_data[:, i, j]
            for m in range(nyears):
                array_variable_avg[i, j] += ts[m]
            array_variable_avg[i, j] /= (nyears * 1.0)
            if array_variable_avg[i, j] < -1 or array_variable_avg[i, j] == nodata_value:
                array_variable_avg[i, j] = nodata_value
                
    return array_variable_avg


def get_point_location(list_lon, list_lat, point_lon, point_lat):
    """
    Get the row and col number in space by latitude and longitude
    lon_list: the list of longitude which corresponds to the col number
    lat_list: the list of lantitude which corresponds to the row number
    point_lon: the longitude of the point
    point_lat: the latitude of the point
    """
    col = list_lon.index(min(list_lon, key=lambda x: abs(x - point_lon)))
    row = list_lat.index(min(list_lat, key=lambda x: abs(x - point_lat)))
    return row, col
