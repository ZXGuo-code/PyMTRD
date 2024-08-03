import shutil
from osgeo import gdal,osr
from netCDF4 import Dataset
import numpy as np
import os
import xarray as xr
import multiprocessing as mp
import pandas as pd
import time
import pymtrd_daily as daily
import pymtrd_monthly as monthly
import pymtrd_preprocess as preprocess
import pymtrd_draw as draw


def process(config_path, bool_draw):
    """
    Determine the parameters in the csv file to calculate the corresponding indicators and 
    output them, as well as plot the corresponding graphs
    """
    start_time = time.time()
    df = pd.read_csv(config_path, header=None)
    row_csv = 0
    product_name = df.iat[row_csv, 0]
    row_csv += 1
    num_files = int(df.iat[row_csv, 1])
    name_var = df.iat[row_csv, 2]
    name_lon = df.iat[row_csv, 3]
    name_lat = df.iat[row_csv, 4]
    
    # Get the list of files
    list_files = []
    for i in range(num_files):
        row_csv += 1
        list_files.append(df.iat[row_csv, 1])
    
    # get basic information of nc
    filename = 'NETCDF:"%s":%s' % (list_files[0], name_var)
    ds = gdal.Open(filename)
    row_nc = ds.RasterYSize
    col_nc = ds.RasterXSize
    geotrans = ds.GetGeoTransform()
    projinfo = ds.GetProjection()
    if projinfo == '':
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        projinfo = outRasterSRS.ExportToWkt()
    print('Bacial information of nc file:')
    print(row_nc, col_nc, geotrans, projinfo)
    # get lon and lat list of nc
    ds = xr.open_dataset(list_files[0])
    list_lon = ds[name_lon].values.tolist()
    list_lat = ds[name_lat].values.tolist()
    
    # get the start year and end year
    row_csv += 1
    year_start = int(df.iat[row_csv, 1])
    row_csv += 1
    year_end = int(df.iat[row_csv, 1])
    # get the nodata_value of the nc file
    row_csv += 1
    nodata_value = int(df.iat[row_csv, 1])
    # time_length
    nyears = year_end - year_start + 1
    year_list = range(year_start, year_end + 1)
    # get the total number of study time
    alldays = 0
    for year in year_list:
        alldays += preprocess.get_days_year(year)
    
    data_daily = np.zeros((alldays, row_nc, col_nc))
    
    data_start = 0
    print('read nc file')
    for year in year_list:
        print(year)
        year_index = year - year_start
        ds = Dataset(list_files[year_index])
        data = ds.variables[name_var]
        var_data = np.array(data)
        days_year = preprocess.get_days_year(year)
        data_daily[data_start:data_start + days_year, :, :] = var_data
        data_start += days_year
    
    # operation model 1 pixel by pixel 2 parameter by parameter
    row_csv += 1
    model_process = int(df.iat[row_csv, 1])
    
    # File output address
    row_csv += 1
    path_output = df.iat[row_csv, 1]
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    
    # Whether to calculate rain intensity
    row_csv += 1
    bool_intensity = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate rain intensity by percentile. if not, the result will be the average
    row_csv += 1
    bool_percentile_intensity = bool(int(df.iat[row_csv, 1]))
    if bool_percentile_intensity:
        list_percentile_intensity = df.iat[row_csv, 2].split()
        list_percentile_intensity = list(map(float, list_percentile_intensity))
    else:
        list_percentile_intensity = []
    
    # Whether to calculate rain times
    row_csv += 1
    bool_times = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate cdds
    row_csv += 1
    bool_cdds = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate cdds by percentile. if not, the result will be the average
    row_csv += 1
    bool_percentile_cdds = bool(int(df.iat[row_csv, 1]))
    if bool_percentile_cdds:
        list_percentile_cdds = df.iat[row_csv, 2].split()
        list_percentile_cdds = list(map(float, list_percentile_cdds))
    else:
        list_percentile_cdds = []
    # Whether to calculate times and cdds by specified thresholds, if not, the threshold will be set to 1
    row_csv += 1
    bool_threshold_times_cdds = bool(int(df.iat[row_csv, 1]))
    if bool_threshold_times_cdds:
        list_threshold_times_cdds = df.iat[row_csv, 2].split()
        list_threshold_times_cdds = list(map(float, list_threshold_times_cdds))
    else:
        list_threshold_times_cdds = np.zeros(1)
        list_threshold_times_cdds[0] = 1
    
    # whether to calculate ugini index
    row_csv += 1
    bool_ugini = bool(int(df.iat[row_csv, 1]))
    # whether to calculate gini index
    row_csv += 1
    bool_gini = bool(int(df.iat[row_csv, 1]))
    # whether to calculate wgini index
    row_csv += 1
    bool_wgini = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate wgini index by specified threshold(which can spend a lot of time),
    # if not, the threshold will be set to 1
    row_csv += 1
    bool_threshold_wgini = bool(int(df.iat[row_csv, 1]))
    if bool_threshold_wgini:
        list_threshold_wgini = df.iat[row_csv, 2].split()
        list_threshold_wgini = list(map(float, list_threshold_wgini))
    else:
        list_threshold_wgini = np.zeros(1)
        list_threshold_wgini[0] = 1
    
    # Whether to calculate pci
    row_csv += 1
    bool_pci = bool(int(df.iat[row_csv, 1]))
    
    # Whether to calculate dsi
    row_csv += 1
    bool_dsi = bool(int(df.iat[row_csv, 1]))
    
    # Whether to calculate si
    row_csv += 1
    bool_si = bool(int(df.iat[row_csv, 1]))
    
    # Creating space for variables
    # intensity
    array_intensity = None
    if bool_intensity:
        if bool_percentile_intensity:
            array_intensity = np.zeros((len(list_percentile_intensity), row_nc, col_nc))
        else:
            array_intensity = np.zeros((1, row_nc, col_nc))
    # time or cdds
    array_times = None
    array_cdds = None
    if bool_times or bool_cdds:
        array_times = np.zeros((len(list_threshold_times_cdds), row_nc, col_nc))
        if bool_percentile_cdds:
            array_cdds = np.zeros((len(list_threshold_times_cdds), len(list_percentile_cdds), row_nc, col_nc))
        else:
            array_cdds = np.zeros((len(list_threshold_times_cdds), 1, row_nc, col_nc))
    
    # gini
    array_gini_min = None
    array_gini_max = None
    array_ugini_min = None
    array_ugini_max = None
    array_wgini_min = None
    array_wgini_max = None
    if bool_gini:
        array_gini_min = np.zeros((nyears, row_nc, col_nc))
        array_gini_max = np.zeros((nyears, row_nc, col_nc))
    if bool_ugini:
        array_ugini_min = np.zeros((nyears, row_nc, col_nc))
        array_ugini_max = np.zeros((nyears, row_nc, col_nc))
    if bool_wgini:
        array_wgini_min = np.zeros((len(list_threshold_wgini), nyears, row_nc, col_nc))
        array_wgini_max = np.zeros((len(list_threshold_wgini), nyears, row_nc, col_nc))
    
    # pci
    array_pci = None
    array_pci_avg = None
    if bool_pci:
        array_pci = np.zeros((nyears, row_nc, col_nc))
        array_pci_avg = np.zeros((row_nc, col_nc))
    # dsi
    array_dsi = None
    if bool_dsi:
        array_dsi = np.zeros((row_nc, col_nc))
    # si
    array_si = None
    array_si_avg = None
    if bool_si:
        array_si = np.zeros((nyears, row_nc, col_nc))
        array_si_avg = np.zeros((row_nc, col_nc))
    
    # monthly precipitation
    data_monthly = None
    data_monthly_avg = None
    if bool_pci or bool_dsi or bool_pci:
        print('Daily to monthly')
        data_monthly = preprocess.daily_to_monthly(
            data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
        for i in range(row_nc):
            for j in range(col_nc):
                ts = data_monthly[:, i, j]
                data_monthly[:, i, j] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
        data_monthly_avg = preprocess.get_average_monthly_raster(data_monthly, nyears, row_nc, col_nc, nodata_value)
    print('Daily to monthly done!')
    # maximum mean annual rainfall in the dataset
    value_max_rainfall = 0.0
    if bool_dsi:
        value_max_rainfall = preprocess.get_max_mean_annual_rainfall_fun2(data_monthly_avg, row_nc, col_nc, nodata_value)
    print("Calculate the metrics of temporal rainfall distribution")
    # Data Processing
    if model_process == 1:
        
        # list/array nodata
        len_intensity = 1
        if bool_percentile_intensity:
            len_intensity = len(list_percentile_intensity)
        list_nodata_intensity = np.zeros(len_intensity)
        for i in range(len_intensity):
            list_nodata_intensity[i] = nodata_value
        
        len_times = 1
        if bool_threshold_times_cdds:
            len_times = len(list_threshold_times_cdds)
        list_nodata_times = np.zeros(len_times)
        for i in range(len_times):
            list_nodata_times = nodata_value
        
        len_cdds = 1
        if bool_percentile_cdds:
            len_cdds = len(list_percentile_cdds)
        array_nodata_cdds = np.zeros((len_times, len_cdds))
        for i in range(len_cdds):
            array_nodata_cdds[:, i] = list_nodata_times
        
        list_nodata_gini = np.zeros(nyears)
        for i in range(nyears):
            list_nodata_gini[i] = nodata_value
        len_wgini = 1
        if bool_wgini:
            len_wgini = len(list_threshold_wgini)
        array_nodata_wgini = np.zeros((len_wgini, nyears))
        for i in range(len_wgini):
            array_nodata_wgini[i, :] = list_nodata_gini
        # row_nc
        for i in range(row_nc):
            print(f'Current calculation row number: {i} Total row number: {row_nc} ')
            for j in range(col_nc):
                ts_daily = data_daily[:, i, j]
                if ts_daily[0] < -1 or str(ts_daily[0]) == '--':
                    if bool_intensity:
                        array_intensity[:, i, j] = list_nodata_intensity
                    if bool_times or bool_cdds:
                        array_cdds[:, :, i, j] = array_nodata_cdds
                        array_times[:, i, j] = list_nodata_times
                    if bool_ugini:
                        array_ugini_max[:, i, j] = list_nodata_gini
                        array_ugini_min[:, i, j] = list_nodata_gini
                    if bool_gini:
                        array_gini_max[:, i, j] = list_nodata_gini
                        array_gini_min[:, i, j] = list_nodata_gini
                    if bool_wgini:
                        array_wgini_max[:, :, i, j] = array_nodata_wgini
                        array_wgini_min[:, :, i, j] = array_nodata_wgini
                
                else:
                    if bool_intensity:
                        array_intensity[:, i, j] = daily.stat_intensity(
                            ts_daily=ts_daily,
                            percentile=list_percentile_intensity,
                            nodata_value=nodata_value)
                    if bool_cdds or bool_times:
                        for threshold in range(len(list_threshold_times_cdds)):
                            array_times[threshold, i, j], array_cdds[threshold, :, i, j] = daily.stat_interval(
                                ts_daily=ts_daily,
                                nyears=nyears,
                                percentile=list_percentile_cdds,
                                threshold=list_threshold_times_cdds[threshold],
                                nodata_value=nodata_value)
                    if bool_ugini:
                        array_ugini_min[:, i, j], array_ugini_max[:, i, j] = daily.stat_ugini(
                            ts_daily=ts_daily,
                            nyears=nyears,
                            year_start=year_start,
                            ts_long=alldays,
                            nodata_value=nodata_value)
                    if bool_gini:
                        array_gini_min[:, i, j], array_gini_max[:, i, j] = daily.stat_gini(
                            ts_daily=ts_daily,
                            nyears=nyears,
                            year_start=year_start,
                            ts_long=alldays,
                            nodata_value=nodata_value)
                    if bool_wgini:
                        for threshold in range(len(list_threshold_wgini)):
                            array_wgini_min[threshold, :, i, j], array_wgini_max[threshold, :, i, j] = daily.stat_wgini(
                                ts_daily=ts_daily,
                                nyears=nyears,
                                year_start=year_start,
                                ts_long=alldays,
                                threshold=list_threshold_wgini[threshold],
                                nodata_value=nodata_value)
                
                if data_monthly is not None:
                    ts_monthly = data_monthly[:, i, j]
                    ts_monthly_avg = data_monthly_avg[:, i, j]
                    if ts_monthly[0] == nodata_value:
                        if bool_pci:
                            array_pci[:, i, j] = list_nodata_gini
                            array_pci_avg[i, j] = nodata_value
                        if bool_dsi:
                            array_dsi[i, j] = nodata_value
                        if bool_si:
                            array_si[:, i, j] = list_nodata_gini
                            array_si_avg[i, j] = nodata_value
                    else:
                        if bool_pci:
                            array_pci[:, i, j] = monthly.stat_pci(
                                ts_monthly=ts_monthly,
                                nyears=nyears,
                                nodata_value=nodata_value)
                            array_pci_avg[i, j] = monthly.stat_pci(
                                ts_monthly=ts_monthly_avg,
                                nyears=1,
                                nodata_value=nodata_value)
                        if bool_dsi:
                            array_dsi[i, j] = monthly.stat_dsi(
                                ts_monthly=ts_monthly,
                                nyears=nyears,
                                value_max_rainfall=value_max_rainfall,
                                nodata_value=nodata_value)
                        if bool_si:
                            array_si[:, i, j] = monthly.stat_si(
                                ts_monthly=ts_monthly,
                                nyears=nyears,
                                nodata_value=nodata_value)
                            array_si_avg[i, j] = monthly.stat_si(
                                ts_monthly=ts_monthly_avg,
                                nyears=1,
                                nodata_value=nodata_value
                            )
    
    elif model_process == 2:
        if bool_intensity:
            array_intensity = daily.stat_intensity_raster(
                sp_daily=data_daily,
                nrows=row_nc,
                ncols=col_nc,
                percentile=list_percentile_intensity,
                nodata_value=nodata_value)
        if bool_cdds or bool_times:
            for i in range(len(list_threshold_times_cdds)):
                array_times[i], array_cdds[i] = daily.stat_interval_raster(
                    sp_daily=data_daily,
                    nyears=nyears,
                    nrows=row_nc,
                    ncols=col_nc,
                    percentile=list_percentile_cdds,
                    threshold=list_threshold_times_cdds[i],
                    nodata_value=nodata_value)
        if bool_ugini:
            array_ugini_min, array_ugini_max = daily.stat_ugini_raster(
                sp_daily=data_daily,
                nyears=nyears,
                nrows=row_nc,
                ncols=col_nc,
                year_start=year_start,
                ts_long=alldays,
                nodata_value=nodata_value)
        if bool_gini:
            array_gini_min, array_gini_max = daily.stat_gini_raster(
                sp_daily=data_daily,
                nyears=nyears,
                nrows=row_nc,
                ncols=col_nc,
                year_start=year_start,
                ts_long=alldays,
                nodata_value=nodata_value)
        if bool_wgini:
            for i in range(len(list_threshold_wgini)):
                array_wgini_min[i, :, :], array_wgini_max[i, :, :] = daily.stat_wgini_raster(
                    sp_daily=data_daily,
                    nyears=nyears,
                    nrows=row_nc,
                    ncols=col_nc,
                    year_start=year_start,
                    ts_long=alldays,
                    threshold=list_threshold_wgini[i],
                    nodata_value=nodata_value)
        if bool_pci:
            array_pci = monthly.stat_pci_raster(
                sp_monthly=data_monthly,
                nyears=nyears,
                nrows=row_nc,
                ncols=col_nc,
                nodata_value=nodata_value)
            array_pci_avg[:, :] = monthly.stat_pci_raster(
                sp_monthly=data_monthly_avg,
                nyears=1,
                nrows=row_nc,
                ncols=col_nc,
                nodata_value=nodata_value
            )
        if bool_dsi:
            array_dsi = monthly.stat_dsi_raster(
                sp_monthly=data_monthly,
                nyears=nyears,
                nrows=row_nc,
                ncols=col_nc,
                value_max_rainfall=value_max_rainfall,
                nodata_value=nodata_value)
        if bool_si:
            array_si = monthly.stat_si_raster(
                sp_monthly=data_monthly,
                nyears=nyears,
                nrows=row_nc,
                ncols=col_nc,
                nodata_value=nodata_value)
            array_si_avg[:, :] = monthly.stat_si_raster(
                sp_monthly=data_monthly_avg,
                nyears=1,
                nrows=row_nc,
                ncols=col_nc,
                nodata_value=nodata_value
            )
    print("Calculate the metrics of temporal rainfall distribution done!")
    print("Output the metrics of temporal rainfall distribution as tif files")
    # output(write tiff)
    if bool_intensity:
        if bool_percentile_intensity:
            for i in range(len(list_percentile_intensity)):
                outputfile_amount = path_output + os.sep + '%s_intensity_%dpercentile_%04dto%04d.tif' % (
                    product_name, list_percentile_intensity[i], year_start, year_end)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_intensity[i])
        else:
            outputfile_amount = path_output + os.sep + '%s_intensity_mean_%04dto%04d.tif' % (
                product_name, year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_intensity[0])
    if bool_times:
        for i in range(len(list_threshold_times_cdds)):
            outputfile_amount = path_output + os.sep + '%s_frequency_%dmm_mean_%04dto%04d.tif' % (
                product_name, list_threshold_times_cdds[i], year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_times[i])
    if bool_cdds:
        for i in range(len(list_threshold_times_cdds)):
            if bool_percentile_cdds:
                for j in range(len(list_percentile_cdds)):
                    outputfile_amount = path_output + '/%s_cdd_%dmm_%dpercentile_%04dto%04d.tif' % (
                        product_name, list_threshold_times_cdds[i], list_percentile_cdds[j], year_start, year_end)
                    draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_cdds[i, j])
            else:
                outputfile_amount = path_output + '/%s_cdd_%dmm_mean_%04dto%04d.tif' % (
                    product_name, list_threshold_times_cdds[i], year_start, year_end)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_cdds[i, 0])
    if bool_ugini:
        array_ugini_min_avg = preprocess.get_average_variable(array_ugini_min, nyears, row_nc, col_nc, nodata_value)
        array_ugini_max_avg = preprocess.get_average_variable(array_ugini_max, nyears, row_nc, col_nc, nodata_value)
        outputfile_amount = path_output + '/%s_ugini_min_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_min_avg)
        outputfile_amount = path_output + '/%s_ugini_max_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_max_avg)
        for i in range(nyears):
            outputfile_amount = path_output + '/%s_ugini_min_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_min[i])
            outputfile_amount = path_output + '/%s_ugini_max_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_max[i])
    if bool_gini:
        array_gini_min_avg = preprocess.get_average_variable(array_gini_min, nyears, row_nc, col_nc, nodata_value)
        array_gini_max_avg = preprocess.get_average_variable(array_gini_max, nyears, row_nc, col_nc, nodata_value)
        outputfile_amount = path_output + '/%s_gini_min_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_min_avg)
        outputfile_amount = path_output + '/%s_gini_max_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_max_avg)
        for i in range(nyears):
            outputfile_amount = path_output + '/%s_gini_min_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_min[i])
            outputfile_amount = path_output + '/%s_gini_max_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_max[i])
    if bool_wgini:
        for i in range(len(list_threshold_wgini)):
            array_wgini_min_avg = preprocess.get_average_variable(array_wgini_min[i], nyears, row_nc, col_nc, nodata_value)
            array_wgini_max_avg = preprocess.get_average_variable(array_wgini_max[i], nyears, row_nc, col_nc, nodata_value)
            outputfile_amount = path_output + '/%s_wgini_min_%dmm_from%04dto%04d_avg.tif' % (
                product_name, list_threshold_wgini[i], year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_wgini_min_avg)
            outputfile_amount = path_output + '/%s_wgini_max_%dmm_from%04dto%04d_avg.tif' % (
                product_name, list_threshold_wgini[i], year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_wgini_max_avg)
            for j in range(nyears):
                outputfile_amount = path_output + '/%s_wgini_min_%dmm_%04d.tif' % (
                    product_name, list_threshold_wgini[i], j + year_start)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_wgini_min[i, j])
                outputfile_amount = path_output + '/%s_wgini_max_%dmm_%04d.tif' % (
                    product_name, list_threshold_wgini[i], j + year_start)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_wgini_max[i, j])
    if bool_pci:
        outputfile_amount = path_output + '/%s_pci_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_pci_avg)
        """
        array_pci_avg = preprocess.get_average_variable(array_pci, nyears, row_nc, col_nc, nodata_value)
        outputfile_amount = path_output + '/%s_pci_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_pci_avg)
        """
        for i in range(nyears):
            outputfile_amount = path_output + '/%s_pci_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_pci[i])
    if bool_si:
        """
        array_si_avg = preprocess.get_average_variable(array_si, nyears, row_nc, col_nc, nodata_value)
        """
        outputfile_amount = path_output + '/%s_si_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_si_avg)
        for i in range(nyears):
            outputfile_amount = path_output + '/%s_si_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_si[i])
    if bool_dsi:
        outputfile_amount = path_output + '/%s_dsi_avg_%04dto%04d.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_dsi)
    print("Output the metrics of temporal rainfall distribution as tif files done!")

    if bool_draw:
        # draw_picture
        print("Draw picture")
        for i in range(row_csv + 1, df.shape[0]):
            print(df.iat[i, 0])
            # draw cor_var
            if df.iat[i, 0] == 'draw1':
                cor_var1 = df.iat[i, 1]
                cor_var2 = df.iat[i, 2]
                cor_var1_name = df.iat[i, 3]
                cor_var2_name = df.iat[i, 4]
                cor_year = int(df.iat[i, 5])
                cor_name = df.iat[i, 6]
                cor_year_index = cor_year - year_start
                cor_path = path_output + os.sep + '/%s' % cor_name
                if eval(cor_var1) is None or eval(cor_var2) is None:
                    print("draw1 error: do not calculate the input var")
                    continue
                if cor_var1 == 'array_dsi':
                    cor_var1_data = eval(cor_var1)
                else:
                    cor_var1_data = eval(cor_var1)[cor_year_index]
                if cor_var2 == 'array_dsi':
                    cor_var2_data = eval(cor_var2)
                else:
                    cor_var2_data = eval(cor_var2)[cor_year_index]
                draw.draw_cor_var(cor_var1_data, cor_var2_data, cor_var1_name, cor_var2_name, cor_path)
            # draw gini_index of a year
            elif df.iat[i, 0] == 'draw2':
                wgini_threshold = float(df.iat[i, 1])
                gini_year = int(df.iat[i, 2])
                gini_lon = float(df.iat[i, 3])
                gini_lat = float(df.iat[i, 4])
                gini_name = df.iat[i, 5]
                gini_path = path_output + os.sep + '/%s' % gini_name
                gini_row, gini_col = preprocess.get_point_location(list_lon, list_lat, gini_lon, gini_lat)
                gini_day_start_index, gini_day_end_index = preprocess.get_start_end_days(year_start, gini_year)
                if data_daily[gini_day_start_index, gini_row, gini_col] < -1:
                    print("gini: data invalid!!!")
                    continue
                draw.draw_gini_all(
                    data_daily[gini_day_start_index:gini_day_end_index, gini_row, gini_col], wgini_threshold, gini_path)
            # draw curve_daily of a year
            elif df.iat[i, 0] == 'draw3':
                curve_daily_year = int(df.iat[i, 1])
                curve_daily_lon = float(df.iat[i, 2])
                curve_daily_lat = float(df.iat[i, 3])
                curve_daily_name = df.iat[i, 4]
                curve_daily_start_index, curve_daily_end_index = preprocess.get_start_end_days(year_start, curve_daily_year)
                curve_daily_row, curve_daily_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_daily_lon, curve_daily_lat)
                curve_daily_index = list(range(1, preprocess.get_days_year(curve_daily_year) + 1))
                curve_daily_path = path_output + os.sep + '/%s' % curve_daily_name
                draw.draw_curve_precipitation_daily(
                    data_daily[curve_daily_start_index:curve_daily_end_index, curve_daily_row, curve_daily_col],
                    curve_daily_index, curve_daily_path)
            # draw curve_monthly of a year
            elif df.iat[i, 0] == 'draw4':
                curve_monthly_year = int(df.iat[i, 1])
                curve_monthly_lon = float(df.iat[i, 2])
                curve_monthly_lat = float(df.iat[i, 3])
                curve_monthly_name = df.iat[i, 4]
                curve_monthly_start_index = 12 * (curve_monthly_year - year_start)
                curve_monthly_end_index = curve_monthly_start_index + 12
                curve_monthly_row, curve_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_monthly_lon, curve_monthly_lat)
                curve_monthly_index = list(range(1, 13))
                curve_monthly_path = path_output + os.sep + '/%s' % curve_monthly_name
                if data_monthly is None:
                    data_monthly = preprocess.daily_to_monthly(
                        data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                    for m in range(row_nc):
                        for n in range(col_nc):
                            ts = data_monthly[:, m, n]
                            data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
                draw.draw_curve_precipitation_monthly(
                    data_monthly[curve_monthly_start_index:curve_monthly_end_index, curve_monthly_row, curve_monthly_col],
                    curve_monthly_index, curve_monthly_path)
            # draw his_daily of a year
            elif df.iat[i, 0] == 'draw5':
                his_daily_year = int(df.iat[i, 1])
                his_daily_lon = float(df.iat[i, 2])
                his_daily_lat = float(df.iat[i, 3])
                his_daily_name = df.iat[i, 4]
                his_daily_start_index, his_daily_end_index = preprocess.get_start_end_days(year_start, his_daily_year)
                his_daily_row, his_daily_col = preprocess.get_point_location(list_lon, list_lat, his_daily_lon, his_daily_lat)
                his_daily_index = list(range(1, preprocess.get_days_year(his_daily_year) + 1))
                his_daily_path = path_output + os.sep + '/%s' % his_daily_name
                draw.draw_his_precipitation_daily(
                    data_daily[his_daily_start_index:his_daily_end_index, his_daily_row, his_daily_col], his_daily_index,
                    his_daily_path)
            # draw his monthly of a year
            elif df.iat[i, 0] == 'draw6':
                his_monthly_year = int(df.iat[i, 1])
                his_monthly_lon = float(df.iat[i, 2])
                his_monthly_lat = float(df.iat[i, 3])
                his_monthly_name = df.iat[i, 4]
                his_monthly_start_index = 12 * (his_monthly_year - year_start)
                his_monthly_end_index = his_monthly_start_index + 12
                his_monthly_row, his_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, his_monthly_lon, his_monthly_lat)
                his_monthly_index = list(range(1, 13))
                his_monthly_path = path_output + os.sep + '/%s' % his_monthly_name
                if data_monthly is None:
                    data_monthly = np.zeros((nyears * 12, row_nc, col_nc))
                    data_monthly[:, :, :] = preprocess.daily_to_monthly(
                        data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                    for m in range(row_nc):
                        for n in range(col_nc):
                            ts = data_monthly[:, m, n]
                            data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
                draw.draw_his_precipitation_monthly(
                    data_monthly[his_monthly_start_index:his_monthly_end_index, his_monthly_row, his_monthly_col],
                    his_monthly_index, his_monthly_path)
            # space_time change
            elif df.iat[i, 0] == 'draw7':
                sp_time_start = int(df.iat[i, 2])
                sp_time_end = int(df.iat[i, 3])
                sp_time_name = df.iat[i, 4]
                sp_time_path = path_output + os.sep + '/%s' % sp_time_name
                if df.iat[i, 1] == 'monthly':
                    if data_monthly is None:
                        data_monthly = np.zeros((nyears * 12, row_nc, col_nc))
                        data_monthly[:, :, :] = preprocess.daily_to_monthly(
                            data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                        for m in range(row_nc):
                            for n in range(col_nc):
                                ts = data_monthly[:, m, n]
                                data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
                    sp_time_start_index = 12 * (sp_time_start - year_start)
                    sp_time_end_index = 12 * (sp_time_end - year_start + 1)
                    list_monthly_title = []
                    for year in range(sp_time_start, sp_time_end + 1):
                        for month in range(12):
                            title = 'precipitation_monthly_%d_%d' % (year, month + 1)
                            list_monthly_title.append(title)
                    draw.draw_space_time_change(
                        data_monthly[sp_time_start_index:sp_time_end_index], sp_time_path, list_monthly_title)

                elif df.iat[i, 1] == 'daily':
                    sp_time_start_index, sp_time_tmp = preprocess.get_start_end_days(year_start, sp_time_start)
                    sp_time_tmp, sp_time_end_index = preprocess.get_start_end_days(year_start, sp_time_end)
                    list_daily_title = []
                    for year in range(sp_time_start, sp_time_end + 1):
                        sp_time_year_days = preprocess.get_days_year(year)
                        for day in range(sp_time_year_days):
                            title = 'precipitation_daily_%d_%d' % (year, day + 1)
                            list_daily_title.append(title)
                    draw.draw_space_time_change(
                        data_daily[sp_time_start_index:sp_time_end_index], sp_time_path, list_daily_title)
            # draw curve_daily_avg
            elif df.iat[i, 0] == 'draw8':
                curve_daily_start_year = int(df.iat[i, 1])
                curve_daily_end_year = int(df.iat[i, 2])
                curve_daily_lon = float(df.iat[i, 3])
                curve_daily_lat = float(df.iat[i, 4])
                curve_daily_name = df.iat[i, 5]
                curve_daily_start_index, curve_daily_tmp = preprocess.get_start_end_days(year_start, curve_daily_start_year)
                curve_daily_tmp, curve_daily_end_index = preprocess.get_start_end_days(year_start, curve_daily_end_year)
                curve_daily_row, curve_daily_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_daily_lon, curve_daily_lat)
                curve_daily_data = preprocess.get_average_daily(
                    data_daily[curve_daily_start_index:curve_daily_end_index, curve_daily_row, curve_daily_col],
                    curve_daily_end_year-curve_daily_start_year+1, curve_daily_start_year, nodata_value)
                curve_daily_index = list(range(1, 366))
                curve_daily_path = path_output + os.sep + '/%s' % curve_daily_name
                draw.draw_curve_precipitation_daily(curve_daily_data, curve_daily_index, curve_daily_path)
            # draw curve_monthly_avg
            elif df.iat[i, 0] == 'draw9':
                curve_monthly_year_start = int(df.iat[i, 1])
                curve_monthly_year_end = int(df.iat[i, 2])
                curve_monthly_lon = float(df.iat[i, 3])
                curve_monthly_lat = float(df.iat[i, 4])
                curve_monthly_name = df.iat[i, 5]
                curve_monthly_start_index = 12 * (curve_monthly_year_start - year_start)
                curve_monthly_end_index = 12 * (curve_monthly_year_end - year_start) + 12
                curve_monthly_row, curve_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_monthly_lon, curve_monthly_lat)
                curve_monthly_index = list(range(1, 13))
                curve_monthly_path = path_output + os.sep + '/%s' % curve_monthly_name
                if data_monthly is None:
                    data_monthly = preprocess.daily_to_monthly(
                        data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                    for m in range(row_nc):
                        for n in range(col_nc):
                            ts = data_monthly[:, m, n]
                            data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
                curve_monthly_data = preprocess.get_average_monthly(
                    data_monthly[curve_monthly_start_index:curve_monthly_end_index, curve_monthly_row, curve_monthly_col],
                    curve_monthly_year_end-curve_monthly_year_start+1, nodata_value)
                draw.draw_curve_precipitation_monthly(curve_monthly_data, curve_monthly_index, curve_monthly_path)
            # draw his_daily_avg
            elif df.iat[i, 0] == 'draw10':
                his_daily_year_start = int(df.iat[i, 1])
                his_daily_year_end = int(df.iat[i, 2])
                his_daily_lon = float(df.iat[i, 3])
                his_daily_lat = float(df.iat[i, 4])
                his_daily_name = df.iat[i, 5]
                his_daily_start_index, his_daily_tmp = preprocess.get_start_end_days(year_start, his_daily_year_start)
                his_daily_tmp, his_daily_end_index = preprocess.get_start_end_days(year_start, his_daily_year_end)
                his_daily_row, his_daily_col = preprocess.get_point_location(
                    list_lon, list_lat, his_daily_lon, his_daily_lat)
                his_daily_index = list(range(1, 366))
                his_daily_path = path_output + os.sep + '/%s' % his_daily_name
                his_daily_data = preprocess.get_average_daily(
                    data_daily[his_daily_start_index:his_daily_end_index, his_daily_row, his_daily_col],
                    his_daily_year_end-his_daily_year_start+1, his_daily_year_start, nodata_value)
                draw.draw_his_precipitation_daily(his_daily_data, his_daily_index, his_daily_path)
            # draw his_monthly_avg
            elif df.iat[i, 0] == 'draw11':
                his_monthly_year_start = int(df.iat[i, 1])
                his_monthly_year_end = int(df.iat[i, 2])
                his_monthly_lon = float(df.iat[i, 3])
                his_monthly_lat = float(df.iat[i, 4])
                his_monthly_name = df.iat[i, 5]
                his_monthly_start_index = 12 * (his_monthly_year_start - year_start)
                his_monthly_end_index = 12 * (his_monthly_year_end - year_start) + 12
                his_monthly_row, his_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, his_monthly_lon, his_monthly_lat)
                his_monthly_index = list(range(1, 13))
                his_monthly_path = path_output + os.sep + '/%s' % his_monthly_name
                if data_monthly is None:
                    data_monthly = np.zeros((nyears * 12, row_nc, col_nc))
                    data_monthly[:, :, :] = preprocess.daily_to_monthly(
                        data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                    for m in range(row_nc):
                        for n in range(col_nc):
                            ts = data_monthly[:, m, n]
                            data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
                his_monthly_data = preprocess.get_average_monthly(
                    data_monthly[his_monthly_start_index:his_monthly_end_index, his_monthly_row, his_monthly_col],
                    his_monthly_year_end-his_monthly_year_start+1, nodata_value)
                draw.draw_his_precipitation_monthly(his_monthly_data, his_monthly_index, his_monthly_path)
            # draw gini_index_avg
            elif df.iat[i, 0] == 'draw12':
                wgini_threshold = float(df.iat[i, 1])
                gini_year_start = int(df.iat[i, 2])
                gini_year_end = int(df.iat[i, 3])
                gini_lon = float(df.iat[i, 4])
                gini_lat = float(df.iat[i, 5])
                gini_name = df.iat[i, 6]
                gini_path = path_output + os.sep + '/%s' % gini_name
                gini_daily_row, gini_daily_col = preprocess.get_point_location(list_lon, list_lat, gini_lon, gini_lat)
                gini_daily_start_index, gini_day_tmp = preprocess.get_start_end_days(year_start, gini_year_start)
                gini_day_tmp, gini_daily_end_index = preprocess.get_start_end_days(year_start, gini_year_end)
                gini_daily_data = preprocess.get_average_daily(
                    data_daily[gini_daily_start_index:gini_daily_end_index, gini_daily_row, gini_daily_col],
                    gini_year_end - gini_year_start + 1, gini_year_start, nodata_value)
                if gini_daily_data[0] < -1:
                    print("draw12: data invalid!!!")
                    continue
                draw.draw_gini_all(gini_daily_data, wgini_threshold, gini_path)
            else:
                break
        print("Draw picture done!")
    print("Done!!!")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


def process_draw(config_path):
    """
    Plot the corresponding graph from the input csv file
    """
    df = pd.read_csv(config_path, header=None)
    row_csv = 0
    num_files = int(df.iat[row_csv, 1])
    name_var = df.iat[row_csv, 2]
    name_lon = df.iat[row_csv, 3]
    name_lat = df.iat[row_csv, 4]
    
    # Get the list of files
    list_files = []
    for i in range(num_files):
        row_csv += 1
        list_files.append(df.iat[row_csv, 1])
    
    # get basic information of nc
    filename = 'NETCDF:"%s":%s' % (list_files[0], name_var)
    ds = gdal.Open(filename)
    row_nc = ds.RasterYSize
    col_nc = ds.RasterXSize
    geotrans = ds.GetGeoTransform()
    projinfo = ds.GetProjection()
    print('Bacial information of nc file:')
    print(row_nc, col_nc, geotrans, projinfo)
    # get lon and lat list of nc
    ds = xr.open_dataset(list_files[0])
    list_lon = ds[name_lon].values.tolist()
    list_lat = ds[name_lat].values.tolist()
    # get the start year and end year
    row_csv += 1
    year_start = int(df.iat[row_csv, 1])
    row_csv += 1
    year_end = int(df.iat[row_csv, 1])
    # get the nodata_value of the nc file
    row_csv += 1
    nodata_value = int(df.iat[row_csv, 1])
    # time_length
    nyears = year_end - year_start + 1
    
    year_list = range(year_start, year_end + 1)
    # get the total number of study time
    alldays = 0
    for year in year_list:
        alldays += preprocess.get_days_year(year)
    
    data_daily = np.zeros((alldays, row_nc, col_nc))
    
    data_start = 0
    print('read nc file')
    for year in year_list:
        print(year)
        year_index = year - year_start
        ds = Dataset(list_files[year_index])
        data = ds.variables[name_var]
        var_data = np.array(data)
        days_year = preprocess.get_days_year(year)
        data_daily[data_start:data_start + days_year, :, :] = var_data
        data_start += days_year
    
    print('Daily to monthly')
    # data_monthly = np.zeros((nyears * 12, row_nc, col_nc))
    data_monthly = preprocess.daily_to_monthly(data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
    for i in range(row_nc):
        for j in range(col_nc):
            ts = data_monthly[:, i, j]
            data_monthly[:, i, j] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
    print('Daily to monthly done!')
    # File output address
    row_csv += 1
    path_output = df.iat[row_csv, 1]
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    print("Draw picture")
    for i in range(row_csv + 1, df.shape[0]):
        print(df.iat[i, 0])
        if df.iat[i, 0] == 'draw1':
            cor_path1 = df.iat[i, 1]
            cor_path2 = df.iat[i, 2]
            cor_var1_name = df.iat[i, 3]
            cor_var2_name = df.iat[i, 4]
            cor_name = df.iat[i, 5]
            cor_path = path_output + os.sep + '/%s' % cor_name
            cor_dataset1 = gdal.Open(str(cor_path1))
            cor_dataset2 = gdal.Open(str(cor_path2))
            if cor_dataset1 is None or cor_dataset2 is None:
                print("file can not open")
                continue
            cor_width1 = cor_dataset1.RasterXSize
            cor_height1 = cor_dataset1.RasterYSize
            cor_width2 = cor_dataset2.RasterXSize
            cor_height2 = cor_dataset2.RasterYSize
            if cor_width1 != cor_width2 or cor_height1 != cor_height2:
                print("cor_width1 != cor_width2 or cor_height1 != cor_height2")
                continue
            cor_data1 = cor_dataset1.ReadAsArray(0, 0, cor_width1, cor_height1)
            cor_data2 = cor_dataset2.ReadAsArray(0, 0, cor_width2, cor_height2)
            draw.draw_cor_var(cor_data1, cor_data2, cor_var1_name, cor_var2_name, cor_path)
        
        # draw gini
        elif df.iat[i, 0] == 'draw2':
            wgini_threshold = float(df.iat[i, 1])
            gini_year = int(df.iat[i, 2])
            gini_lon = float(df.iat[i, 3])
            gini_lat = float(df.iat[i, 4])
            gini_name = df.iat[i, 5]
            gini_path = path_output + os.sep + '/%s' % gini_name
            gini_row, gini_col = preprocess.get_point_location(list_lon, list_lat, gini_lon, gini_lat)
            gini_day_start_index, gini_day_end_index = preprocess.get_start_end_days(year_start, gini_year)
            draw.draw_gini_all(
                data_daily[gini_day_start_index:gini_day_end_index, gini_row, gini_col], wgini_threshold, gini_path)
        
        # draw curve_daily
        elif df.iat[i, 0] == 'draw3':
            curve_daily_year = int(df.iat[i, 1])
            curve_daily_lon = float(df.iat[i, 2])
            curve_daily_lat = float(df.iat[i, 3])
            curve_daily_name = df.iat[i, 4]
            curve_daily_start_index, curve_daily_end_index = preprocess.get_start_end_days(year_start, curve_daily_year)
            curve_daily_row, curve_daily_col = preprocess.get_point_location(
                list_lon, list_lat, curve_daily_lon, curve_daily_lat)
            curve_daily_index = list(range(1, preprocess.get_days_year(curve_daily_year) + 1))
            curve_daily_path = path_output + os.sep + '/%s' % curve_daily_name
            draw.draw_curve_precipitation_daily(
                data_daily[curve_daily_start_index:curve_daily_end_index, curve_daily_row, curve_daily_col],
                curve_daily_index, curve_daily_path)
        # draw curve_monthly
        elif df.iat[i, 0] == 'draw4':
            curve_monthly_year = int(df.iat[i, 1])
            curve_monthly_lon = float(df.iat[i, 2])
            curve_monthly_lat = float(df.iat[i, 3])
            curve_monthly_name = df.iat[i, 4]
            curve_monthly_start_index = 12 * (curve_monthly_year - year_start)
            curve_monthly_end_index = curve_monthly_start_index + 12
            curve_monthly_row, curve_monthly_col = preprocess.get_point_location(
                list_lon, list_lat, curve_monthly_lon, curve_monthly_lat)
            curve_monthly_index = list(range(1, 13))
            curve_monthly_path = path_output + os.sep + '/%s' % curve_monthly_name
            draw.draw_curve_precipitation_monthly(
                data_monthly[curve_monthly_start_index:curve_monthly_end_index, curve_monthly_row, curve_monthly_col],
                curve_monthly_index, curve_monthly_path)
        # draw his_daily
        elif df.iat[i, 0] == 'draw5':
            his_daily_year = int(df.iat[i, 1])
            his_daily_lon = float(df.iat[i, 2])
            his_daily_lat = float(df.iat[i, 3])
            his_daily_name = df.iat[i, 4]
            his_daily_start_index, his_daily_end_index = preprocess.get_start_end_days(year_start, his_daily_year)
            his_daily_row, his_daily_col = preprocess.get_point_location(list_lon, list_lat, his_daily_lon, his_daily_lat)
            his_daily_index = list(range(1, preprocess.get_days_year(his_daily_year) + 1))
            his_daily_path = path_output + os.sep + '/%s' % his_daily_name
            draw.draw_his_precipitation_daily(
                data_daily[his_daily_start_index:his_daily_end_index, his_daily_row, his_daily_col], his_daily_index,
                his_daily_path)
        # draw his monthly
        elif df.iat[i, 0] == 'draw6':
            his_monthly_year = int(df.iat[i, 1])
            his_monthly_lon = float(df.iat[i, 2])
            his_monthly_lat = float(df.iat[i, 3])
            his_monthly_name = df.iat[i, 4]
            his_monthly_start_index = 12 * (his_monthly_year - year_start)
            his_monthly_end_index = his_monthly_start_index + 12
            his_monthly_row, his_monthly_col = preprocess.get_point_location(
                list_lon, list_lat, his_monthly_lon, his_monthly_lat)
            his_monthly_index = list(range(1, 13))
            his_monthly_path = path_output + os.sep + '/%s' % his_monthly_name
            draw.draw_his_precipitation_monthly(
                data_monthly[his_monthly_start_index:his_monthly_end_index, his_monthly_row, his_monthly_col],
                his_monthly_index, his_monthly_path)
        # space_time change
        elif df.iat[i, 0] == 'draw7':
            sp_time_start = int(df.iat[i, 2])
            sp_time_end = int(df.iat[i, 3])
            sp_time_name = df.iat[i, 4]
            sp_time_path = path_output + os.sep + '/%s' % sp_time_name
            if df.iat[i, 1] == 'monthly':
                sp_time_start_index = 12 * (sp_time_start - year_start)
                sp_time_end_index = 12 * (sp_time_end - year_start + 1)
                list_monthly_title = []
                for year in range(sp_time_start, sp_time_end + 1):
                    for month in range(12):
                        title = 'precipitation_monthly_%d_%d' % (year, month + 1)
                        list_monthly_title.append(title)
                draw.draw_space_time_change(
                    data_monthly[sp_time_start_index:sp_time_end_index], sp_time_path, list_monthly_title)
            elif df.iat[i, 1] == 'daily':
                sp_time_start_index, sp_time_tmp = preprocess.get_start_end_days(year_start, sp_time_start)
                sp_time_tmp, sp_time_end_index = preprocess.get_start_end_days(year_start, sp_time_end)
                list_daily_title = []
                for year in range(sp_time_start, sp_time_end + 1):
                    sp_time_year_days = preprocess.get_days_year(year)
                    for day in range(sp_time_year_days):
                        title = 'precipitation_daily_%d_%d' % (year, day + 1)
                        list_daily_title.append(title)
                draw.draw_space_time_change(
                    data_daily[sp_time_start_index:sp_time_end_index], sp_time_path, list_daily_title)
        # draw curve_daily_avg
        elif df.iat[i, 0] == 'draw8':
            curve_daily_start_year = int(df.iat[i, 1])
            curve_daily_end_year = int(df.iat[i, 2])
            curve_daily_lon = float(df.iat[i, 3])
            curve_daily_lat = float(df.iat[i, 4])
            curve_daily_name = df.iat[i, 5]
            curve_daily_start_index, curve_daily_tmp = preprocess.get_start_end_days(year_start, curve_daily_start_year)
            curve_daily_tmp, curve_daily_end_index = preprocess.get_start_end_days(year_start, curve_daily_end_year)
            curve_daily_row, curve_daily_col = preprocess.get_point_location(
                list_lon, list_lat, curve_daily_lon, curve_daily_lat)
            curve_daily_data = preprocess.get_average_daily(
                data_daily[curve_daily_start_index:curve_daily_end_index, curve_daily_row, curve_daily_col],
                curve_daily_end_year-curve_daily_start_year+1, curve_daily_start_year, nodata_value)
            curve_daily_index = list(range(1, 366))
            curve_daily_path = path_output + os.sep + '/%s' % curve_daily_name
            draw.draw_curve_precipitation_daily(curve_daily_data, curve_daily_index, curve_daily_path)
        # draw curve_monthly_avg
        elif df.iat[i, 0] == 'draw9':
            curve_monthly_year_start = int(df.iat[i, 1])
            curve_monthly_year_end = int(df.iat[i, 2])
            curve_monthly_lon = float(df.iat[i, 3])
            curve_monthly_lat = float(df.iat[i, 4])
            curve_monthly_name = df.iat[i, 5]
            curve_monthly_start_index = 12 * (curve_monthly_year_start - year_start)
            curve_monthly_end_index = 12 * (curve_monthly_year_end - year_start) + 12
            curve_monthly_row, curve_monthly_col = preprocess.get_point_location(
                list_lon, list_lat, curve_monthly_lon, curve_monthly_lat)
            curve_monthly_index = list(range(1, 13))
            curve_monthly_path = path_output + os.sep + '/%s' % curve_monthly_name
            if data_monthly is None:
                data_monthly = preprocess.daily_to_monthly(
                    data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                for m in range(row_nc):
                    for n in range(col_nc):
                        ts = data_monthly[:, m, n]
                        data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
            curve_monthly_data = preprocess.get_average_monthly(
                data_monthly[curve_monthly_start_index:curve_monthly_end_index, curve_monthly_row, curve_monthly_col],
                curve_monthly_year_end-curve_monthly_year_start+1, nodata_value)
            
            print(curve_monthly_data)
            
            draw.draw_curve_precipitation_monthly(curve_monthly_data, curve_monthly_index, curve_monthly_path)
        # draw his_daily_avg
        elif df.iat[i, 0] == 'draw10':
            his_daily_year_start = int(df.iat[i, 1])
            his_daily_year_end = int(df.iat[i, 2])
            his_daily_lon = float(df.iat[i, 3])
            his_daily_lat = float(df.iat[i, 4])
            his_daily_name = df.iat[i, 5]
            his_daily_start_index, his_daily_tmp = preprocess.get_start_end_days(year_start, his_daily_year_start)
            his_daily_tmp, his_daily_end_index = preprocess.get_start_end_days(year_start, his_daily_year_end)
            his_daily_row, his_daily_col = preprocess.get_point_location(
                list_lon, list_lat, his_daily_lon, his_daily_lat)
            his_daily_index = list(range(1, 366))
            his_daily_path = path_output + os.sep + '/%s' % his_daily_name
            his_daily_data = preprocess.get_average_daily(
                data_daily[his_daily_start_index:his_daily_end_index, his_daily_row, his_daily_col],
                his_daily_year_end-his_daily_year_start+1, his_daily_year_start, nodata_value)
            draw.draw_his_precipitation_daily(his_daily_data, his_daily_index, his_daily_path)
        # draw his_monthly_avg
        elif df.iat[i, 0] == 'draw11':
            his_monthly_year_start = int(df.iat[i, 1])
            his_monthly_year_end = int(df.iat[i, 2])
            his_monthly_lon = float(df.iat[i, 3])
            his_monthly_lat = float(df.iat[i, 4])
            his_monthly_name = df.iat[i, 5]
            his_monthly_start_index = 12 * (his_monthly_year_start - year_start)
            his_monthly_end_index = 12 * (his_monthly_year_end - year_start) + 12
            his_monthly_row, his_monthly_col = preprocess.get_point_location(
                list_lon, list_lat, his_monthly_lon, his_monthly_lat)
            his_monthly_index = list(range(1, 13))
            his_monthly_path = path_output + os.sep + '/%s' % his_monthly_name
            if data_monthly is None:
                data_monthly = np.zeros((nyears * 12, row_nc, col_nc))
                data_monthly[:, :, :] = preprocess.daily_to_monthly(
                    data_daily, year_start, nyears, row_nc, col_nc, nodata_value)
                for m in range(row_nc):
                    for n in range(col_nc):
                        ts = data_monthly[:, m, n]
                        data_monthly[:, m, n] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
            his_monthly_data = preprocess.get_average_monthly(
                data_monthly[his_monthly_start_index:his_monthly_end_index, his_monthly_row, his_monthly_col],
                his_monthly_year_end-his_monthly_year_start+1, nodata_value)
            draw.draw_his_precipitation_monthly(his_monthly_data, his_monthly_index, his_monthly_path)
        elif df.iat[i, 0] == 'draw12':
            wgini_threshold = float(df.iat[i, 1])
            gini_year_start = int(df.iat[i, 2])
            gini_year_end = int(df.iat[i, 3])
            gini_lon = float(df.iat[i, 4])
            gini_lat = float(df.iat[i, 5])
            gini_name = df.iat[i, 6]
            gini_path = path_output + os.sep + '/%s' % gini_name
            gini_daily_row, gini_daily_col = preprocess.get_point_location(list_lon, list_lat, gini_lon, gini_lat)
            gini_daily_start_index, gini_day_tmp = preprocess.get_start_end_days(year_start, gini_year_start)
            gini_day_tmp, gini_daily_end_index = preprocess.get_start_end_days(year_start, gini_year_end)
            gini_daily_data = preprocess.get_average_daily(
                data_daily[gini_daily_start_index:gini_daily_end_index, gini_daily_row, gini_daily_col],
                gini_year_end - gini_year_start + 1, gini_year_start, nodata_value)
            
            rain_avg = 0.0
            for k in range(365):
                rain_avg += gini_daily_data[k]
            print(gini_name, rain_avg)
            
            if gini_daily_data[0] < -1:
                print("draw12: data invalid!!!")
                continue
            draw.draw_gini_all(gini_daily_data, wgini_threshold, gini_path)
        else:
            break
    print("Draw picture done!")
    print("Done!!!")


def sub_process(tmp_dir, config_path, row_index, col_nc, value_max_rainfall, row_nc):
    """
    Calculate the metrics of temporal rainfall distribution
    tmp_dir: the output dir of temp files
    configure_path: the path of configuration file
    row_index: current calculation row index
    col_nc: the total col number of nc file
    value_max_rainfall: the value of max precipitation of input data
    row_nc: the total row number of nc file
    """
    print(f'Current calculation row number: {row_index+1} Total row number: {row_nc} ')
    df = pd.read_csv(config_path, header=None)
    row_csv = 0
    product_name = df.iat[row_csv, 0]
    row_csv += 1
    num_files = int(df.iat[row_csv, 1])
    name_var = df.iat[row_csv, 2]
    name_lon = df.iat[row_csv, 3]
    name_lat = df.iat[row_csv, 4]

    # Get the list of files
    list_files = []
    for i in range(num_files):
        row_csv += 1
        list_files.append(df.iat[row_csv, 1])
    # get the start year and end year
    row_csv += 1
    year_start = int(df.iat[row_csv, 1])
    row_csv += 1
    year_end = int(df.iat[row_csv, 1])
    # get the nodata_value of the nc file
    row_csv += 1
    nodata_value = int(df.iat[row_csv, 1])
    # time_length
    nyears = year_end - year_start + 1
    year_list = range(year_start, year_end + 1)
    # get the total number of study time
    alldays = 0
    for year in year_list:
        alldays += preprocess.get_days_year(year)
    data_daily = np.zeros((alldays, 1, col_nc))
    data_start = 0
    for year in year_list:
        year_index = year - year_start
        ds = Dataset(list_files[year_index])
        var_data = np.array(ds.variables[name_var][:, row_index, :])
        days_year = preprocess.get_days_year(year)
        data_daily[data_start:data_start + days_year, 0, :] = var_data
        data_start += days_year
    # operation model 1 pixel by pixel 2 parameter by parameter
    # unuse
    row_csv += 1

    # File output address
    # unuse
    row_csv += 1

    # Whether to calculate rain intensity
    row_csv += 1
    bool_intensity = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate rain intensity by percentile. if not, the result will be the average
    row_csv += 1
    bool_percentile_intensity = bool(int(df.iat[row_csv, 1]))
    if bool_percentile_intensity:
        list_percentile_intensity = df.iat[row_csv, 2].split()
        list_percentile_intensity = list(map(float, list_percentile_intensity))
    else:
        list_percentile_intensity = []

    # Whether to calculate rain times
    row_csv += 1
    bool_times = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate cdds
    row_csv += 1
    bool_cdds = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate cdds by percentile. if not, the result will be the average
    row_csv += 1
    bool_percentile_cdds = bool(int(df.iat[row_csv, 1]))
    if bool_percentile_cdds:
        list_percentile_cdds = df.iat[row_csv, 2].split()
        list_percentile_cdds = list(map(float, list_percentile_cdds))
    else:
        list_percentile_cdds = []
    # Whether to calculate times and cdds by specified thresholds, if not, the threshold will be set to 1
    row_csv += 1
    bool_threshold_times_cdds = bool(int(df.iat[row_csv, 1]))
    if bool_threshold_times_cdds:
        list_threshold_times_cdds = df.iat[row_csv, 2].split()
        list_threshold_times_cdds = list(map(float, list_threshold_times_cdds))
    else:
        list_threshold_times_cdds = np.zeros(1)
        list_threshold_times_cdds[0] = 1

    # whether to calculate ugini index
    row_csv += 1
    bool_ugini = bool(int(df.iat[row_csv, 1]))
    # whether to calculate gini index
    row_csv += 1
    bool_gini = bool(int(df.iat[row_csv, 1]))
    # whether to calculate wgini index
    row_csv += 1
    bool_wgini = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate wgini index by specified threshold(which can spend a lot of time),
    # if not, the threshold will be set to 1
    row_csv += 1
    bool_threshold_wgini = bool(int(df.iat[row_csv, 1]))
    if bool_threshold_wgini:
        list_threshold_wgini = df.iat[row_csv, 2].split()
        list_threshold_wgini = list(map(float, list_threshold_wgini))
    else:
        list_threshold_wgini = np.zeros(1)
        list_threshold_wgini[0] = 1

    # Whether to calculate pci
    row_csv += 1
    bool_pci = bool(int(df.iat[row_csv, 1]))

    # Whether to calculate dsi
    row_csv += 1
    bool_dsi = bool(int(df.iat[row_csv, 1]))

    # Whether to calculate si
    row_csv += 1
    bool_si = bool(int(df.iat[row_csv, 1]))

    # Creating space for variables
    # intensity
    array_intensity = None
    if bool_intensity:
        if bool_percentile_intensity:
            array_intensity = np.zeros((len(list_percentile_intensity), 1, col_nc))
        else:
            array_intensity = np.zeros((1, 1, col_nc))
    # time or cdds
    array_times = None
    array_cdds = None
    if bool_times or bool_cdds:
        array_times = np.zeros((len(list_threshold_times_cdds), 1, col_nc))
        if bool_percentile_cdds:
            array_cdds = np.zeros((len(list_threshold_times_cdds), len(list_percentile_cdds), 1, col_nc))
        else:
            array_cdds = np.zeros((len(list_threshold_times_cdds), 1, 1, col_nc))

    # gini
    array_gini_min = None
    array_gini_max = None
    array_ugini_min = None
    array_ugini_max = None
    array_wgini_min = None
    array_wgini_max = None
    if bool_gini:
        array_gini_min = np.zeros((nyears, 1, col_nc))
        array_gini_max = np.zeros((nyears, 1, col_nc))
    if bool_ugini:
        array_ugini_min = np.zeros((nyears, 1, col_nc))
        array_ugini_max = np.zeros((nyears, 1, col_nc))
    if bool_wgini:
        array_wgini_min = np.zeros((len(list_threshold_wgini), nyears, 1, col_nc))
        array_wgini_max = np.zeros((len(list_threshold_wgini), nyears, 1, col_nc))

    # pci
    array_pci = None
    array_pci_avg = None
    if bool_pci:
        array_pci = np.zeros((nyears, 1, col_nc))
        array_pci_avg = np.zeros((1, col_nc))
    # dsi
    array_dsi = None
    if bool_dsi:
        array_dsi = np.zeros((1, col_nc))
    # si
    array_si = None
    array_si_avg = None
    if bool_si:
        array_si = np.zeros((nyears, 1, col_nc))
        array_si_avg = np.zeros((1, col_nc))

    # monthly precipitation
    data_monthly = None
    data_monthly_avg = None

    if bool_pci or bool_dsi or bool_pci:
        data_monthly = np.zeros((nyears * 12, 1, col_nc))
        data_monthly_avg = np.zeros((12, 1, col_nc))
        data_monthly_path = tmp_dir + os.sep + "monthly_row" + str(row_index) + '.npy'
        data_monthly_avg_path = tmp_dir + os.sep + "monthly_avg_row" + str(row_index) + '.npy'
        data_monthly[:, 0, :] = np.load(data_monthly_path)
        data_monthly_avg[:, 0, :] = np.load(data_monthly_avg_path)

    # calculate
    # list/array nodata
    len_intensity = 1
    if bool_percentile_intensity:
        len_intensity = len(list_percentile_intensity)
    list_nodata_intensity = np.zeros(len_intensity)
    for i in range(len_intensity):
        list_nodata_intensity[i] = nodata_value

    len_times = 1
    if bool_threshold_times_cdds:
        len_times = len(list_threshold_times_cdds)
    list_nodata_times = np.zeros(len_times)
    for i in range(len_times):
        list_nodata_times = nodata_value

    len_cdds = 1
    if bool_percentile_cdds:
        len_cdds = len(list_percentile_cdds)
    array_nodata_cdds = np.zeros((len_times, len_cdds))
    for i in range(len_cdds):
        array_nodata_cdds[:, i] = list_nodata_times

    list_nodata_gini = np.zeros(nyears)
    for i in range(nyears):
        list_nodata_gini[i] = nodata_value
    len_wgini = 1
    if bool_wgini:
        len_wgini = len(list_threshold_wgini)
    array_nodata_wgini = np.zeros((len_wgini, nyears))
    for i in range(len_wgini):
        array_nodata_wgini[i, :] = list_nodata_gini
    # row_nc
    for j in range(col_nc):
        ts_daily = data_daily[:, 0, j]
        if ts_daily[0] < -1 or str(ts_daily[0]) == '--':
            if bool_intensity:
                array_intensity[:, 0, j] = list_nodata_intensity
            if bool_times or bool_cdds:
                array_cdds[:, :, 0, j] = array_nodata_cdds
                array_times[:, 0, j] = list_nodata_times
            if bool_ugini:
                array_ugini_max[:, 0, j] = list_nodata_gini
                array_ugini_min[:, 0, j] = list_nodata_gini
            if bool_gini:
                array_gini_max[:, 0, j] = list_nodata_gini
                array_gini_min[:, 0, j] = list_nodata_gini
            if bool_wgini:
                array_wgini_max[:, :, 0, j] = array_nodata_wgini
                array_wgini_min[:, :, 0, j] = array_nodata_wgini

        else:
            if bool_intensity:
                array_intensity[:, 0, j] = daily.stat_intensity(
                    ts_daily=ts_daily,
                    percentile=list_percentile_intensity,
                    nodata_value=nodata_value)
            if bool_cdds or bool_times:
                for threshold in range(len(list_threshold_times_cdds)):
                    array_times[threshold, 0, j], array_cdds[threshold, :, 0, j] = daily.stat_interval(
                        ts_daily=ts_daily,
                        nyears=nyears,
                        percentile=list_percentile_cdds,
                        threshold=list_threshold_times_cdds[threshold],
                        nodata_value=nodata_value)
            if bool_ugini:
                array_ugini_min[:, 0, j], array_ugini_max[:, 0, j] = daily.stat_ugini(
                    ts_daily=ts_daily,
                    nyears=nyears,
                    year_start=year_start,
                    ts_long=alldays,
                    nodata_value=nodata_value)
            if bool_gini:
                array_gini_min[:, 0, j], array_gini_max[:, 0, j] = daily.stat_gini(
                    ts_daily=ts_daily,
                    nyears=nyears,
                    year_start=year_start,
                    ts_long=alldays,
                    nodata_value=nodata_value)
            if bool_wgini:
                for threshold in range(len(list_threshold_wgini)):
                    array_wgini_min[threshold, :, 0, j], array_wgini_max[threshold, :, 0, j] = daily.stat_wgini(
                        ts_daily=ts_daily,
                        nyears=nyears,
                        year_start=year_start,
                        ts_long=alldays,
                        threshold=list_threshold_wgini[threshold],
                        nodata_value=nodata_value)

        if data_monthly is not None:
            ts_monthly = data_monthly[:, 0, j]
            ts_monthly_avg = data_monthly_avg[:, 0, j]
            if ts_monthly[0] == nodata_value:
                if bool_pci:
                    array_pci[:, 0, j] = list_nodata_gini
                    array_pci_avg[0, j] = nodata_value
                if bool_dsi:
                    array_dsi[0, j] = nodata_value
                if bool_si:
                    array_si[:, 0, j] = list_nodata_gini
                    array_si_avg[0, j] = nodata_value
            else:
                if bool_pci:
                    array_pci[:, 0, j] = monthly.stat_pci(
                        ts_monthly=ts_monthly,
                        nyears=nyears,
                        nodata_value=nodata_value)
                    array_pci_avg[0, j] = monthly.stat_pci(
                        ts_monthly=ts_monthly_avg,
                        nyears=1,
                        nodata_value=nodata_value)
                if bool_dsi:
                    array_dsi[0, j] = monthly.stat_dsi(
                        ts_monthly=ts_monthly,
                        nyears=nyears,
                        value_max_rainfall=value_max_rainfall,
                        nodata_value=nodata_value)
                if bool_si:
                    array_si[:, 0, j] = monthly.stat_si(
                        ts_monthly=ts_monthly,
                        nyears=nyears,
                        nodata_value=nodata_value)
                    array_si_avg[0, j] = monthly.stat_si(
                        ts_monthly=ts_monthly_avg,
                        nyears=1,
                        nodata_value=nodata_value
                    )
    # output calculate results as temp file
    if bool_intensity:
        if bool_percentile_intensity:
            for i in range(len(list_percentile_intensity)):
                tmp_output = (tmp_dir + os.sep + 'intensity_precentile_' + str(list_percentile_intensity[i]) +
                              '_' + str(row_index) + '.npy')
                np.save(tmp_output, array_intensity[i])
        else:
            tmp_output = tmp_dir + os.sep + 'intensity_mean_' + str(row_index) + '.npy'
            np.save(tmp_output, array_intensity[0])
    if bool_times:
        for i in range(len(list_threshold_times_cdds)):
            tmp_output = (tmp_dir + os.sep + 'frequency_mean_' + str(list_threshold_times_cdds[i]) + '_mm_' +
                          str(row_index) + '.npy')
            np.save(tmp_output, array_times[i])
    if bool_cdds:
        for i in range(len(list_threshold_times_cdds)):
            if bool_percentile_cdds:
                for j in range(len(list_percentile_cdds)):
                    tmp_output = (tmp_dir + os.sep + 'cdd_' + str(list_threshold_times_cdds[i]) + '_precentile_' +
                                  str(list_percentile_cdds[j]) + '_' + str(row_index) + '.npy')
                    np.save(tmp_output, array_cdds[i, j])
            else:
                tmp_output = (tmp_dir + os.sep + 'cdd_' + str(list_threshold_times_cdds[i]) + '_' + str(row_index) +
                              '.npy')
                np.save(tmp_output, array_cdds[i, 0])
    if bool_ugini:
        array_ugini_min_avg = preprocess.get_average_variable(array_ugini_min, nyears, 1, col_nc, nodata_value)
        array_ugini_max_avg = preprocess.get_average_variable(array_ugini_max, nyears, 1, col_nc, nodata_value)
        tmp_output = tmp_dir + os.sep + 'ugini_min_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_ugini_min_avg)
        tmp_output = tmp_dir + os.sep + 'ugini_max_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_ugini_max_avg)
        for i in range(nyears):
            tmp_output = tmp_dir + os.sep + 'ugini_min_year_' + str(i+year_start) + '_' + str(row_index) + '.npy'
            np.save(tmp_output, array_ugini_min[i])
            tmp_output = tmp_dir + os.sep + 'ugini_max_year_' + str(i+year_start) + '_' + str(row_index) + '.npy'
            np.save(tmp_output, array_ugini_max[i])
    if bool_gini:
        array_gini_min_avg = preprocess.get_average_variable(array_gini_min, nyears, 1, col_nc, nodata_value)
        array_gini_max_avg = preprocess.get_average_variable(array_gini_max, nyears, 1, col_nc, nodata_value)
        tmp_output = tmp_dir + os.sep + 'gini_min_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_gini_min_avg)
        tmp_output = tmp_dir + os.sep + 'gini_max_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_gini_max_avg)
        for i in range(nyears):
            tmp_output = tmp_dir + os.sep + 'gini_min_year_' + str(i+year_start) + '_' + str(row_index) + '.npy'
            np.save(tmp_output, array_gini_min[i])
            tmp_output = tmp_dir + os.sep + 'gini_max_year_' + str(i+year_start) + '_' + str(row_index) + '.npy'
            np.save(tmp_output, array_gini_max[i])
    if bool_wgini:
        for i in range(len(list_threshold_wgini)):
            array_wgini_min_avg = preprocess.get_average_variable(array_wgini_min[i], nyears, 1, col_nc, nodata_value)
            array_wgini_max_avg = preprocess.get_average_variable(array_wgini_max[i], nyears, 1, col_nc, nodata_value)
            tmp_output = (tmp_dir + os.sep + 'wgini_min_avg_' + str(i+year_start) + '_threshold_' +
                          str(list_threshold_wgini[i]) + '_' + str(row_index) + '.npy')
            np.save(tmp_output, array_wgini_min_avg)
            tmp_output = (tmp_dir + os.sep + 'wgini_max_avg_' + str(i + year_start) + '_threshold_' +
                          str(list_threshold_wgini[i]) + '_' + str(row_index) + '.npy')
            np.save(tmp_output, array_wgini_max_avg)
            for j in range(nyears):
                tmp_output = (tmp_dir + os.sep + 'wgini_min_year_' + str(j+year_start) + '_threshold_' +
                              str(list_threshold_wgini[i]) + '_' + str(row_index) + '.npy')
                np.save(tmp_output, array_wgini_min[i, j])
                tmp_output = (tmp_dir + os.sep + 'wgini_max_year_' + str(j + year_start) + '_threshold_' +
                              str(list_threshold_wgini[i]) + '_' + str(row_index) + '.npy')
                np.save(tmp_output, array_wgini_max[i, j])
    if bool_pci:
        tmp_output = tmp_dir + os.sep + 'pci_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_pci_avg)
        for i in range(nyears):
            tmp_output = tmp_dir + os.sep + 'pci_year_' + str(i + year_start) + '_' + str(row_index) + '.npy'
            np.save(tmp_output, array_pci[i])
    if bool_si:
        tmp_output = tmp_dir + os.sep + 'si_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_si_avg)
        for i in range(nyears):
            tmp_output = tmp_dir + os.sep + 'si_year_' + str(i + year_start) + '_' + str(row_index) + '.npy'
            np.save(tmp_output, array_si[i])
    if bool_dsi:
        tmp_output = tmp_dir + os.sep + 'dsi_avg_' + str(row_index) + '.npy'
        np.save(tmp_output, array_dsi)


def process_parallel(config_path, num_processes, bool_draw):
    """
    Determine the parameters in the csv file to calculate the corresponding indicators and
    output them, as well as draw pictures
    congigure_path: the path of configuration file
    num_threads: the number of thread to calculate
    """
    start_time = time.time()
    # Assign a thread to the main thread
    num_processes -= 1
    df = pd.read_csv(config_path, header=None)
    row_csv = 0
    product_name = df.iat[row_csv, 0]
    row_csv += 1
    num_files = int(df.iat[row_csv, 1])
    name_var = df.iat[row_csv, 2]
    name_lon = df.iat[row_csv, 3]
    name_lat = df.iat[row_csv, 4]

    # Get the list of files
    list_files = []
    for i in range(num_files):
        row_csv += 1
        list_files.append(df.iat[row_csv, 1])

    # get basic information of nc
    filename = 'NETCDF:"%s":%s' % (list_files[0], name_var)
    ds = gdal.Open(filename)
    row_nc = ds.RasterYSize
    col_nc = ds.RasterXSize
    geotrans = ds.GetGeoTransform()
    projinfo = ds.GetProjection()
    if projinfo == '':
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        projinfo = outRasterSRS.ExportToWkt()
    print('Bacial information of nc file:')
    print(row_nc, col_nc, geotrans, projinfo)
    # get lon and lat list of nc
    ds = xr.open_dataset(list_files[0])
    list_lon = ds[name_lon].values.tolist()
    list_lat = ds[name_lat].values.tolist()

    # get the start year and end year
    row_csv += 1
    year_start = int(df.iat[row_csv, 1])
    row_csv += 1
    year_end = int(df.iat[row_csv, 1])
    # get the nodata_value of the nc file
    row_csv += 1
    nodata_value = int(df.iat[row_csv, 1])
    # time_length
    nyears = year_end - year_start + 1
    year_list = range(year_start, year_end + 1)
    # operation model 1 pixel by pixel 2 parameter by parameter
    row_csv += 1
    model_process = int(df.iat[row_csv, 1])

    # File output address
    row_csv += 1
    path_output = df.iat[row_csv, 1]
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    tmp_dir = path_output + os.sep + "tmp"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # Whether to calculate rain intensity
    row_csv += 1
    bool_intensity = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate rain intensity by percentile. if not, the result will be the average
    row_csv += 1
    bool_percentile_intensity = bool(int(df.iat[row_csv, 1]))
    if bool_percentile_intensity:
        list_percentile_intensity = df.iat[row_csv, 2].split()
        list_percentile_intensity = list(map(float, list_percentile_intensity))
    else:
        list_percentile_intensity = []

    # Whether to calculate rain times
    row_csv += 1
    bool_times = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate cdds
    row_csv += 1
    bool_cdds = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate cdds by percentile. if not, the result will be the average
    row_csv += 1
    bool_percentile_cdds = bool(int(df.iat[row_csv, 1]))
    if bool_percentile_cdds:
        list_percentile_cdds = df.iat[row_csv, 2].split()
        list_percentile_cdds = list(map(float, list_percentile_cdds))
    else:
        list_percentile_cdds = []
    # Whether to calculate times and cdds by specified thresholds, if not, the threshold will be set to 1
    row_csv += 1
    bool_threshold_times_cdds = bool(int(df.iat[row_csv, 1]))
    if bool_threshold_times_cdds:
        list_threshold_times_cdds = df.iat[row_csv, 2].split()
        list_threshold_times_cdds = list(map(float, list_threshold_times_cdds))
    else:
        list_threshold_times_cdds = np.zeros(1)
        list_threshold_times_cdds[0] = 1

    # whether to calculate ugini index
    row_csv += 1
    bool_ugini = bool(int(df.iat[row_csv, 1]))
    # whether to calculate gini index
    row_csv += 1
    bool_gini = bool(int(df.iat[row_csv, 1]))
    # whether to calculate wgini index
    row_csv += 1
    bool_wgini = bool(int(df.iat[row_csv, 1]))
    # Whether to calculate wgini index by specified threshold(which can spend a lot of time),
    # if not, the threshold will be set to 1
    row_csv += 1
    bool_threshold_wgini = bool(int(df.iat[row_csv, 1]))
    if bool_threshold_wgini:
        list_threshold_wgini = df.iat[row_csv, 2].split()
        list_threshold_wgini = list(map(float, list_threshold_wgini))
    else:
        list_threshold_wgini = np.zeros(1)
        list_threshold_wgini[0] = 1

    # Whether to calculate pci
    row_csv += 1
    bool_pci = bool(int(df.iat[row_csv, 1]))

    # Whether to calculate dsi
    row_csv += 1
    bool_dsi = bool(int(df.iat[row_csv, 1]))

    # Whether to calculate si
    row_csv += 1
    bool_si = bool(int(df.iat[row_csv, 1]))

    # Creating space for variables
    # intensity
    array_intensity = None
    if bool_intensity:
        if bool_percentile_intensity:
            array_intensity = np.zeros((len(list_percentile_intensity), row_nc, col_nc))
        else:
            array_intensity = np.zeros((1, row_nc, col_nc))
    # time or cdds
    array_times = None
    array_cdds = None
    if bool_times or bool_cdds:
        array_times = np.zeros((len(list_threshold_times_cdds), row_nc, col_nc))
        if bool_percentile_cdds:
            array_cdds = np.zeros((len(list_threshold_times_cdds), len(list_percentile_cdds), row_nc, col_nc))
        else:
            array_cdds = np.zeros((len(list_threshold_times_cdds), 1, row_nc, col_nc))

    # gini
    array_gini_min = None
    array_gini_max = None
    array_ugini_min = None
    array_ugini_max = None
    array_wgini_min = None
    array_wgini_max = None
    if bool_gini:
        array_gini_min = np.zeros((nyears, row_nc, col_nc))
        array_gini_max = np.zeros((nyears, row_nc, col_nc))
    if bool_ugini:
        array_ugini_min = np.zeros((nyears, row_nc, col_nc))
        array_ugini_max = np.zeros((nyears, row_nc, col_nc))
    if bool_wgini:
        array_wgini_min = np.zeros((len(list_threshold_wgini), nyears, row_nc, col_nc))
        array_wgini_max = np.zeros((len(list_threshold_wgini), nyears, row_nc, col_nc))

    # pci
    array_pci = None
    array_pci_avg = None
    if bool_pci:
        array_pci = np.zeros((nyears, row_nc, col_nc))
        array_pci_avg = np.zeros((row_nc, col_nc))
    # dsi
    array_dsi = None
    if bool_dsi:
        array_dsi = np.zeros((row_nc, col_nc))
    # si
    array_si = None
    array_si_avg = None
    if bool_si:
        array_si = np.zeros((nyears, row_nc, col_nc))
        array_si_avg = np.zeros((row_nc, col_nc))

    # calculate monthly precipitation
    # num_cores = 10
    pool = mp.Pool(num_processes)
    print('Daily to monthly')
    for year in year_list:
        # print(year)
        year_index = year - year_start
        tmp_daily_path = list_files[year_index]
        tmp_output = tmp_dir + os.sep + "monthly_" + str(year) + '.npy'
        pool.apply_async(preprocess.daily_to_monthly_oneyear_parallel, args=(tmp_daily_path, name_var, tmp_output,
                                                                             year, row_nc, col_nc, nodata_value))
    pool.close()
    pool.join()
    print('Daily to monthly done!')
    data_monthly = np.zeros((12*nyears, row_nc, col_nc))
    for year in year_list:
        year_index = year - year_start
        tmp_monthly_data = tmp_dir + os.sep + "monthly_" + str(year) + '.npy'
        data_monthly[year_index*12:year_index*12+12, :, :] = np.load(tmp_monthly_data)
    # clear invalid data
    for i in range(row_nc):
        for j in range(col_nc):
            ts = data_monthly[:, i, j]
            data_monthly[:, i, j] = preprocess.clear_invalid_data(ts, nyears, nodata_value)
    data_monthly_avg = preprocess.get_average_monthly_raster(data_monthly, nyears, row_nc, col_nc, nodata_value)
    # maximum mean annual rainfall in the dataset
    value_max_rainfall = 0.0
    if bool_dsi:
        value_max_rainfall = preprocess.get_max_mean_annual_rainfall_fun2(data_monthly_avg, row_nc, col_nc, nodata_value)
    # output monthly data as row
    print('Output monthly data by row as temp files')
    for i in range(row_nc):
        data_tmp = data_monthly[:, i, :]
        data_tmp_path = tmp_dir + os.sep + "monthly_row" + str(i) + '.npy'
        data_avg_tmp = data_monthly_avg[:, i, :]
        data_avg_tmp_path = tmp_dir + os.sep + "monthly_avg_row" + str(i) + '.npy'
        np.save(data_tmp_path, data_tmp)
        np.save(data_avg_tmp_path, data_avg_tmp)
    print('Output monthly data done!')
    # get the total number of study time
    alldays = 0
    for year in year_list:
        alldays += preprocess.get_days_year(year)
    print("Calculate the metrics of temporal rainfall distribution")
    # calculate
    # num_cores = 10
    pool = mp.Pool(num_processes)
    for i in range(row_nc):
        pool.apply_async(sub_process, args=(tmp_dir, config_path, i, col_nc, value_max_rainfall, row_nc))
    pool.close()
    pool.join()
    print("Calculate the metrics of temporal rainfall distribution done!")
    print("Output the metrics of temporal rainfall distribution as tif files")
    # output tiff file by reading temp file
    if bool_intensity:
        if bool_percentile_intensity:
            for i in range(len(list_percentile_intensity)):
                for j in range(row_nc):
                    tmp_output = (tmp_dir + os.sep + 'intensity_precentile_' + str(list_percentile_intensity[i]) +
                                  '_' + str(j) + '.npy')
                    array_intensity[i, j, :] = np.load(tmp_output)
                outputfile_amount = path_output + os.sep + '%s_intensity_%dpercentile_%04dto%04d.tif' % (
                    product_name, list_percentile_intensity[i], year_start, year_end)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_intensity[i])
        else:
            for j in range(row_nc):
                tmp_output = tmp_dir + os.sep + 'intensity_mean_' + str(j) + '.npy'
                array_intensity[0, j, :] = np.load(tmp_output)
            outputfile_amount = path_output + os.sep + '%s_intensity_mean_%04dto%04d.tif' % (
                product_name, year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_intensity[0])
    if bool_times:
        for i in range(len(list_threshold_times_cdds)):
            for j in range(row_nc):
                tmp_output = (tmp_dir + os.sep + 'frequency_mean_' + str(list_threshold_times_cdds[i]) + '_mm_' +
                              str(j) + '.npy')
                array_times[i, j, :] = np.load(tmp_output)
            outputfile_amount = path_output + os.sep + '%s_frequency_%dmm_mean_%04dto%04d.tif' % (
                product_name, list_threshold_times_cdds[i], year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_times[i])
    if bool_cdds:
       for i in range(len(list_threshold_times_cdds)):
           if bool_percentile_cdds:
               for j in range(len(list_percentile_cdds)):
                   for k in range(row_nc):
                       tmp_output = (tmp_dir + os.sep + 'cdd_' + str(list_threshold_times_cdds[i]) + '_precentile_' +
                                     str(list_percentile_cdds[j]) + '_' + str(k) + '.npy')
                       array_cdds[i, j, k, :] = np.load(tmp_output)
                   outputfile_amount = path_output + '/%s_cdd_%dmm_%dpercentile_%04dto%04d.tif' % (
                       product_name, list_threshold_times_cdds[i], list_percentile_cdds[j], year_start, year_end)
                   draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_cdds[i, j])
           else:
               for j in range(row_nc):
                   tmp_output = (tmp_dir + os.sep + 'cdd_' + str(list_threshold_times_cdds[i]) + '_' + str(j) +
                                 '.npy')
                   array_cdds[i, 0, j, :] = np.load(tmp_output)
                   outputfile_amount = path_output + '/%s_cdd_%dmm_mean_%04dto%04d.tif' % (
                       product_name, list_threshold_times_cdds[i], year_start, year_end)
                   draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value,
                                   array_cdds[i, 0])
    if bool_ugini:
        array_ugini_min_avg = np.zeros((row_nc, col_nc))
        array_ugini_max_avg = np.zeros((row_nc, col_nc))
        for i in range(row_nc):
            tmp_output = tmp_dir + os.sep + 'ugini_min_avg_' + str(i) + '.npy'
            array_ugini_min_avg[i, :] = np.load(tmp_output)
            tmp_output = tmp_dir + os.sep + 'ugini_max_avg_' + str(i) + '.npy'
            array_ugini_max_avg[i, :] = np.load(tmp_output)
        outputfile_amount = path_output + '/%s_ugini_min_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_min_avg)
        outputfile_amount = path_output + '/%s_ugini_max_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_max_avg)
        for i in range(nyears):
            for j in range(row_nc):
                tmp_output = tmp_dir + os.sep + 'ugini_min_year_' + str(i + year_start) + '_' + str(j) + '.npy'
                array_ugini_min[i, j, :] = np.load(tmp_output)
                tmp_output = tmp_dir + os.sep + 'ugini_max_year_' + str(i + year_start) + '_' + str(j) + '.npy'
                array_ugini_max[i, j, :] = np.load(tmp_output)
            outputfile_amount = path_output + '/%s_ugini_min_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_min[i])
            outputfile_amount = path_output + '/%s_ugini_max_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_ugini_max[i])
    if bool_gini:
        array_gini_min_avg = np.zeros((row_nc, col_nc))
        array_gini_max_avg = np.zeros((row_nc, col_nc))
        for i in range(row_nc):
            tmp_output = tmp_dir + os.sep + 'gini_min_avg_' + str(i) + '.npy'
            array_gini_min_avg[i, :] = np.load(tmp_output)
            tmp_output = tmp_dir + os.sep + 'gini_max_avg_' + str(i) + '.npy'
            array_gini_max_avg[i, :] = np.load(tmp_output)
        outputfile_amount = path_output + '/%s_gini_min_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_min_avg)
        outputfile_amount = path_output + '/%s_gini_max_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_max_avg)
        for i in range(nyears):
            for j in range(row_nc):
                tmp_output = tmp_dir + os.sep + 'gini_min_year_' + str(i + year_start) + '_' + str(j) + '.npy'
                array_gini_min[i, j, :] = np.load(tmp_output)
                tmp_output = tmp_dir + os.sep + 'gini_max_year_' + str(i + year_start) + '_' + str(j) + '.npy'
                array_gini_max[i, j, :] = np.load(tmp_output)
            outputfile_amount = path_output + '/%s_gini_min_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_min[i])
            outputfile_amount = path_output + '/%s_gini_max_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_gini_max[i])
    if bool_wgini:
        for i in range(len(list_threshold_wgini)):
            array_wgini_min_avg = np.zeros((row_nc, col_nc))
            array_wgini_max_avg = np.zeros((row_nc, col_nc))
            for j in range(row_nc):
                tmp_output = (tmp_dir + os.sep + 'wgini_min_avg_' + str(i + year_start) + '_threshold_' +
                              str(list_threshold_wgini[i]) + '_' + str(j) + '.npy')
                array_wgini_min_avg[j, ] = np.load(tmp_output)
                tmp_output = (tmp_dir + os.sep + 'wgini_max_avg_' + str(i + year_start) + '_threshold_' +
                              str(list_threshold_wgini[i]) + '_' + str(j) + '.npy')
                array_wgini_max_avg[j,] = np.load(tmp_output)
            outputfile_amount = path_output + '/%s_wgini_min_%dmm_from%04dto%04d_avg.tif' % (
                product_name, list_threshold_wgini[i], year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_wgini_min_avg)
            outputfile_amount = path_output + '/%s_wgini_max_%dmm_from%04dto%04d_avg.tif' % (
                product_name, list_threshold_wgini[i], year_start, year_end)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_wgini_max_avg)
            for j in range(nyears):
                for k in range(row_nc):
                    tmp_output = (tmp_dir + os.sep + 'wgini_min_year_' + str(j + year_start) + '_threshold_' +
                                  str(list_threshold_wgini[i]) + '_' + str(k) + '.npy')
                    array_wgini_min[i, j, k, :] = np.load(tmp_output)
                    tmp_output = (tmp_dir + os.sep + 'wgini_max_year_' + str(j + year_start) + '_threshold_' +
                                  str(list_threshold_wgini[i]) + '_' + str(k) + '.npy')
                    array_wgini_max[i, j, k, :] = np.load(tmp_output)
                outputfile_amount = path_output + '/%s_wgini_min_%dmm_%04d.tif' % (
                    product_name, list_threshold_wgini[i], j + year_start)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value,
                                array_wgini_min[i, j])
                outputfile_amount = path_output + '/%s_wgini_max_%dmm_%04d.tif' % (
                    product_name, list_threshold_wgini[i], j + year_start)
                draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value,
                                array_wgini_max[i, j])
    if bool_pci:
        for i in range(row_nc):
            tmp_output = tmp_dir + os.sep + 'pci_avg_' + str(i) + '.npy'
            array_pci_avg[i, :] = np.load(tmp_output)
        outputfile_amount = path_output + '/%s_pci_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_pci_avg)
        for i in range(nyears):
            for j in range(row_nc):
                tmp_output = tmp_dir + os.sep + 'pci_year_' + str(i + year_start) + '_' + str(j) + '.npy'
                array_pci[i, j, :] = np.load(tmp_output)
            outputfile_amount = path_output + '/%s_pci_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_pci[i])
    if bool_si:
        for i in range(row_nc):
            tmp_output = tmp_dir + os.sep + 'si_avg_' + str(i) + '.npy'
            array_si_avg[i, :] = np.load(tmp_output)
        outputfile_amount = path_output + '/%s_si_from%04dto%04d_avg.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_si_avg)
        for i in range(nyears):
            for j in range(row_nc):
                tmp_output = tmp_dir + os.sep + 'si_year_' + str(i + year_start) + '_' + str(j) + '.npy'
                array_si[i, j, :] = np.load(tmp_output)
            outputfile_amount = path_output + '/%s_si_%04d.tif' % (product_name, i + year_start)
            draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_si[i])
    if bool_dsi:
        for i in range(row_nc):
            tmp_output = tmp_dir + os.sep + 'dsi_avg_' + str(i) + '.npy'
            array_dsi[i, :] = np.load(tmp_output)
        outputfile_amount = path_output + '/%s_dsi_avg_%04dto%04d.tif' % (product_name, year_start, year_end)
        draw.write_tiff(outputfile_amount, row_nc, col_nc, geotrans, projinfo, nodata_value, array_dsi)
    print("Output the metrics of temporal rainfall distribution as tif files done!")
    # draw_picture
    if bool_draw:
        print("Draw picture")
        data_daily = np.zeros((alldays, row_nc, col_nc))
        data_start = 0
        for year in year_list:
            year_index = year - year_start
            ds = Dataset(list_files[year_index])
            data = ds.variables[name_var]
            var_data = np.array(data)
            days_year = preprocess.get_days_year(year)
            data_daily[data_start:data_start + days_year, :, :] = var_data
            data_start += days_year

        for i in range(row_csv + 1, df.shape[0]):
            print(df.iat[i, 0])
            # draw cor_var
            if df.iat[i, 0] == 'draw1':
                cor_var1 = df.iat[i, 1]
                cor_var2 = df.iat[i, 2]
                cor_var1_name = df.iat[i, 3]
                cor_var2_name = df.iat[i, 4]
                cor_year = int(df.iat[i, 5])
                cor_name = df.iat[i, 6]
                cor_year_index = cor_year - year_start
                cor_path = path_output + os.sep + '/%s' % cor_name
                if eval(cor_var1) is None or eval(cor_var2) is None:
                    print("draw1 error: do not calculate the input var")
                    continue
                if cor_var1 == 'array_dsi':
                    cor_var1_data = eval(cor_var1)
                else:
                    cor_var1_data = eval(cor_var1)[cor_year_index]
                if cor_var2 == 'array_dsi':
                    cor_var2_data = eval(cor_var2)
                else:
                    cor_var2_data = eval(cor_var2)[cor_year_index]
                draw.draw_cor_var(cor_var1_data, cor_var2_data, cor_var1_name, cor_var2_name, cor_path)
            # draw gini_index of a year
            elif df.iat[i, 0] == 'draw2':
                wgini_threshold = float(df.iat[i, 1])
                gini_year = int(df.iat[i, 2])
                gini_lon = float(df.iat[i, 3])
                gini_lat = float(df.iat[i, 4])
                gini_name = df.iat[i, 5]
                gini_path = path_output + os.sep + '/%s' % gini_name
                gini_row, gini_col = preprocess.get_point_location(list_lon, list_lat, gini_lon, gini_lat)
                gini_day_start_index, gini_day_end_index = preprocess.get_start_end_days(year_start, gini_year)
                if data_daily[gini_day_start_index, gini_row, gini_col] < -1:
                    print("gini: data invalid!!!")
                    continue
                draw.draw_gini_all(
                    data_daily[gini_day_start_index:gini_day_end_index, gini_row, gini_col], wgini_threshold, gini_path)
            # draw curve_daily of a year
            elif df.iat[i, 0] == 'draw3':
                curve_daily_year = int(df.iat[i, 1])
                curve_daily_lon = float(df.iat[i, 2])
                curve_daily_lat = float(df.iat[i, 3])
                curve_daily_name = df.iat[i, 4]
                curve_daily_start_index, curve_daily_end_index = preprocess.get_start_end_days(year_start, curve_daily_year)
                curve_daily_row, curve_daily_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_daily_lon, curve_daily_lat)
                curve_daily_index = list(range(1, preprocess.get_days_year(curve_daily_year) + 1))
                curve_daily_path = path_output + os.sep + '/%s' % curve_daily_name
                draw.draw_curve_precipitation_daily(
                    data_daily[curve_daily_start_index:curve_daily_end_index, curve_daily_row, curve_daily_col],
                    curve_daily_index, curve_daily_path)
            # draw curve_monthly of a year
            elif df.iat[i, 0] == 'draw4':
                curve_monthly_year = int(df.iat[i, 1])
                curve_monthly_lon = float(df.iat[i, 2])
                curve_monthly_lat = float(df.iat[i, 3])
                curve_monthly_name = df.iat[i, 4]
                curve_monthly_start_index = 12 * (curve_monthly_year - year_start)
                curve_monthly_end_index = curve_monthly_start_index + 12
                curve_monthly_row, curve_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_monthly_lon, curve_monthly_lat)
                curve_monthly_index = list(range(1, 13))
                curve_monthly_path = path_output + os.sep + '/%s' % curve_monthly_name
                draw.draw_curve_precipitation_monthly(
                    data_monthly[curve_monthly_start_index:curve_monthly_end_index, curve_monthly_row, curve_monthly_col],
                    curve_monthly_index, curve_monthly_path)
            # draw his_daily of a year
            elif df.iat[i, 0] == 'draw5':
                his_daily_year = int(df.iat[i, 1])
                his_daily_lon = float(df.iat[i, 2])
                his_daily_lat = float(df.iat[i, 3])
                his_daily_name = df.iat[i, 4]
                his_daily_start_index, his_daily_end_index = preprocess.get_start_end_days(year_start, his_daily_year)
                his_daily_row, his_daily_col = preprocess.get_point_location(list_lon, list_lat, his_daily_lon,
                                                                             his_daily_lat)
                his_daily_index = list(range(1, preprocess.get_days_year(his_daily_year) + 1))
                his_daily_path = path_output + os.sep + '/%s' % his_daily_name
                draw.draw_his_precipitation_daily(
                    data_daily[his_daily_start_index:his_daily_end_index, his_daily_row, his_daily_col], his_daily_index,
                    his_daily_path)
            # draw his monthly of a year
            elif df.iat[i, 0] == 'draw6':
                his_monthly_year = int(df.iat[i, 1])
                his_monthly_lon = float(df.iat[i, 2])
                his_monthly_lat = float(df.iat[i, 3])
                his_monthly_name = df.iat[i, 4]
                his_monthly_start_index = 12 * (his_monthly_year - year_start)
                his_monthly_end_index = his_monthly_start_index + 12
                his_monthly_row, his_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, his_monthly_lon, his_monthly_lat)
                his_monthly_index = list(range(1, 13))
                his_monthly_path = path_output + os.sep + '/%s' % his_monthly_name
                draw.draw_his_precipitation_monthly(
                    data_monthly[his_monthly_start_index:his_monthly_end_index, his_monthly_row, his_monthly_col],
                    his_monthly_index, his_monthly_path)
            # space_time change
            elif df.iat[i, 0] == 'draw7':
                sp_time_start = int(df.iat[i, 2])
                sp_time_end = int(df.iat[i, 3])
                sp_time_name = df.iat[i, 4]
                sp_time_path = path_output + os.sep + '/%s' % sp_time_name
                if df.iat[i, 1] == 'monthly':
                    sp_time_start_index = 12 * (sp_time_start - year_start)
                    sp_time_end_index = 12 * (sp_time_end - year_start + 1)
                    list_monthly_title = []
                    for year in range(sp_time_start, sp_time_end + 1):
                        for month in range(12):
                            title = 'precipitation_monthly_%d_%d' % (year, month + 1)
                            list_monthly_title.append(title)
                    draw.draw_space_time_change(
                        data_monthly[sp_time_start_index:sp_time_end_index], sp_time_path, list_monthly_title)

                elif df.iat[i, 1] == 'daily':
                    sp_time_start_index, sp_time_tmp = preprocess.get_start_end_days(year_start, sp_time_start)
                    sp_time_tmp, sp_time_end_index = preprocess.get_start_end_days(year_start, sp_time_end)
                    list_daily_title = []
                    for year in range(sp_time_start, sp_time_end + 1):
                        sp_time_year_days = preprocess.get_days_year(year)
                        for day in range(sp_time_year_days):
                            title = 'precipitation_daily_%d_%d' % (year, day + 1)
                            list_daily_title.append(title)
                    draw.draw_space_time_change(
                        data_daily[sp_time_start_index:sp_time_end_index], sp_time_path, list_daily_title)
            # draw curve_daily_avg
            elif df.iat[i, 0] == 'draw8':
                curve_daily_start_year = int(df.iat[i, 1])
                curve_daily_end_year = int(df.iat[i, 2])
                curve_daily_lon = float(df.iat[i, 3])
                curve_daily_lat = float(df.iat[i, 4])
                curve_daily_name = df.iat[i, 5]
                curve_daily_start_index, curve_daily_tmp = preprocess.get_start_end_days(year_start, curve_daily_start_year)
                curve_daily_tmp, curve_daily_end_index = preprocess.get_start_end_days(year_start, curve_daily_end_year)
                curve_daily_row, curve_daily_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_daily_lon, curve_daily_lat)
                curve_daily_data = preprocess.get_average_daily(
                    data_daily[curve_daily_start_index:curve_daily_end_index, curve_daily_row, curve_daily_col],
                    curve_daily_end_year - curve_daily_start_year + 1, curve_daily_start_year, nodata_value)
                curve_daily_index = list(range(1, 366))
                curve_daily_path = path_output + os.sep + '/%s' % curve_daily_name
                draw.draw_curve_precipitation_daily(curve_daily_data, curve_daily_index, curve_daily_path)
            # draw curve_monthly_avg
            elif df.iat[i, 0] == 'draw9':
                curve_monthly_year_start = int(df.iat[i, 1])
                curve_monthly_year_end = int(df.iat[i, 2])
                curve_monthly_lon = float(df.iat[i, 3])
                curve_monthly_lat = float(df.iat[i, 4])
                curve_monthly_name = df.iat[i, 5]
                curve_monthly_start_index = 12 * (curve_monthly_year_start - year_start)
                curve_monthly_end_index = 12 * (curve_monthly_year_end - year_start) + 12
                curve_monthly_row, curve_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, curve_monthly_lon, curve_monthly_lat)
                curve_monthly_index = list(range(1, 13))
                curve_monthly_path = path_output + os.sep + '/%s' % curve_monthly_name
                curve_monthly_data = preprocess.get_average_monthly(
                    data_monthly[curve_monthly_start_index:curve_monthly_end_index, curve_monthly_row, curve_monthly_col],
                    curve_monthly_year_end - curve_monthly_year_start + 1, nodata_value)
                draw.draw_curve_precipitation_monthly(curve_monthly_data, curve_monthly_index, curve_monthly_path)
            # draw his_daily_avg
            elif df.iat[i, 0] == 'draw10':
                his_daily_year_start = int(df.iat[i, 1])
                his_daily_year_end = int(df.iat[i, 2])
                his_daily_lon = float(df.iat[i, 3])
                his_daily_lat = float(df.iat[i, 4])
                his_daily_name = df.iat[i, 5]
                his_daily_start_index, his_daily_tmp = preprocess.get_start_end_days(year_start, his_daily_year_start)
                his_daily_tmp, his_daily_end_index = preprocess.get_start_end_days(year_start, his_daily_year_end)
                his_daily_row, his_daily_col = preprocess.get_point_location(
                    list_lon, list_lat, his_daily_lon, his_daily_lat)
                his_daily_index = list(range(1, 366))
                his_daily_path = path_output + os.sep + '/%s' % his_daily_name
                his_daily_data = preprocess.get_average_daily(
                    data_daily[his_daily_start_index:his_daily_end_index, his_daily_row, his_daily_col],
                    his_daily_year_end - his_daily_year_start + 1, his_daily_year_start, nodata_value)
                draw.draw_his_precipitation_daily(his_daily_data, his_daily_index, his_daily_path)
            # draw his_monthly_avg
            elif df.iat[i, 0] == 'draw11':
                his_monthly_year_start = int(df.iat[i, 1])
                his_monthly_year_end = int(df.iat[i, 2])
                his_monthly_lon = float(df.iat[i, 3])
                his_monthly_lat = float(df.iat[i, 4])
                his_monthly_name = df.iat[i, 5]
                his_monthly_start_index = 12 * (his_monthly_year_start - year_start)
                his_monthly_end_index = 12 * (his_monthly_year_end - year_start) + 12
                his_monthly_row, his_monthly_col = preprocess.get_point_location(
                    list_lon, list_lat, his_monthly_lon, his_monthly_lat)
                his_monthly_index = list(range(1, 13))
                his_monthly_path = path_output + os.sep + '/%s' % his_monthly_name
                his_monthly_data = preprocess.get_average_monthly(
                    data_monthly[his_monthly_start_index:his_monthly_end_index, his_monthly_row, his_monthly_col],
                    his_monthly_year_end - his_monthly_year_start + 1, nodata_value)
                draw.draw_his_precipitation_monthly(his_monthly_data, his_monthly_index, his_monthly_path)
            # draw gini_index_avg
            elif df.iat[i, 0] == 'draw12':
                wgini_threshold = float(df.iat[i, 1])
                gini_year_start = int(df.iat[i, 2])
                gini_year_end = int(df.iat[i, 3])
                gini_lon = float(df.iat[i, 4])
                gini_lat = float(df.iat[i, 5])
                gini_name = df.iat[i, 6]
                gini_path = path_output + os.sep + '/%s' % gini_name
                gini_daily_row, gini_daily_col = preprocess.get_point_location(list_lon, list_lat, gini_lon, gini_lat)
                gini_daily_start_index, gini_day_tmp = preprocess.get_start_end_days(year_start, gini_year_start)
                gini_day_tmp, gini_daily_end_index = preprocess.get_start_end_days(year_start, gini_year_end)
                gini_daily_data = preprocess.get_average_daily(
                    data_daily[gini_daily_start_index:gini_daily_end_index, gini_daily_row, gini_daily_col],
                    gini_year_end - gini_year_start + 1, gini_year_start, nodata_value)
                if gini_daily_data[0] < -1:
                    print("draw12: data invalid!!!")
                    continue
                draw.draw_gini_all(gini_daily_data, wgini_threshold, gini_path)
            else:
                break
        print("Draw picture done!")
    print("Delete temp files")
    shutil.rmtree(tmp_dir)
    print("Delete temp files done!")
    print("Done!!!")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
