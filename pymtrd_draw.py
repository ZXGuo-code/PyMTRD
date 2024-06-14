import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import gif
from osgeo import gdal
from scipy import stats

plt.switch_backend('agg')
font_size_label = 16
font_size_title = 18
font_size_ticks = 14


def write_tiff(outputfile, nrows, ncols, geotrans, projinfo, nodata_value, data):
    """
    Write tiff image
    outputfile, the address of output file
    nrows, the rows of output image
    ncols, the cols of output image
    geotrans, the geotransform of output image
    projinfo, the projection of output image
    nodata_vale, the invalid data
    data, the pixel values of output image 2-dim (rows, cols)
    """
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(outputfile, ncols, nrows, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(geotrans)
    ds_out.SetProjection(projinfo)
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(nodata_value)
    band_out.WriteArray(data)
    ds_out.FlushCache()

def draw_cor_var(var1, var2, var1_name, var2_name, path_output):
    """
    Plotting the correlation of two different variables
    var1: 2dim
    var2: 2dim
    var1_name: name of var1
    var2_name: name of var2
    """
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    var1_1dim = []
    var2_1dim = []
    
    for i in range(var1.shape[0]):
        for j in range(var1.shape[1]):
            if var1[i][j] > -1 and var2[i][j] > -1:
                var1_1dim.append(var1[i][j])
                var2_1dim.append(var2[i][j])

    H = ax.hist2d(var1_1dim, var2_1dim, bins=400, cmap=cm.jet, norm=matplotlib.colors.LogNorm())

    # Calculate slope and intercept
    array_var1 = np.array(var1_1dim)
    array_var2 = np.array(var2_1dim)
    slope, intercept = np.polyfit(array_var1, array_var2, 1)
    # print(slope, intercept)
    # print(np.corrcoef(array_var1, array_var2))
    r, p_value = stats.pearsonr(array_var1, array_var2)
    plt.plot(array_var1, slope * array_var1 + intercept, color='r')
    ax.set_title('{}_{} $r={:.3f}, p={:.3f}$'.format(var1_name, var2_name, r, p_value), loc='left',fontsize=font_size_title)
    ax.set_xlabel(var1_name, fontsize=font_size_label)
    ax.set_ylabel(var2_name, fontsize=font_size_label)
    cbar = fig.colorbar(H[3], ax=ax)
    cbar.ax.tick_params(labelsize=font_size_ticks)
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.tight_layout()
    plt.savefig(path_output)
    plt.close()


# gini
def draw_gini(ax, list_x, list_y, label_x, label_y, title):
    """
    draw the picture of gini index or wet_gini index
    ax: a subplot
    list_x: a list of data
    list_y: a list of data
    label_x: the label of x
    label_x: the label of y
    title: the title of the subplot
    """
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', -0))
    ax.spines['left'].set_position(('data', 0))
    ax.plot(list_x, list_y, color='black')
    ax.plot(list_x, list_x, color='black')
    ax.plot([0, 1, 1, 1], [0, 0, 0, 1], color='black')
    ax.fill_between(list_x, list_y)
    ax.fill_between(list_x, list_x, list_y, where=list_y <= list_x)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(title)


def draw_ugini(ax, list_x, list_y_max, list_y_min, label_x, label_y, title):
    """
    draw the picture of unranked gini index
    ax: a subplot
    list_x: a list of data
    list_y_max: a list of max_y
    list_y_min: a list of min_y
    label_x: the label of x
    label_x: the label of y
    title: the title of the subplot
    """
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', -0))
    ax.spines['left'].set_position(('data', 0))
    ax.plot(list_x, list_y_max, color='black')
    ax.plot(list_x, list_y_min, color='black')
    ax.plot([0, 1, 1, 1], [0, 0, 0, 1], color='black')
    ax.fill_between(list_x, list_y_min)
    ax.fill_between(list_x, list_y_min, list_y_max, where=list_y_min <= list_y_max)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(title)


def draw_gini_all(ts_daily, threshold, path_output, title='gini_index'):
    """
    draw a picture of gini index which include 3 sub_graphs(unranked gini index, gini index and wet_gini index)
    ts_daily: 1-dim, a list of precipitation in a year
    threshold: the threshold which is used to determine whether it is a wet day
    path_output: the output path of the picture
    """
    fig = plt.figure(dpi=300, figsize=(9, 3))
    # fig = plt.figure(dpi=300, figsize=(3, 9))
    ts_daily = [n for n in ts_daily if n > -1]

    # Unranked Gini
    cum_wealths_u = np.cumsum(ts_daily)
    sum_wealths_u = cum_wealths_u[-1]
    xarray_u = np.array(range(0, len(cum_wealths_u))) / float(len(cum_wealths_u) - 1)
    yarray_u = cum_wealths_u / sum_wealths_u
    yarray_u_max = np.maximum(xarray_u, yarray_u)
    yarray_u_min = np.minimum(xarray_u, yarray_u)
    ax1 = fig.add_subplot(131)
    # ax1 = fig.add_subplot(311)
    draw_ugini(ax1, xarray_u, yarray_u_max, yarray_u_min, 'Normalized days', 'Normalized cumulative precipitation',
               'Unranked Gini index')

    ts_daily.sort()
    # gini
    cum_wealths = np.cumsum(ts_daily)
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / float(len(cum_wealths) - 1)
    yarray = cum_wealths / sum_wealths
    ax2 = fig.add_subplot(132)
    # ax2 = fig.add_subplot(312)
    draw_gini(ax2, xarray, yarray, 'Normalized days', 'Normalized cumulative precipitation', 'Gini index')

    # wet_gini
    data_w = [n for n in ts_daily if n > threshold]
    data_w.insert(0, 0)
    cum_wealths_w = np.cumsum(data_w)
    sum_wealths_w = cum_wealths_w[-1]
    xarray_w = np.array(range(0, len(cum_wealths_w))) / float(len(cum_wealths_w) - 1)
    yarray_w = cum_wealths_w / sum_wealths_w
    ax3 = fig.add_subplot(133)
    # ax3 = fig.add_subplot(313)
    draw_gini(ax3, xarray_w, yarray_w, 'Normalized days', 'Normalized cumulative precipitation', 'Wet-day Gini index')

    # plt.title(title)
    plt.tight_layout()
    plt.savefig(path_output)
    plt.close()


def draw_curve_precipitation_daily(ts_daily, list_date, path_output, title='Daily precipitation'):
    """
    draw a curve of daily precipitation
    ts_daily: 1-dim, a list of precipitation in a year
    date_daily: list of time which correspond to the precipitation list
    path_output: the output path of the picture
    """
    plt.figure(dpi=400)
    plt.plot(list_date, ts_daily)
    plt.xlabel('Day', fontsize=font_size_label)
    plt.ylabel('Precipitation(mm)', fontsize=font_size_label)
    plt.title(title, fontsize=font_size_title)
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.tight_layout()
    plt.savefig(path_output)
    plt.close()


def draw_curve_precipitation_monthly(ts_monthly, list_date, path_output, title='Monthly precipitation'):
    """
    draw a curve of monthly precipitation
    ts_monthly: 1-dim, a list of precipitation in a year
    date_daily: list of time which correspond to the precipitation list
    path_output: the output path of the picture
    """
    plt.figure(dpi=400)
    plt.ylim(ymin=0, ymax=max(ts_monthly)*1.2)
    plt.plot(list_date, ts_monthly)
    plt.xlabel('Month', fontsize=font_size_label)
    plt.ylabel('Precipitation(mm)', fontsize=font_size_label)
    plt.title(title, fontsize=font_size_title)
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.tight_layout()
    plt.savefig(path_output)
    plt.close()


def draw_his_precipitation_daily(ts_daily, list_date, path_output, title='Daily precipitation'):
    """
    draw a histogram of daily precipitation
    ts_daily: 1-dim, a list of precipitation in a year
    date_daily: list of time which correspond to the precipitation list
    path_output: the output path of the picture
    """
    plt.figure(dpi=400)
    plt.bar(list_date, ts_daily, width=0.8)
    plt.xlabel('Day', fontsize=font_size_label)
    plt.ylabel('Precipitation(mm)', fontsize=font_size_label)
    plt.title(title, fontsize=font_size_title)
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.title(title, fontsize=font_size_title)
    plt.tight_layout()
    plt.savefig(path_output)
    plt.close()


def draw_his_precipitation_monthly(ts_monthly, list_date, path_output, title='Monthly precipitation'):
    """
    draw a histogram of monthly precipitation
    ts_monthly: 1-dim, a list of precipitation in a year
    date_daily: list of time which is correspond to the precipitation list
    path_output: the output path of the picture
    """
    plt.figure(dpi=400)
    plt.bar(list_date, ts_monthly)
    plt.xlabel('Month', fontsize=font_size_label)
    plt.ylabel('Precipitation(mm)', fontsize=font_size_label)
    plt.title(title, fontsize=font_size_title)
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.title(title, fontsize=font_size_title)
    plt.tight_layout()
    plt.savefig(path_output)
    plt.close()


# Space-time animation
@gif.frame
def draw_one_ts(sp_data, title, clim_percent, nodata_value=-9999):
    """
    plot a time step of data as a frame of a gif image
    sp_data: 2-dim, space data of a time
    clim_percent: get the clim_percent and 100-clim_percent of the corresponding image value, 
    and then get the display range of the output picture
    nodata_value: the background value of the data
    """
    sp_data[sp_data == nodata_value] = np.nan
    per_min = np.nanpercentile(sp_data, clim_percent)
    per_max = np.nanpercentile(sp_data, 100 - clim_percent)

    plt.figure(dpi=300)
    plt.imshow(sp_data, cmap=plt.cm.jet)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar(orientation='horizontal')
    # range of display
    plt.clim(per_min, per_max)


def draw_space_time_change(sp_data, path_output, list_title, clim_percent=2, nodata_value=-9999, duration=1000):
    """
    plot data to a gif image
    sp_data: 3-dim, the first is time
    path_output: the output path of the picture
    list_title: a list of title
    clim_percent: get the clim_percent and 100-clim_percent of the corresponding image value, 
    and then get the display range of the output picture
    nodata_value: the background value of the data
    duration: the duration between two picture
    """
    frames = []
    for i in range(sp_data.shape[0]):
        frame = draw_one_ts(sp_data[i], list_title[i], clim_percent, nodata_value)
        frames.append(frame)

    gif.options.matplotlib["dpi"] = 300
    gif.save(frames, path_output, duration=duration)
