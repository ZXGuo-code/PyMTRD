# Format of configuration file

### config.csv(Data Processing and Drawing)

Write the following in order

1 model name of precipitation

e.g. 

|        |
| ------ |
| cmorph |

2 str("path_file_nc"),  int(number of years corresponding to the study data), variable names for precipitation data of the input nc files, variable names for longitude data of the input nc files, variables names for latitude data of the input nc files

e.g. 

|              |      |        |           |          |
| ------------ | ---- | ------ | --------- | -------- |
| path_file_nc | 10   | Precip | longitude | latitude |

3 the index(the first column) and address(the second column) of the nc files

e.g. 

|      |                                          |
| ---- | ---------------------------------------- |
| 0    | .\input\Precip.CMORPH_v1_BLD_EOD.2001.nc |
| 1    | .\input\Precip.CMORPH_v1_BLD_EOD.2002.nc |
| 2    | .\input\Precip.CMORPH_v1_BLD_EOD.2003.nc |
| 3    | .\input\Precip.CMORPH_v1_BLD_EOD.2004.nc |
| 4    | .\input\Precip.CMORPH_v1_BLD_EOD.2005.nc |
| 5    | .\input\Precip.CMORPH_v1_BLD_EOD.2006.nc |
| 6    | .\input\Precip.CMORPH_v1_BLD_EOD.2007.nc |
| 7    | .\input\Precip.CMORPH_v1_BLD_EOD.2008.nc |
| 8    | .\input\Precip.CMORPH_v1_BLD_EOD.2009.nc |
| 9    | .\input\Precip.CMORPH_v1_BLD_EOD.2010.nc |

4 str("year_start"), the start year of the study data

e.g.

|            |      |
| ---------- | ---- |
| year_start | 2001 |

5 str("year_end"), the end year of the study data

e.g.

|          |      |
| -------- | ---- |
| year_end | 2010 |

6 str("nodata_value"), the nodata value of the input nc files

e.g.

|              |       |
| ------------ | ----- |
| nodata_value | -9999 |

7 str("operation"), 1/2(1 pixel by pixel 2 parameter by parameter), the two operations are available using single cpu core,  but only operation 1 is available using multiple cpu cores

e.g.

|                |      |                                          |
| -------------- | ---- | ---------------------------------------- |
| operation_mode | 1    | 1pixel by piexel 2parameter by parameter |

8 str("output_dir"), the folder of the output files

e.g.

|            |          |
| ---------- | -------- |
| output_dir | .\output |

9 str("intensity"), 0/1(0 don't calculate intensity of precipitation, 1 calculate intensity of precipitaiton)

e.g.

|           |      |
| --------- | ---- |
| intensity | 1    |

10 str("list_intensity_percentile"), 0/1(whether to use the input percentiles, if not, the intensity of precipitation will be the average of percipitation), the list of percentiles(occupy a cell with different percentiles separated by spaces)

e.g.

|                           |      |         |
| ------------------------- | ---- | ------- |
| list_intensity_percentile | 1    | 1 50 90 |

11 str("times"), 0/1(whether to calculate the frequency of precipitation)

e.g.

|       |      |
| ----- | ---- |
| times | 1    |

12 str("cdds"), 0/1(whether to calculate the consecutive dry days)

e.g.

|      |      |
| ---- | ---- |
| cdds | 1    |

13 str("list_cdd_percentile"), 0/1(whether to use the input percentiles, if not, the cdds will be the average value of the maximum n(the number of years) precipitation intervals), the list of percentiles(occupy a cell with different percentiles separated by spaces)

e.g.

|                     |      |         |
| ------------------- | ---- | ------- |
| list_cdd_percentile | 0    | 5 50 90 |

14 str("list_times_cdds_threshold"), 0/1(whether to use the input thresholds to calculate times and cdds, if not, the threshold value will be initialized to 1), the list of thresholds(occupy a cell with different percentiles separated by spaces)

e.g.

|                           |      |       |
| ------------------------- | ---- | ----- |
| list_times_cdds_threshold | 0    | 1 3 5 |

15 str("ugini"), 0/1(whether to calculate unranked gini index)

e.g.

|       |      |
| ----- | ---- |
| ugini | 1    |

16 str("gini"), 0/1(wherther to calculate gini index)

e.g.

|      |
| ---- |
| gini |

17 str("wgini"), 0/1(whether to calculate wet-day gini index)

e.g.

|       |      |
| ----- | ---- |
| wgini | 1    |

18 str("list_wgini_threshold"), 0/1(whether to use the input thresholds to calculate wet-day gini index, if not, the threshold value will be initialized to 1), the list of thresholds(occupy a cell with different percentiles separated by spaces)

e.g.

|                      |      |      |
| -------------------- | ---- | ---- |
| list_wgini_threshold | 1    | 1 3  |

19 str("pci"), 0/1(whether to calculate Precipitation concentration index)

e.g.

|      |      |
| ---- | ---- |
| pci  | 1    |

21 str("dsi"), 0/1(whether to calculate Dimensionless seasonality index)

e.g.

|      |      |
| ---- | ---- |
| dsi  | 1    |

20 str("si"), 0/1(whether to calculate seasonality index)

e.g.

|      |      |
| ---- | ---- |
| si   | 1    |

The next part is the graphing part(the first column could be draw1, draw2, draw3, draw4, draw5, draw6, draw7, draw 8, draw 9, draw 10, draw 11, and draw 12, which will be described the function and the format requirements. Drawings can be disordered and repeated)

1 draw1: plot the correlation of different variables for a year

str("draw1"), variable name of x(array_pci or array_dsi or array_si), variable name of y(array_pci or array_dsi or array_si), xlable name, ylable name, the year corresponding to the variable(from start year to end year), output name(*.jpg, such as pci_si_2001.jpg)

e.g.

|       |           |          |      |      |      |                 |
| ----- | --------- | -------- | ---- | ---- | ---- | --------------- |
| draw1 | array_pci | array_si | pci  | si   | 2001 | pci_si_2001.jpg |

2 draw2: draw a picture of gini index which include 3 subgraphs(unranked gini index, gini index, wet-day gini index) for a year

str("draw2"), the thrshold of wet-day gini index, the year corresponding to the variables(from start year to end year), longitude of the location, latitude of the location, output name(*.jpg)

e.g.

|       |      |      |       |      |                       |
| ----- | ---- | ---- | ----- | ---- | --------------------- |
| draw2 | 1    | 2001 | 118.8 | 32.1 | nanjing_gini_2001.jpg |

3 draw3: draw a curve of daily precipitation for a year

str("draw3"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                              |
| ----- | ---- | ----- | ---- | ---------------------------- |
| draw3 | 2001 | 118.8 | 32.1 | nanjing_curve_daily_2001.jpg |

4 draw4: draw a curve of monthly precipitation for a year

str("draw4"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                                |
| ----- | ---- | ----- | ---- | ------------------------------ |
| draw4 | 2001 | 118.8 | 32.1 | nanjing_curve_monthly_2001.jpg |

5 draw5: draw a histogram of daily precipitation for a year

str("draw5"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                            |
| ----- | ---- | ----- | ---- | -------------------------- |
| draw5 | 2001 | 118.8 | 32.1 | nanjing_his_daily_2001.jpg |

6 draw6: draw a histogram of monthly precipitation for a year

str("draw6"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                              |
| ----- | ---- | ----- | ---- | ---------------------------- |
| draw6 | 2001 | 118.8 | 32.1 | nanjing_his_monthly_2001.jpg |

7 draw7: plot data of precipitation to a gif image over multiple years

str("draw7"), data type(monthly/daily), start year of the gif image(from start year to end year), end year of the gif image(from start year to end year), output name(*.jpg)

e.g.

|       |         |      |      |                                     |
| ----- | ------- | ---- | ---- | ----------------------------------- |
| draw7 | monthly | 2001 | 2010 | precipitation_2001_2010_monthly.gif |



8 draw8: draw a curve of average daily precipitation over multiple years

e.g.

|       |      |      |       |      |                                                 |
| ----- | ---- | ---- | ----- | ---- | ----------------------------------------------- |
| draw8 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_curve_precipitation_2001_2010_daily.jpg |

9 draw9: draw a curve of average monthly precipitation over multiple years

e.g.

|       |      |      |       |      |                                                   |
| ----- | ---- | ---- | ----- | ---- | ------------------------------------------------- |
| draw9 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_curve_precipitation_2001_2010_monthly.jpg |

10 draw10: draw a histogram of average daily precipitation over multiple years

e.g.

|        |      |      |       |      |                                               |
| ------ | ---- | ---- | ----- | ---- | --------------------------------------------- |
| draw10 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_his_precipitation_2001_2010_daily.jpg |

11 draw11: draw a histogram of average monthly precipitation over multiple years

e.g.

|        |      |      |       |      |                                                 |
| ------ | ---- | ---- | ----- | ---- | ----------------------------------------------- |
| draw11 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_his_precipitation_2001_2010_monthly.jpg |

12 draw12: draw a picture of average gini index which include 3 subgraphs(unranked gini index, gini index, wet-day gini index) over multiple years

e.g.

|        |      |      |      |       |      |                            |
| ------ | ---- | ---- | ---- | ----- | ---- | -------------------------- |
| draw12 | 1    | 2001 | 2010 | 118.8 | 32.1 | Nanjing_gini_2001_2010.jpg |

### config_draw.csv(Drawing)

Write the following in order

1 str("path_file_nc"), int(number of years corresponding to the study data), variable names for precipitation data of the input nc files, variable names for longitude data of the input nc files, variables names for latitude data of the input nc files

e.g. 

|              |      |        |           |          |
| ------------ | ---- | ------ | --------- | -------- |
| path_file_nc | 10   | Precip | longitude | latitude |

2 the index(the first column) and address(the second column) of the nc files

e.g. 

|      |                                          |
| ---- | ---------------------------------------- |
| 0    | .\input\Precip.CMORPH_v1_BLD_EOD.2001.nc |
| 1    | .\input\Precip.CMORPH_v1_BLD_EOD.2002.nc |
| 2    | .\input\Precip.CMORPH_v1_BLD_EOD.2003.nc |
| 3    | .\input\Precip.CMORPH_v1_BLD_EOD.2004.nc |
| 4    | .\input\Precip.CMORPH_v1_BLD_EOD.2005.nc |
| 5    | .\input\Precip.CMORPH_v1_BLD_EOD.2006.nc |
| 6    | .\input\Precip.CMORPH_v1_BLD_EOD.2007.nc |
| 7    | .\input\Precip.CMORPH_v1_BLD_EOD.2008.nc |
| 8    | .\input\Precip.CMORPH_v1_BLD_EOD.2009.nc |
| 9    | .\input\Precip.CMORPH_v1_BLD_EOD.2010.nc |

3 str("year_start"), the start year of the study data

e.g.

|            |      |
| ---------- | ---- |
| year_start | 2001 |

4 str("year_end"), the end year of the study data

e.g.

|          |      |
| -------- | ---- |
| year_end | 2010 |

5 str("nodata_value"), the nodata value of the input nc files

e.g.

|              |       |
| ------------ | ----- |
| nodata_value | -9999 |

6 str("output_dir"), the folder of the output files

e.g.

|            |          |
| ---------- | -------- |
| output_dir | .\output |

The next part is the graphing part(the first column could be draw1, draw2, draw3, draw4, draw5, draw6 or draw7, which will be described the function and the format requirements. Drawings can be disordered and repeated)

1 draw1: plot the correlation of different variables

str("draw1"), path of variable(.tif), path of variable(.tif), xlable name, ylable name, output name(*.jpg, such as pci_si_2001.jpg)

The next part is the graphing part(the first column could be draw1, draw2, draw3, draw4, draw5, draw6, draw7, draw 8, draw 9, draw 10, draw 11, and draw 12, which will be described the function and the format requirements. Drawings can be disordered and repeated)

1 draw1: plot the correlation of different variables

str("draw1"), path of variable(.tif), path of variable(.tif), xlable name, ylable name, output name(*.jpg, such as pci_si_2001.jpg)

e.g.

|       |                       |                      |      |      |                 |
| ----- | --------------------- | -------------------- | ---- | ---- | --------------- |
| draw1 | ./cmorph_pci_2001.tif | ./cmorph_si_2001.tif | pci  | si   | pci_si_2001.jpg |

2 draw2: draw a picture of gini index which include 3 subgraphs(unranked gini index, gini index, wet-day gini index) for a year

str("draw2"), the thrshold of wet-day gini index, the year corresponding to the variables(from start year to end year), longitude of the location, latitude of the location, output name(*.jpg)

e.g.

|       |      |      |       |      |                       |
| ----- | ---- | ---- | ----- | ---- | --------------------- |
| draw2 | 1    | 2001 | 118.8 | 32.1 | nanjing_gini_2001.jpg |

3 draw3: draw a curve of daily precipitation for a year

str("draw3"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                              |
| ----- | ---- | ----- | ---- | ---------------------------- |
| draw3 | 2001 | 118.8 | 32.1 | nanjing_curve_daily_2001.jpg |

4 draw4: draw a curve of monthly precipitation for a year

str("draw4"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                                |
| ----- | ---- | ----- | ---- | ------------------------------ |
| draw4 | 2001 | 118.8 | 32.1 | nanjing_curve_monthly_2001.jpg |

5 draw5: draw a histogram of daily precipitation for a year

str("draw5"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                            |
| ----- | ---- | ----- | ---- | -------------------------- |
| draw5 | 2001 | 118.8 | 32.1 | nanjing_his_daily_2001.jpg |

6 draw6: draw a histogram of monthly precipitation for a year

str("draw6"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

e.g.

|       |      |       |      |                              |
| ----- | ---- | ----- | ---- | ---------------------------- |
| draw6 | 2001 | 118.8 | 32.1 | nanjing_his_monthly_2001.jpg |

7 draw7: plot data of precipitation to a gif image over multiple years

str("draw7"), data type(monthly/daily), start year of the gif image(from start year to end year), end year of the gif image(from start year to end year), output name(*.jpg)

e.g.

|       |         |      |      |                                     |
| ----- | ------- | ---- | ---- | ----------------------------------- |
| draw7 | monthly | 2001 | 2010 | precipitation_2001_2010_monthly.gif |



8 draw8: draw a curve of average daily precipitation over multiple years

e.g.

|       |      |      |       |      |                                                 |
| ----- | ---- | ---- | ----- | ---- | ----------------------------------------------- |
| draw8 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_curve_precipitation_2001_2010_daily.jpg |

9 draw9: draw a curve of average monthly precipitation over multiple years

e.g.

|       |      |      |       |      |                                                   |
| ----- | ---- | ---- | ----- | ---- | ------------------------------------------------- |
| draw9 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_curve_precipitation_2001_2010_monthly.jpg |

10 draw10: draw a histogram of average daily precipitation over multiple years

e.g.

|        |      |      |       |      |                                               |
| ------ | ---- | ---- | ----- | ---- | --------------------------------------------- |
| draw10 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_his_precipitation_2001_2010_daily.jpg |

11 draw11: draw a histogram of average monthly precipitation over multiple years

e.g.

|        |      |      |       |      |                                                 |
| ------ | ---- | ---- | ----- | ---- | ----------------------------------------------- |
| draw11 | 2001 | 2010 | 118.8 | 32.1 | Nanjing_his_precipitation_2001_2010_monthly.jpg |

12 draw12: draw a picture of average gini index which include 3 subgraphs(unranked gini index, gini index, wet-day gini index) over multiple years

e.g.

|        |      |      |      |       |      |                            |
| ------ | ---- | ---- | ---- | ----- | ---- | -------------------------- |
| draw12 | 1    | 2001 | 2010 | 118.8 | 32.1 | Nanjing_gini_2001_2010.jpg |

