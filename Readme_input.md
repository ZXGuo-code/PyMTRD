# Format of input csv file

### input.csv(Data Processing and Drawing)

Write the following in order

1 model name of precipitation

2 str("path_file_nc"),  int(number of years corresponding to the study data), variable names for precipitation data of the input nc files, variable names for longitude data of the input nc files, variables names for latitude data of the input nc files

3 the index(the first column) and address(the second column) of the nc files

4 str("year_start"), the start year of the study data

5 str("year_end"), the end year of the study data

6 str("nodata_value"), the nodata value of the input nc files

7 str("operation"), 1/2(1 pixel by pixel 2 parameter by parmeter)

8 str("output_dir"), the folder of the output files

9 str("intensity"), 0/1(0 don't calculate intensity of precipitation, 1 calculate intensity of precipitaiton)

10 str("list_intensity_percentile"), 0/1(whether to use the input percentiles, if not, the intensity of precipitation will be the average of percipitation), the list of percentiles(occupy a cell with different percentiles separated by spaces)

11 str("times"), 0/1(whether to calculate the frequency of precipitation)

12 str("cdds"), 0/1(whether to calculate the consecutive dry days)

13 str("list_cdd_percentile"), 0/1(whether to use the input percentiles, if not, the cdds will be the average value of the maximum n(the number of years) precipitation intervals), the list of percentiles(occupy a cell with different percentiles separated by spaces)

14 str("list_times_cdds_threshold"), 0/1(whether to use the input thresholds to calculate times and cdds, if not, the threshold value will be initialized to 1), the list of thresholds(occupy a cell with different percentiles separated by spaces)

15 str("ugini"), 0/1(whether to calculate unranked gini index)

16 str("gini"), 0/1(wherther to calculate gini index)

17 str("wgini"), 0/1(whether to calculate wet-day gini index)

18 str("list_wgini_threshold"), 0/1(whether to use the input thresholds to calculate wet-day gini index, if not, the threshold value will be initialized to 1), the list of thresholds(occupy a cell with different percentiles separated by spaces)

19 str("pci"), 0/1(whether to calculate Precipitation concentration index)

21 str("dsi"), 0/1(whether to calculate Dimensionless seasonality index)

20 str("si"), 0/1(whether to calculate seasonality index)



The next part is the graphing part(the first column could be draw1, draw2, draw3, draw4, draw5, draw6 or draw7, which will be described the function and the format requirements. Drawings can be disordered and repeated)

1 draw1: plot the correlation of different variables

str("draw1"), variable name of x(array_pci or array_dsi or array_si), variable name of y(array_pci or array_dsi or array_si), xlable name, ylable name, the year corresponding to the variable(from start year to end year), output name(*.jpg, such as pci_si_2001.jpg)

2 draw2: draw a picture of gini index which include 3 subgraphs(unranked gini index, gini index, wet-day gini index)

str("draw2"), the thrshold of wet-day gini index, the year corresponding to the variables(from start year to end year), longitude of the location, latitude of the location, output name(*.jpg)

3 draw3: draw a curve of daily precipitation

str("draw3"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

4 draw4: draw a curve of monthly precipitation

str("draw4"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

5 draw5: draw a histogram of daily precipitation

str("draw5"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

6 draw6: draw a histogram of monthly precipitation

str("draw6"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

7 draw7: plot data of precipitation to a gif image

str("draw7"), data type(monthly/daily), start year of the gif image(from start year to end year), end year of the gif image(from start year to end year), output name(*.jpg)



### input_draw.csv(Drawing)

Write the following in order

1 str("path_file_nc"), int(number of years corresponding to the study data), variable names for precipitation data of the input nc files, variable names for longitude data of the input nc files, variables names for latitude data of the input nc files

2 the index(the first column) and address(the second column) of the nc files

3 str("year_start"), the start year of the study data

4 str("year_end"), the end year of the study data

5 str("nodata_value"), the nodata value of the input nc files

6 str("output_dir"), the folder of the output files



The next part is the graphing part(the first column could be draw1, draw2, draw3, draw4, draw5, draw6 or draw7, which will be described the function and the format requirements. Drawings can be disordered and repeated)

1 draw1: plot the correlation of different variables

str("draw1"), path of variable(.tif), path of variable(.tif), xlable name, ylable name, output name(*.jpg, such as pci_si_2001.jpg)

2 draw2: draw a picture of gini index which include 3 subgraphs(unranked gini index, gini index, wet-day gini index)

str("draw2"), the thrshold of wet-day gini index, the year corresponding to the variables(from start year to end year), longitude of the location, latitude of the location, output name(*.jpg)

3 draw3: draw a curve of daily precipitation

str("draw3"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

4 draw4: draw a curve of monthly precipitation

str("draw4"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

5 draw5: draw a histogram of daily precipitation

str("draw5"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

6 draw6: draw a histogram of monthly precipitation

str("draw6"), the year corresponding to the curve(from start year to end year), longitude of the location, latitude of the locatioin, output name(*.jpg)

7 draw7: plot data of precipitation to a gif image

str("draw7"), data type(monthly/daily), start year of the gif image(from start year to end year), end year of the gif image(from start year to end year), output name(*.jpg) 