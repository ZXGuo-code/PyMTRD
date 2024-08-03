# PyMTRD

PyMTRD is  an open-source, and easy-to-use Python package for calculating the metrics of temporal rainfall distribution

# What do I need to run PyMTRD?

* ## [![python](https://img.shields.io/badge/Python-3-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) and the required modules

The packages required to run PyMTRD are:

```
- gif
- matplotlib
- multiprocess
- netCDF4
- numpy
- gdal
- pandas
- scipy
- shutilwhich 
- xarray
- ctypes
- heapq
- os
- platform
- threading
- tkinter
```

* Windows: [![mingw](https://img.shields.io/badge/MinGW-w64-3776AB.svg)](https://www.mingw-w64.org/)
* Linux: g++ package

# How to run PyMTRD?



PyMTRD runs based on the configuration file, which contains the definition of dataset’s structure and the basic information needed by PyMTRD. In Readme_config.md, you will find an explanation of the format of the configuration file, including any required and optional sections, and detailed descriptions of each parameter that can be set in the configuration file, including acceptable values and examples. Besides, we provided  two completed example files (config.csv and config_draw.csv). 

To facilitate usage by different users, we provide two invocation methods.
1、 Directly invoke the relevant code: The example can be attached at pymtrd_example.py.

2、Invoke through graphical user interface (pymtrd_gui.py)

(1) Select Configuration File: Choose the appropriate configuration file that defines the dataset's structure and the basic information such as the paths for inputs and outputs, the metrics to be calculated and the data to be visualized.

(2) Select Processing Mode: Choose from three modes: Calculating the metrics of temporal rainfall distribution, Drawing and Calculating the metrics of temporal rainfall distribution & Drawing.

(3) Choose Whether to Use Parallel Computing: If parallel computing is selected, users should input the number of processes based on the number of logical cores of the CPU. This feature enhances computational efficiency, especially when working with large datasets.

(4) Run the Program: Click the “Run Program” button to execute the analysis based on the selected options and configuration.

![Image](https://github.com/ZXGuo-code/Image_Storage/blob/main/PyMTRD_GUI.png?raw=true)
