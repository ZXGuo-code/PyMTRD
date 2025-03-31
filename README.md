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

* Windows: [![mingw](https://img.shields.io/badge/MinGW-w64-3776AB.svg)](https://www.mingw-w64.org/) , and compile pymtrd_gini.dll using: g++ -o pymtrd_gini.dll -shared -fPIC pymtrd_gini.cpp
* Linux: g++ package, and compile pymtrd_gini.so using: g++ -o pymtrd_gini.so -shared -fPIC pymtrd_gini.cpp

# How to run PyMTRD?



PyMTRD runs based on the configuration file, which contains the definition of dataset’s structure and the basic information needed by PyMTRD. In Readme_config.md, you will find an explanation of the format of the configuration file, including any required and optional sections, and detailed descriptions of each parameter that can be set in the configuration file, including acceptable values and examples. Besides, we provided  two completed example files (config.csv and config_draw.csv). 

To facilitate usage by different users, we provide two invocation methods.

1、 Directly invoke the relevant code: The example can be attached at pymtrd_example.py.

2、Invoke through the graphical user interface (pymtrd_gui.py)

In order to calculate these metrics based on the GUI, users need to follow these steps:

(1)	Select configuration file: Choose the appropriate configuration file that defines the dataset's structure and the basic information such as the paths for inputs and outputs, the metrics to be calculated and the data to be visualized.

(2)	Select processing mode: Choose the processing mode among three options: “Calculating metrics”, “Drawing”, and “Calculating metrics and drawing”.

(3)	Choose whether to use parallel computing: If parallel computing is selected, users need to input the number of processes. This feature enhances computational efficiency, especially when working with large datasets.

(4)	Run the program: Click the “Run program” button to execute the analysis based on the selected options and configuration.

![Image](https://github.com/ZXGuo-code/Image_Storage/blob/main/PyMTRD_GUI.png?raw=true)

# Citation
Guo, Z., Wang, Y., Liu, C., Yang, W., Liu, J., PyMTRD: A Python package for calculating the metrics of temporal rainfall distribution, Environmental Modelling and Software, https:// doi.org/10.1016/j.envsoft.2024.106201.
