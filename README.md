![](https://i.imgur.com/oAryM9f.png)

# Microstim Analysis
Microstimulation is a technique that stimulates a small population of neurons by passing a small electrical current through a nearby microelectrode. We from the Larkum lab in Berlin have used microstimulation as a learning paradigm [[1](https://www.science.org/doi/10.1126/science.aaz3136),[2](https://www.science.org/doi/10.1126/science.abk1859)].


Within this repository you will find the software implementation for analyzing data that was aquired within the microstimulation paradigm. The code, as well as a rundown of its structure, function and implementation are provided.


## Usage
The ```Mouse_Data``` Class contains all data that is associated with the behavioural experiment. To load experimental data, both folder and file structure has to be formatted.

A detailed description guiding you through the data analysis is provided within the jupyter notebook ```analysis.ipyn```. This file can be found in the code folder. You can test run it yourself using the sample data within the data folder.

In short the code functions as follows
1. Import your raw data, extracted from SPIKE2 as .txt files into the ```Mouse_Data``` class.
2. Inspect your data using ```Mouse_Data.full_data()```
3. Analyse using the ```get_threshold_data()``` and ```get_cum_score()``` functions.
4. Visualize using ```plot_daily_threshold()``` and ```plot_intensity_trials()```

# Dependencies
The code depends on the following packages
* numpy (v1.21.5)
* pandas (v1.4.2)
* matplotlib (v3.5.1)
* scipy (v1.7.3)

In order to install all dependencies run
```
pip install requirements.txt
```
# Data aquisition
Software has been written to analyse data from [SPIKE2](https://ced.co.uk/products/spike2) (CED). Our [Microstimulation Setup repository](https://github.com/open-make/mik-delft-microstim) provides the script, the hardware and a detailed instruction on how to get started gathering your own data. 

![](https://i.imgur.com/arHQWjf.png)


# Demo the code
We provide a Jupyter Notebook that walks you through a test dataset to verify functionality. Within this ```analysis.ipyn``` notebook you also find instructions on how to use your own data. It is possible to use data gathered from different sources on the condition that filename and file content are standardized. 
