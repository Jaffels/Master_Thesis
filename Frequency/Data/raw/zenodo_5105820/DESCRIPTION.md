# Overview 
This repository contains ready-to-use frequency time series as well as the corresponding pre-processing scripts in python. The data covers three synchronous areas of the European power grid:
* Continental Europe 
* Great Britain
* Nordic

This work is part of the paper "Predictability of Power Grid Frequency"[1]. Please cite this paper, when using the data and the code. For a detailed documentation of the pre-processing procedure we refer to the supplementary material of the paper. 

# Data sources
We downloaded the frequency recordings from publically available repositories of three different Transmission System Operators (TSOs).
* **Continental Europe** [2]: We downloaded the data from the German TSO *TransnetBW GmbH*, which retains the Copyright on the data, but allows to re-publish it upon request [3]. 
* **Great Britain** [4]: The download was supported by National Grid ESO Open Data, which belongs to the British TSO *National Grid*. They publish the frequency recordings under the NGESO Open License [5]. 
* **Nordic** [6]: We obtained the data from the Finish TSO *Fingrid*, which provides the data under the open license CC-BY 4.0 [7]. 

# Content of the repository

## Scripts
1) In the `Download_scripts` folder you will find three scripts to automatically download frequency data from the TSO's websites. 
2) In `convert_data_format.py` we save the data with corrected
timestamp formats. Missing data is marked as NaN (processing step (1) in the supplementary material of [1]).
3) In `clean_corrupted_data.py` we load the converted data and identify corrupted recordings. We mark them as NaN and clean some of the resulting data holes (processing step (2) in the supplementary material of [1]). 

The python scripts run with Python 3.7 and with the packages found in `requirements.txt`.

## Yearly converted and cleansed data
The folders `<year>_converted` contain the output of `convert_data_format.py` and `<year>_cleansed` contain the output of `clean_corrupted_data.py`. 
* **File type**: The files are zipped csv-files, where each file comprises one year. 
* **Data format**: The files contain two columns. The second column contains the frequency values in Hz.  The first one represents the time stamps in the format *Year-Month-Day Hour-Minute-Second*, which is given as naive local time. The local time refers to the following time zones and includes Daylight Saving Times (python time zone in brackets):
    * TransnetBW: Continental European Time (*CET*)
    * Nationalgrid: Great Britain (*GB*)
    * Fingrid: Finland (*Europe/Helsinki*)
* **NaN representation**: We mark corrupted and missing data as "NaN" in the csv-files. 

# Use cases
We point out that this repository can be used in two different was:
1) **Use pre-processed data**: You can directly use the converted or the cleansed data. Note however, that both data sets include segments of NaN-values due to missing and corrupted recordings. Only a very small part of the NaN-values were eliminated in the cleansed data to not manipulate the data too much. 
2) **Produce your own cleansed data**: Depending on your application, you might want to cleanse the data in a custom way. You can easily add your custom cleansing procedure in `clean_corrupted_data.py` and then produce cleansed data from the raw data in `<year>_converted`. 

# Acknowledgements
We thank Mark Thiele for fruitful discussions. Furthermore, we gratefully acknowledge support from the German Federal Ministry of Education and Research (BMBF grant no. 03EK3055B) and the Helmholtz Association (via the the *Helmholtz School for Data Science in Life, Earth and Energy* (HDS-LEE), the joint initiative *Energy System 2050 - A Contribution of the Research Field Energy* and via the grant No. VH-NG-1025). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 840825. 

# Changelog
Version 2:
* Add time zone information to description
* Include new frequency data 
* Update references 
* Change folder structure to yearly folders
Version 3: 
* Correct TransnetBW files for missing data in May 2016

# References
* [1] J. Kruse, B. Schäfer, and D. Witthaut, “Predictability of Power Grid Frequency,” IEEE Access, vol. 8, pp. 149435–149446, 2020.
* [2] TransnetBW GmbH. "Netzfrequenz". https://www.transnetbw.de/de/strommarkt/systemdienstleistungen/regelenergie-bedarf-und-abruf (accessed on 26.03.2020)
* [3] TransnetBW GmbH. "Conditions of Use". https://www.transnetbw.com/en/conditions-of-use (accessed on 26.03.2020)
* [4] National Grid ESO. "Historic frequency data". https://www.nationalgrideso.com/balancing-services/frequency-response-services/historic-frequency-data (accessed on 26.03.2020)
* [5] National Grid ESO. "NGESO Open Licence". https://national-grid.ckan.io/licence (accessed on 26.03.2020)
* [6] Fingrid Oyj. "Frequency - historical data". https://data.fingrid.fi/en/dataset/frequency-historical-data (accessed on 26.03.2020)
* [7] Creative Commons. "Attribution 4.0 International (CC BY 4.0)". https://creativecommons.org/licenses/by/4.0/deed.en (accessed on 26.03.2020)

