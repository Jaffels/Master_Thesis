# Pre-Processed Power Grid Frequency Time Series (2020-2023)

## Overview

This dataset provides pre-processed frequency time series data for 2020-2023, covering three synchronous areas of the European power grid:
* Continental Europe
* Nordic

This work is part of the paper "Probabilistic and Explainable Machine Learning for Tabular Power Grid Data"[1]. Please cite this paper, when using the data and the code.

## Relationship to Previous Work

This dataset extends the time coverage of the original dataset [2], which covered 2012-2021. For a detailed documentation of the pre-processing procedure we refer to the supplementary material of paper "Predictability of Power Grid Frequency"[3]. The same methodology and preprocessing procedures have been applied to maintain consistency and comparability with the original work.

## Data Sources

* Continental Europe [4]: To get the data just replace the year and month in the url with the needed one. In June 2022 the frequency data of Continental Europe was moved to [5]. The download and preprocessing scripts were adapted to the new data sources. To distinguish between both data sources, the data from the SG HoBA data [5] was stored in a separate folder. In order to keep the structure with one file per year, combine_2022_ce_data.ipynb notebook was used to combine the data from the first and second half of 2022 and the data was saved in the 2022_cleansed/TransnetBW/2022.zip folder.
* Nordic [6]: We obtained the data from the Finish TSO Fingrid, which provides the data under the open license CC-BY 4.0 [7]. 

## Content of the repository

### Scripts

1) In the `Download_scripts` folder you will find three scripts to automatically download frequency data from the TSO's websites.
2) In `convert_data_format.py` we save the data with corrected timestamp formats.
3) In `clean_corrupted_data.py` we load the converted data and identify corrupted recordings. We mark them as NaN and clean some of the resulting data holes (processing step (2) in the supplementary material of [3]).

The python scripts were adapted to run with Python 3.11 and with the packages found in `requirements.txt`.

## Yearly converted and cleansed data

The folder `Data_cleansed` contains the output of `clean_corrupted_data.py`.

* File type: The files are zipped csv-files, where each file comprises one year.
* Data format: The files contain two columns. The second column contains the frequency values in Hz. The first one represents the time stamps in the format *Year-Month-Day Hour-Minute-Second*, which is given as naive local time. The local time refers to the following time zones and includes Daylight Saving Times (python time zone in brackets):
    * TransnetBW: Continental European Time (*CET*)
    * Fingrid: Finland (*Europe/Helsinki*)
    * **NaN representation**: We mark corrupted and missing data as "NaN" in the csv-files.


## References
[1] A. Nikoltchovska, S. Pütz, X. Li, V. Hagenmeyer, and B. Schäfer, “Probabilistic and explainable machine learning for tabular power grid data,” in Proc. 16th ACM Int. Conf. Future and Sustainable Energy Systems (E-Energy '25), New York, NY, USA: ACM, 2025, pp. 213–231.
[2] J. Kruse, B. Schäferand D. Witthaut, “Pre-Processed Power Grid Frequency Time Series”. Zenodo, Jul. 15, 2021. doi: 10.5281/zenodo.5105820.
[3] J. Kruse, B. Schäfer, and D. Witthaut, “Predictability of Power Grid Frequency,” IEEE Access, vol. 8, pp. 149435–149446, 2020.
[4] TransnetBW GmbH. "Netzfrequenz". https://webservices.transnetbw.de/files/bis/netzfrequenz/202001_Frequenz.zip (accessed on 21.05.2024)
[5] Steuerungsgruppe Horizontaler Belastungsausgleich (SG HoBA). "Sekündliche Frequenz". https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Sek%C3%BCndliche-Daten  (accessed on 21.05.2024).
[6] Fingrid Oyj. "Frequency - historical data". https://data.fingrid.fi/en/datasets/177 (accessed on 21.05.2024)
[7] Creative Commons. "Attribution 4.0 International (CC BY 4.0)". https://creativecommons.org/licenses/by/4.0/deed.en (accessed on 26.03.2025)