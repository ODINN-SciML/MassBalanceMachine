# Data Processing

- ```Climate``` folder not included, as the data is too large to be on the repository. Data can be downloaded, with the right variables, timeframe, and coordinates, with the designated scripts. Be aware that it can take up to 2 hours to retrieve this data from the Copernicus website. 

## Code Organization

1. ### Data (Pre)Processing

   1. ```get_stake_measurements.py```
      For Iceland's three largest icecaps, stake data is stored in individual files, which are subsequently used to make API requests to retrieve their data. Each stake's data is then stored separately in a .csv file. Data is retrieved from: [icelandicglaciers.is](https://icelandicglaciers.is/#/page/map). The stakes are manually selected and can be found in the folder ```Stakes```. In the folder ```Misc```, the file ```stake_metadata.txt``` can be found, containing the metadata of the stake measurements.
   2. ```merg_stake_measurements.py``` Merge all stake measurement data for each icecap into individual `.csv` files.
   3. The glacier ID for each stake corresponds to a specific glacier. These  IDs are essential for the subsequent step, which involves retrieving  topographical features for each point measurement. To obtain these  glacier IDs, the [Randolph Glacier Inventory](https://www.glims.org/RGI/) (RGI) is utilized. The RGI  contains outlines of glaciers. By overlaying the shape file of Icelandic  glaciers with the point measurements in QGIS, the RGI IDs can be  associated with the respective locations. Subsequently, these layers are  exported as `.csv` files, with one file generated for each icecap. Files can be found in folder ```RGIIDV6```.
      1. This step has been automated. ```match_rgiids_wth_stake.py``` Matches the RGI IDs of the glaciers on the icecaps with the stake measurements only if their location is within the section that is associated with this RGI ID. Only a shape file is needed of the region of interest that can be obtained from [here](https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v6/). The outputs of this operation can be found in the ```Stake Measurements RGIId``` folder.
   4. ```merge_rgiids.py``` Merges the three separate ```.csv``` files, one for each icecap with for each stake a RGI ID, into a single file ```Iceland_Stake_Data_Merged.csv```.
   5. ```remove_nan_rgi_records.py``` Some stakes in the dataset lack an RGI ID assignment. These stakes are filtered out of the dataset and saved in a separate file ```Iceland_Stake_Data_Nan_Glaciers.csv```.
   6. ```get_ogggm_data.py``` The script in the ```OGGM``` folder, located outside the current working  directory, retrieves topographical features for each stake on the  glaciers. These features include slope, aspect, slope factor, and  distance from the border. This script is stored in a separate folder because it requires a remote environment, WSL, to run. For the future, the aim is to implement this as a separate step in the data pipeline without the need to do it manually e.g., to start the remote process and activate the Conda environment. The output file of this script is: ```Iceland_Stake_Data_T_Attributes.csv```.
   7. ```get_climate_data.py``` For each stake measurement, we retrieve two types of climate data from  the Copernicus Climate Data Store (CDS) spanning from 1950 to 2024. The  first type is the ERA5-Land monthly averaged climate variables,  accessible [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview).  These variables include '10m\_u\_component\_of\_wind', '10m\_v\_component\_of\_wind', '2m\_temperature', 'forecast\_albedo', 'snow\_albedo', 'snow\_cover', 'snow\_depth', 'snow\_depth\_water\_equivalent', 'surface\_latent\_heat\_flux', 'surface\_net\_solar\_radiation', 'surface\_net\_thermal\_radiation',  'surface\_sensible\_heat\_flux', 'surface\_solar\_radiation\_downwards', 'surface\_thermal\_radiation\_downwards', and 'total\_precipitation'. The second type is the ERA5-Land monthly averaged pressure variable, accessible [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview). Specifically, the geopential was retrieve 1000 hPa. All data is retrieved for the region of Iceland and covers each month from January to December for the specified years.The ouptut of this script is: ```Iceland_Stake_Data_Climate.csv```.
      1. Data from the CDS can be fetched by: ```get_ERA5_monthly_averaged_climate_data.py``` and ```get_ERA5_monlty_pressure_data.py```.
   8. ```data_to_wgms.py``` Transforms and refactor the data to the [WGMS](https://wgms.ch/) format, in case the data will ever be uploaded. The output of this script is: ```Iceland_Stake_WGMS.csv```.
   9. ```clean_data.py``` Remove unnecessary columns from the dataset, rename certain columns, and delete records without climate and altitude data. The ouput of this script is: ```Iceland_Stake_Data_Cleaned.csv```.

## TODO

* [ ]  Make a single Python file that includes all the Data (Pre)Processing steps, and executes them in order
* [ ]  Make it so that every file of the (Pre)Processing step takes arguments for the input and output file name
* [ ]  Make it that the OGGM step is automated in the process instead of a singular process
* [ ]  Make sure that geopotential for pressure data, with 1000hPa, is correct. Double check
* [X]  ~~Make it so that the rgi_ids can be matched with the stake data without the use of QGIS~~
* [ ]  Make sure the lon and lat are in the right coordinate system. Double check
