"""
This script retrieves the monthly averaged ERA5 climate data from the Copernicus website.
Variables, region, months, and years, of interest should be specified by the user. This
file is for the climate data in Iceland.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 04/06/2024
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        # Define the climate variables you want to have in your dataset as training feature
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'forecast_albedo', 'snow_albedo',
            'snow_cover', 'snow_density', 'snow_depth',
            'snow_depth_water_equivalent', 'snow_evaporation', 'snowfall',
            'snowmelt', 'surface_latent_heat_flux', 'surface_net_solar_radiation',
            'surface_net_thermal_radiation', 'surface_pressure', 'surface_sensible_heat_flux',
            'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards', 'temperature_of_snow_layer',
            'total_precipitation',
        ],
        # Define the years you want to have the climate data of
        'year': [
            '1950', '1951', '1952',
            '1953', '1954', '1955',
            '1956', '1957', '1958',
            '1959', '1960', '1961',
            '1962', '1963', '1964',
            '1965', '1966', '1967',
            '1968', '1969', '1970',
            '1971', '1972', '1973',
            '1974', '1975', '1976',
            '1977', '1978', '1979',
            '1980', '1981', '1982',
            '1983', '1984', '1985',
            '1986', '1987', '1988',
            '1989', '1990', '1991',
            '1992', '1993', '1994',
            '1995', '1996', '1997',
            '1998', '1999', '2000',
            '2001', '2002', '2003',
            '2004', '2005', '2006',
            '2007', '2008', '2009',
            '2010', '2011', '2012',
            '2013', '2014', '2015',
            '2016', '2017', '2018',
            '2019', '2020', '2021',
            '2022', '2023', '2024',
        ],
        # Define what months you want to have the average data of
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        # Define the coordinates of the region of interest. Order of the coordinates: latitude North, longitude West,
        # latitude South, longitude East
        'area': [
            66.5, -24.53, 63,
            -13,
        ],
        # Define the file format you want to download
        'format': 'netcdf.zip',
    },
    # Define the output directory and the filename of the downloaded data
    '.././data/climate/ERA5_monthly_averaged_climate_data.zip')
