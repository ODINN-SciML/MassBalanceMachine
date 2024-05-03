import cdsapi  # Importing the cdsapi module

# Creating a Client object
c = cdsapi.Client()

# Retrieving data from the Copernicus Climate Data Store (CDS)
c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',  # Dataset name
    {
        'product_type': 'monthly_averaged_reanalysis',  # Product type
        'variable': 'geopotential',  # Variable
        'pressure_level': '1000',  # Pressure level
        'year': [  # List of years
            '1940', '1941', '1942',
            '1943', '1944', '1945',
            '1946', '1947', '1948',
            '1949', '1950', '1951',
            '1952', '1953', '1954',
            '1955', '1956', '1957',
            '1958', '1959', '1960',
            '1961', '1962', '1963',
            '1964', '1965', '1966',
            '1967', '1968', '1969',
            '1970', '1971', '1972',
            '1973', '1974', '1975',
            '1976', '1977', '1978',
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022', '2023',
            '2024',
        ],
        'month': [  # List of months
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',  # Time
        'area': [  # Geographic area
            66.5, -24.53, 63,
            -13,
        ],
        'format': 'netcdf',  # File format
    },
    'download.nc'  # Output file name
)
