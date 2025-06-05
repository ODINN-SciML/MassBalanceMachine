import pandas as pd
import numpy as np
from datetime import datetime
import geopandas as gpd
import pyproj
import xarray as xr
import zipfile
import logging
from tqdm import tqdm
from itertools import combinations
from geopy.distance import geodesic
from oggm import utils, workflow, tasks
from oggm import cfg as oggmCfg
from pathlib import Path
from shapely.geometry import Point


from scripts.wgs84_ch1903 import *
from scripts.config_FR import *
from scripts.helpers import *

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Preprocess --- #

def extract_glacioclim_files(path_PMB_GLACIOCLIM_raw):
    """
    Extract GLACIOCLIM zipfiles into organized directories.
    """
    path_PMB_GLACIOCLIM_raw = Path(path_PMB_GLACIOCLIM_raw)
    glacioclim_dir = path_PMB_GLACIOCLIM_raw.parent

    seasons = ['annual', 'summer', 'winter']

    path_PMB_GLACIOCLIM_raw.mkdir(parents=True, exist_ok=True)

    emptyfolder(path_PMB_GLACIOCLIM_raw)

    for glacier_dir in glacioclim_dir.glob('*Glacier*'):
        glacier_name = glacier_dir.name
        print(f"\nProcessing {glacier_name}")
        
        for season in seasons:
            season_dir = glacier_dir / season
            if not season_dir.exists():
                print(f"  Skipping {season} - directory not found")
                continue
                
            zip_files = list(season_dir.glob('*.zip'))
            print(f"  Found {len(zip_files)} zip files in {season}")
            
            for zip_path in zip_files:
                extract_dir = path_PMB_GLACIOCLIM_raw / glacier_name / season / zip_path.stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                        print(f"    Extracted {zip_path.name} to {extract_dir}")
                except Exception as e:
                    print(f"    Error extracting {zip_path.name}: {str(e)}")

def format_date(date_str):
    """
    Convert date strings like '3. 10. 2004' to 'yyyymmdd' format
    """
    date_str = str(date_str).strip()
    
    # Split by dots and remove spaces
    parts = [p.strip() for p in date_str.split('.') if p.strip()]
    
    # Extract parts and add leading zeros where needed
    day = parts[0].zfill(2)
    month = parts[1].zfill(2)
    year = parts[2]
    
    # Return in yyyymmdd format
    return f"{year}{month}{day}"

def extract_sarennes_data(all_sheets):
    """
    Extract winter, summer, and annual mass balance data from Sarennes Excel sheets
    Each sheet corresponds to a year and contains multiple stake measurements
    """
    sarennes_dfs = {}
    
    # Process each sheet (year)
    for sheet_name, df in all_sheets.items():
            
        year = sheet_name
        
        # Find rows with end_dates (these rows contain stake measurements)
        end_date_rows = df[(pd.notna(df['Unnamed: 13'])) & (df['Unnamed: 13'] != 0)]
            
        # Process each season (winter, summer, annual)
        for season_idx, season in enumerate(['winter', 'summer', 'annual']):
            # Column mapping for balance values
            balance_col = f'Unnamed: {17 + season_idx}'
            
            # Date columns mapping based on season
            if season == 'winter':
                from_date_col = 'Unnamed: 11'  # start date
                to_date_col = 'Unnamed: 12'    # spring date
            elif season == 'summer':
                from_date_col = 'Unnamed: 12'  # spring date
                to_date_col = 'Unnamed: 13'    # end date
            else:  # annual
                from_date_col = 'Unnamed: 11'  # start date
                to_date_col = 'Unnamed: 13'    # end date
            
            # Create rows for this season
            rows = []
            for i, (_, row) in enumerate(end_date_rows.iterrows(), 1):                   
                # Skip if no balance value
                balance_value = row.get(balance_col)
                if pd.isna(balance_value) or balance_value == 0:
                    continue
                
                # Create data row
                rows.append({
                    'POINT_ID': f'sarennes_complete_{season}_{year}_{i}',
                    'x_lambert3': row['Unnamed: 14'],
                    'y_lambert3': row['Unnamed: 15'],
                    'POINT_ELEVATION': row['Unnamed: 16'],
                    'FROM_DATE': format_date(row.get(from_date_col)),
                    'TO_DATE': format_date(row.get(to_date_col)),
                    'POINT_BALANCE': row[balance_col],
                    'GLACIER': 'sarennes',
                    'PERIOD': season,
                    'GLACIER_ZONE': 'complete'
                })
            
            # Create DataFrame if we have rows
            if rows:
                key = f'sarennes_complete_{season}_{year}'
                sarennes_dfs[key] = pd.DataFrame(rows)
    
    return sarennes_dfs


def lamberttoWGS84(df, lambert_type="III"):
    """Converts from x & y Lambert III (EPSG:27563) or Lambert II (EPSG:27562) to lat/lon WGS84 (EPSG:4326) coordinate system
    """

    if lambert_type == "II":
        transformer = pyproj.Transformer.from_crs("EPSG:27562",
                                              "EPSG:4326",
                                              always_xy=True)
    else:
        transformer = pyproj.Transformer.from_crs("EPSG:27563",
                                              "EPSG:4326",
                                              always_xy=True)

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_lambert3, df.y_lambert3)

    df['lat'] = latitude
    df['lon'] = lon
    df.drop(['x_lambert3', 'y_lambert3'], axis=1, inplace=True)
    return df

def lambert_transform(df):
    """
    Transform coordinates from Lambert to WGS84

    Columns are named differently depending on the glacier
    """
    transformed_df = df.copy()
    
    # Coordinate transformation
    for key, value in transformed_df.items():
        if key.startswith(('mdg', 'Argentiere')):
            value = value.rename(columns={
                    'x_lambert2e': 'x_lambert3',
                    'y_lambert2e': 'y_lambert3'
                })
            transformed_df[key] = lamberttoWGS84(value,'II')
        # 3 years in the Saint-Sorlin are falsely named lambert2.
        elif key == 'stso_winter_smb_accu_2019' or 'stso_winter_smb_accu_2019' or 'stso_winter_smb_accu_2019':
            value = value.rename(columns={
                        'x_lambert2e': 'x_lambert3',
                        'y_lambert2e': 'y_lambert3'
                    })
            transformed_df[key] = lamberttoWGS84(value)
        else:
            transformed_df[key] = lamberttoWGS84(value)
        
        lat_check = value['lat'].between(45, 46).all()
        lon_check = value['lon'].between(6, 7.5).all()
        
        if not (lat_check and lon_check):
            print(f"\nWarning for {key}:")
            if not lat_check:
                print(f"Latitude range: {value['lat'].min():.4f} to {value['lat'].max():.4f}")
            if not lon_check:
                print(f"Longitude range: {value['lon'].min():.4f} to {value['lon'].max():.4f}")
    return transformed_df

def transform_WGMS_df(df, key):
    """
    Transform df into WGMS format
    """
    
    new_df = df.copy()
    
    # Extract glacier name (everything before first '_')
    glacier_name = key.split('_')[0]
    
    # Determine period and balance type from key
    if 'winter' in key:
        period = 'winter'
        balance_col = 'winter_smb'
    elif 'summer' in key:
        period = 'summer'
        balance_col = 'summer_smb'
    elif 'annual' in key:
        period = 'annual'
        balance_col = 'annual_smb'
    else:
        print('ERROR')
    
    # Create POINT_ID
    new_df['POINT_ID'] = key + '_' + 'setup' + new_df['stake_year_setup'].astype(str) + '_' + new_df['stake_number'].astype(str)
    
    # Create dates
    new_df['FROM_DATE'] = (new_df['year_start'].astype(str) + 
                          new_df['month_start'].astype(str).str.zfill(2) + 
                          new_df['day_start'].astype(str).str.zfill(2)).astype(int)
    
    new_df['TO_DATE'] = (new_df['year_end'].astype(str) + 
                         new_df['month_end'].astype(str).str.zfill(2) + 
                         new_df['day_end'].astype(str).str.zfill(2)).astype(int)
    
    
    # Create final DataFrame with required columns
    final_df = pd.DataFrame({
        'POINT_ID': new_df['POINT_ID'],
        'POINT_LAT': new_df['lat'],
        'POINT_LON': new_df['lon'],
        'POINT_ELEVATION': new_df['altitude'],
        'FROM_DATE': new_df['FROM_DATE'],
        'TO_DATE': new_df['TO_DATE'],
        'POINT_BALANCE': new_df[balance_col],
        'GLACIER': glacier_name,
        'PERIOD': period,
        'GLACIER_ZONE': new_df['profile_name']
    })
    
    return final_df

def find_close_stakes(df, distance_threshold=10):
    """
    Find stakes that are within distance_threshold meters of each other.
    Only compares stakes within the same glacier, year, and period.
    """
    close_stakes = []
    
    # Group by glacier, year, AND period
    for (glacier, year, period), group in tqdm(df.groupby(['GLACIER', 'YEAR', 'PERIOD']), 
                                              desc='Processing glacier-year-periods'):
        # Skip if there's only one stake in this group
        if len(group) <= 1:
            continue
            
        # Get all unique pairs of stakes in this glacier-year-period
        for idx1, idx2 in combinations(group.index, 2):
            # Get coordinates
            point1 = (group.loc[idx1, 'POINT_LAT'], group.loc[idx1, 'POINT_LON'])
            point2 = (group.loc[idx2, 'POINT_LAT'], group.loc[idx2, 'POINT_LON'])
            
            # Calculate distance in meters
            distance = geodesic(point1, point2).meters
            
            # Check if within threshold
            if distance <= distance_threshold:
                close_stakes.append({
                    'GLACIER': glacier,
                    'YEAR': year,
                    'PERIOD': period,
                    'POINT_ID_1': group.loc[idx1, 'POINT_ID'],
                    'POINT_ID_2': group.loc[idx2, 'POINT_ID'],
                    'LAT_1': point1[0],
                    'LON_1': point1[1],
                    'LAT_2': point2[0],
                    'LON_2': point2[1],
                    'POINT_BALANCE_1': group.loc[idx1, 'POINT_BALANCE'],
                    'POINT_BALANCE_2': group.loc[idx2, 'POINT_BALANCE'],
                    'DISTANCE_M': distance
                })
    
    # Convert to DataFrame
    if close_stakes:
        result_df = pd.DataFrame(close_stakes)
        print(f"Found {len(result_df)} pairs of stakes that are {distance_threshold}m or closer")
        return result_df
    else:
        print(f"No stakes found within {distance_threshold}m of each other")
        # Return empty df if triggered
        return pd.DataFrame(columns=['GLACIER', 'YEAR', 'PERIOD', 'POINT_ID_1', 'POINT_ID_2', 
                                     'LAT_1', 'LON_1', 'LAT_2', 'LON_2', 
                                     'POINT_BALANCE_1', 'POINT_BALANCE_2', 'DISTANCE_M'])

def merge_close_stakes(df, close_stakes_df, distance_threshold=10):
    """
    Merges stakes that are within distance_threshold meters of each other.
    For each pair, it averages the balances and keeps the first point.
    Drops the second point from the DataFrame.
    """
    df_test = df.copy()
    
    # Sort close_stakes_df by distance to process closest pairs first
    close_stakes_df = close_stakes_df.sort_values('DISTANCE_M')
    
    # Keep track of points that have been dropped
    dropped_points = set()
    
    # Process each pair of close stakes
    for _, row in tqdm(close_stakes_df.iterrows(), desc=f'Merging stakes within {distance_threshold}m'):
        point_id_1 = row['POINT_ID_1']
        point_id_2 = row['POINT_ID_2']
        
        # Skip if either point has already been dropped
        if point_id_2 in dropped_points:
            continue
            
        # Calculate average balance
        avg_balance = (row['POINT_BALANCE_1'] + row['POINT_BALANCE_2']) / 2
        
        # Update the balance of the first point and set MODIFICATION Flag
        mask_1 = df_test['POINT_ID'] == point_id_1
        if mask_1.any():
            df_test.loc[mask_1, 'POINT_BALANCE'] = avg_balance
            df_test.loc[mask_1, 'DATA_MODIFICATION'] = 'Merged stake with ID ' + point_id_2 + ' by taking average mb and dropping second stake'
            
            # Drop the second point
            mask_2 = df_test['POINT_ID'] == point_id_2
            if mask_2.any():
                df_test = df_test[~mask_2]
                dropped_points.add(point_id_2)
                
    print(f"Merged {len(dropped_points)} pairs of close stakes")
    return df_test

def check_period_consistency(df):
    """
    Checks if date ranges make sense for annual and winter periods:
    Returns dataframes with inconsistent periods
    """
    df_check = df.copy()
    
    # Convert dates to datetime objects
    df_check['FROM_DATE_DT'] = pd.to_datetime(df_check['FROM_DATE'], format='%Y%m%d')
    df_check['TO_DATE_DT'] = pd.to_datetime(df_check['TO_DATE'], format='%Y%m%d')
    
    df_check['MONTH_DIFF'] = ((df_check['TO_DATE_DT'].dt.year - df_check['FROM_DATE_DT'].dt.year) * 12 + 
                             df_check['TO_DATE_DT'].dt.month - df_check['FROM_DATE_DT'].dt.month)
    
    # Identify inconsistent periods
    ## 9-15 and 4-9 and 3-8 excludes the normal varying range
    annual_inconsistent = df_check[(df_check['PERIOD'] == 'annual') & 
                                 ((df_check['MONTH_DIFF'] < 9) | (df_check['MONTH_DIFF'] > 15))]
    
    winter_inconsistent = df_check[(df_check['PERIOD'] == 'winter') & 
                                 ((df_check['MONTH_DIFF'] < 4) | (df_check['MONTH_DIFF'] > 9))]
    
    summer_inconsistent = df_check[(df_check['PERIOD'] == 'summer') & 
                                 ((df_check['MONTH_DIFF'] < 3) | (df_check['MONTH_DIFF'] > 8))]
    
    

    total_annual = len(df_check[df_check['PERIOD'] == 'annual'])
    total_winter = len(df_check[df_check['PERIOD'] == 'winter'])
    total_summer = len(df_check[df_check['PERIOD'] == 'summer'])
    
    print(f"Annual periods: {len(annual_inconsistent)} out of {total_annual} ({len(annual_inconsistent)/total_annual*100:.1f}%) are inconsistent")
    print(f"Winter periods: {len(winter_inconsistent)} out of {total_winter} ({len(winter_inconsistent)/total_winter*100:.1f}%) are inconsistent")
    print(f"Summer periods: {len(summer_inconsistent)} out of {total_summer} ({len(summer_inconsistent)/total_summer*100:.1f}%) are inconsistent")
    
    return annual_inconsistent, winter_inconsistent, summer_inconsistent

# --- OGGM --- #

def initialize_oggm_glacier_directories(
    working_dir = path_OGGM,
    rgi_region="11",
    rgi_version="6",
    base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/",
    log_level='WARNING',
    task_list=None,
):
    # Initialize OGGM config
    oggmCfg.initialize(logging_level=log_level)
    oggmCfg.PARAMS['border'] = 10
    oggmCfg.PARAMS['use_multiprocessing'] = True
    oggmCfg.PARAMS['continue_on_error'] = True

    # Module logger
    log = logging.getLogger('.'.join(__name__.split('.')[:-1]))
    log.setLevel(log_level)

    # Set working directory
    oggmCfg.PATHS['working_dir'] = working_dir

    # Get RGI file
    rgi_dir = utils.get_rgi_dir(version=rgi_version)
    path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
    rgidf = gpd.read_file(path)

    # Initialize glacier directories from preprocessed data
    gdirs = workflow.init_glacier_directories(
        rgidf,
        from_prepro_level=3,
        prepro_base_url=base_url,
        prepro_border=10,
        reset=True,
        force=True,
    )

    # Default task list if none provided
    if task_list is None:
        task_list = [
            tasks.gridded_attributes,
            # tasks.gridded_mb_attributes,
            # get_gridded_features,
        ]

    # Run tasks
    for task in task_list:
        workflow.execute_entity_task(task, gdirs, print_log=False)

    return gdirs, rgidf


def export_oggm_grids(gdirs,
                      subset_rgis=None,
                      output_path=path_OGGM_xrgrids):

    # Save OGGM xr for all needed glaciers:
    emptyfolder(output_path)
    for gdir in gdirs:
        RGIId = gdir.rgi_id
        # only save a subset if it's not empty
        if subset_rgis is not None:
            # check if the glacier is in the subset
            # if not, skip it
            if RGIId not in subset_rgis:
                continue
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()
        # save ds
        ds.to_zarr(os.path.join(output_path, f'{RGIId}.zarr'))


def merge_pmb_with_oggm_data(df_pmb,
                             gdirs,
                             rgi_region="11",
                             rgi_version="6",
                             variables_of_interest=None,
                             verbose=True):
    if variables_of_interest is None:
        variables_of_interest = [
            "aspect",
            "slope",
            "topo",
            "hugonnet_dhdt",
            "consensus_ice_thickness",
            "millan_v",
        ]
        # other options: "millan_ice_thickness", "millan_vx", "millan_vy", "dis_from_border"

    # Load RGI shapefile
    path = utils.get_rgi_region_file(region=rgi_region, version=rgi_version)
    rgidf = gpd.read_file(path)

    # Initialize empty columns
    for var in variables_of_interest:
        df_pmb[var] = np.nan
    df_pmb['within_glacier_shape'] = False

    grouped = df_pmb.groupby("RGIId")

    for rgi_id, group in grouped:
        # Find corresponding glacier directory
        gdir = next((gd for gd in gdirs if gd.rgi_id == rgi_id), None)
        if gdir is None:
            if verbose:
                log.error(
                    f"Warning: No glacier directory for RGIId {rgi_id}, skipping..."
                )
            continue

        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()

        # Match RGI shape
        glacier_shape = rgidf[rgidf["RGIId"] == rgi_id]
        if glacier_shape.empty:
            if verbose:
                log.error(
                    f"Warning: No shape found for RGIId {rgi_id}, skipping...")
            continue

        # Coordinate transformation from WGS84 to the projection of OGGM data
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),
            pyproj.CRS.from_user_input(ds.pyproj_srs),
            always_xy=True)
        lon, lat = group["POINT_LON"].values, group["POINT_LAT"].values
        x_stake, y_stake = transf.transform(lon, lat)

        # Create GeoDataFrame of points
        geometry = [Point(xy) for xy in zip(lon, lat)]
        points_rgi = gpd.GeoDataFrame(group,
                                      geometry=geometry,
                                      crs="EPSG:4326")

        # Intersect with glacier shape
        glacier_shape = glacier_shape.to_crs(points_rgi.crs)
        points_in_glacier = gpd.sjoin(points_rgi.loc[group.index],
                                      glacier_shape,
                                      predicate="within",
                                      how="inner")

        # Get nearest OGGM grid data for points
        stake = ds.sel(x=xr.DataArray(x_stake, dims="points"),
                       y=xr.DataArray(y_stake, dims="points"),
                       method="nearest")
        stake_var_df = stake[variables_of_interest].to_dataframe()

        # Assign to original DataFrame
        for var in variables_of_interest:
            df_pmb.loc[group.index, var] = stake_var_df[var].values

        df_pmb.loc[points_in_glacier.index, 'within_glacier_shape'] = True

    # Convert radians to degrees
    df_pmb['aspect'] = df_pmb['aspect'].apply(lambda x: math.degrees(x)
                                              if not pd.isna(x) else x)
    df_pmb['slope'] = df_pmb['slope'].apply(lambda x: math.degrees(x)
                                            if not pd.isna(x) else x)

    if verbose:
        log.info('-- Number of winter and annual samples:', len(df_pmb))
        log.info('-- Number of annual samples:',
                 len(df_pmb[df_pmb.PERIOD == 'annual']))
        log.info('-- Number of winter samples:',
                 len(df_pmb[df_pmb.PERIOD == 'winter']))

    return df_pmb