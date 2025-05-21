"""
Script to download all stake measurements from Icelandic glaciers,
including both regular and irregular stakes.
If API changes, use the JavaScript in the bottom comment code in the browser console to find all stake-related URLs:
"""

import os
import requests
import json
import csv
import time

# Directory to save all stake measurements
OUTPUT_DIR = '/home/mburlet/scratch/data/DATA_MB/WGMS/Iceland/data/all-stake-measurements'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters for stake measurements
PARAMS = "stake,yr,d1,d2,d3,lat,lon,elevation,dw,rhow,ice_melt_spring,ds,rhos,ice_melt_fall,nswe_fall,bw_stratigraphic,bs_stratigraphic,ba_stratigraphic,bw_floating_date,bs_floating_date,ba_floating_date"

def get_all_stake_ids():
    """Fetch the complete list of stakes from the API endpoints, both regular and irregular"""
    stake_ids = []
    
    # Get regular stakes (sporadic=false)
    url_regular = "https://geo.vedur.is/geoserver/wfs?typeName=glaciology%3Astakev&request=GetFeature&service=WFS&outputFormat=application%2Fjson&cql_filter=sporadic%3Dfalse"
    
    response = requests.get(url_regular)
    data = response.json()
    
    # Extract regular stake IDs
    regular_stakes = []
    for feature in data.get('features', []):
        properties = feature.get('properties', {})
        stake_id = properties.get('stake')
        if stake_id:
            regular_stakes.append(stake_id)
    
    print(f"Found {len(regular_stakes)} regular stakes")
    stake_ids.extend(regular_stakes)
    
    # Get irregular stakes (sporadic=true)
    url_irregular = "https://geo.vedur.is/geoserver/wfs?typeName=glaciology%3Astakev&request=GetFeature&service=WFS&outputFormat=application%2Fjson&cql_filter=sporadic%3Dtrue"
    
    response = requests.get(url_irregular)
    data = response.json()
    
    # Extract irregular stake IDs
    irregular_stakes = []
    for feature in data.get('features', []):
        properties = feature.get('properties', {})
        stake_id = properties.get('stake')
        if stake_id:
            irregular_stakes.append(stake_id)
    
    print(f"Found {len(irregular_stakes)} irregular stakes")
    stake_ids.extend(irregular_stakes)
    
    # Save stakes to files for reference
    with open(os.path.join(OUTPUT_DIR, 'regular_stakes.txt'), 'w') as f:
        for stake in regular_stakes:
            f.write(f"{stake}\n")
    
    with open(os.path.join(OUTPUT_DIR, 'irregular_stakes.txt'), 'w') as f:
        for stake in irregular_stakes:
            f.write(f"{stake}\n")
    
    print(f"Total: {len(stake_ids)} stakes")
    return stake_ids

def download_stake_data(stake_id):
    """Download measurement data for a specific stake and save as proper CSV"""
    url = f'https://api.vedur.is/glaciers/stake/{stake_id}/measurements?params={PARAMS}'
    
    response = requests.get(url)
    
    if response.status_code == 200 and len(response.text.strip()) > 100:
        # Parse the JSON response
        data = json.loads(response.text)
        
        if data and len(data) > 0:
            # Save in proper CSV format
            output_file = os.path.join(OUTPUT_DIR, f'{stake_id}.csv')
            
            # Get field names from the first object
            fieldnames = list(data[0].keys())
            
            # Write as proper CSV
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                
            print(f"✓ {stake_id}: {len(data)} records")
            return True
    
    print(f"✗ {stake_id}")
    return False

# Main execution
stake_ids = get_all_stake_ids()
success_count = 0

for stake_id in stake_ids:
    if download_stake_data(stake_id):
        success_count += 1
    time.sleep(0.2)

print(f"Download complete: {success_count}/{len(stake_ids)} stakes")


## The following in javascript code that was used to find all stake-related URLs in the browser console
# simply run this on a freshly loaded map, then activate stakes in the map and it shows url
"""
console.clear();
console.log("Finding all stake-related URLs...");

let resourceList = performance.getEntriesByType('resource');
let glacierUrls = [];

for (let i = 0; i < resourceList.length; i++) {
  let url = resourceList[i].name;
  if (url.includes('stake') || url.includes('glacier')) {
    glacierUrls.push(url);
    console.log(`${glacierUrls.length}: ${url}`);
  }
}

console.log("\nMonitoring for new network requests...");
console.log("Try interacting with the map to trigger more requests.");

let requestWatcher = new PerformanceObserver((list) => {
  list.getEntries().forEach(entry => {
    if (entry.name.includes('stake') || entry.name.includes('glacier')) {
      glacierUrls.push(entry.name);
      console.log(`New request found: ${entry.name}`);
    }
  });
});

requestWatcher.observe({entryTypes: ['resource']});

console.log(`Found ${glacierUrls.length} stake/glacier related URLs initially`);
console.log("Continue using the map - new requests will appear automatically");
"""