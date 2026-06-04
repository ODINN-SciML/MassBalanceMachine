import os, sys
import requests
import json
import csv
import time
import urllib3
import tqdm

from data_processing.product_utils import mbm_path

iceland_path = os.path.join(mbm_path, ".data/stakes/iceland/")


"""
This script downloads stakes data from https://joklavefsja.vedur.is/ through API

## The following javascript code was used to find all stake-related URLs in the browser console
# simply run this on a freshly loaded map at https://joklavefsja.vedur.is/, then activate stakes in the map and it shows the stake URLs

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


PARAMS = "stake,yr,d1,d2,d3,lat,lon,elevation,dw,rhow,ice_melt_spring,ds,rhos,ice_melt_fall,nswe_fall,bw_stratigraphic,bs_stratigraphic,ba_stratigraphic,bw_floating_date,bs_floating_date,ba_floating_date"


def get_all_stake_ids():
    """Fetch the complete list of stakes from the API endpoints, both regular and irregular"""
    stake_ids = []

    # Get regular stakes (sporadic=false)
    url_regular = "https://geo.vedur.is/geoserver/wfs?typeName=glaciology%3Astakev&request=GetFeature&service=WFS&outputFormat=application%2Fjson&cql_filter=sporadic%3Dfalse"

    response = requests.get(url_regular, verify=False)
    data = response.json()

    # Extract regular stake IDs
    regular_stakes = []
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        stake_id = properties.get("stake")
        if stake_id:
            regular_stakes.append(stake_id)

    print(f"Found {len(regular_stakes)} regular stakes")
    stake_ids.extend(regular_stakes)

    # Get irregular stakes (sporadic=true)
    url_irregular = "https://geo.vedur.is/geoserver/wfs?typeName=glaciology%3Astakev&request=GetFeature&service=WFS&outputFormat=application%2Fjson&cql_filter=sporadic%3Dtrue"

    response = requests.get(url_irregular, verify=False)
    data = response.json()

    # Extract irregular stake IDs
    irregular_stakes = []
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        stake_id = properties.get("stake")
        if stake_id:
            irregular_stakes.append(stake_id)

    print(f"Found {len(irregular_stakes)} irregular stakes")
    stake_ids.extend(irregular_stakes)

    print(f"Total: {len(stake_ids)} stakes")
    return stake_ids


def download_stake_data(stake_id, data_path):
    """Download measurement data for a specific stake and save as proper CSV"""
    url = f"https://api.vedur.is/glaciers/stake/{stake_id}/measurements?params={PARAMS}"

    response = requests.get(url, verify=False)

    if response.status_code == 200 and len(response.text.strip()) > 100:
        data = json.loads(response.text)

        if data and len(data) > 0:
            output_file = os.path.join(data_path, f"{stake_id}.csv")

            fieldnames = list(data[0].keys())

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            return True, len(data)

    return False, 0


def download_all_stakes_data():
    print("Downloading Iceland stakes data")
    os.makedirs(iceland_path, exist_ok=True)

    urllib3.disable_warnings()
    stake_ids = get_all_stake_ids()
    success_count = 0

    pbar = tqdm.tqdm(stake_ids, total=len(stake_ids))
    for stake_id in pbar:
        success, ndata = download_stake_data(stake_id, iceland_path)
        if success:
            pbar.set_description(f"✓ {stake_id}: {ndata} records", refresh=True)
            success_count += 1
        else:
            pbar.set_description(f"✗ {stake_id}", refresh=True)
        time.sleep(0.2)

    print(f"Download complete: {success_count}/{len(stake_ids)} stakes")
