# fog: Fluctuations of Glaciers Database

- `description` Internationally collected, standardized dataset on changes in glaciers (length, area, volume, mass) based on in-situ and remotely sensed observations, as well as on reconstructions.
- `version` 2023-09
- `created` 2023-11-03
- `id` https://doi.org/10.5904/wgms-fog-2023-09
- `homepage` https://wgms.ch/data_databaseversions
- `publisher` World Glacier Monitoring Service (WGMS)
- `spatialCoverage` Global
- `temporalCoverage` 1127/2022
- `usageTerms` Open access under the requirement of correct citation
- `citation` WGMS (2023): Fluctuations of Glaciers Database. World Glacier Monitoring Service (WGMS), Zurich, Switzerland. https://doi.org/10.5904/wgms-fog-2023-09
- `publications` WGMS (2023): Global Glacier Change Bulletin No. 5 (2020-2021). Michael Zemp, Isabelle Gärtner-Roer, Samuel U. Nussbaumer, Ethan Z. Welty, Inès Dussaillant, and Jacqueline Bannwart (eds.), ISC (WDS) / IUGG (IACS) / UNEP / UNESCO / WMO, World Glacier Monitoring Service, Zurich, Switzerland, 134 pp. Based on database version https://doi.org/10.5904/wgms-fog-2023-09.

  WGMS (2013): Glacier Mass Balance Bulletin No. 12 (2010-2011). Michael Zemp, Samuel U. Nussbaumer, Kathrin Naegeli, Isabelle Gärtner-Roer, Frank Paul, Martin Hoelzle, and Wilfried Haeberli (eds.), ICSU (WDS) / IUGG (IACS) / UNEP / UNESCO / WMO, World Glacier Monitoring Service, Zurich, Switzerland, 106 pp. Based on database version https://doi.org/10.5904/wgms-fog-2013-11.

  WGMS (2012): Fluctuations of Glaciers 2005-2010 (Vol. X): Michael Zemp, Holger Frey, Isabelle Gärtner-Roer, Samuel U. Nussbaumer, Martin Hoelzle, Frank Paul, and Wilfried Haeberli (eds.), ICSU (WDS) / IUGG (IACS) / UNEP / UNESCO / WMO, World Glacier Monitoring Service, Zurich, Switzerland. Based on database version https://doi.org/10.5904/wgms-fog-2012-11.

  ... and earlier issues (https://wgms.ch/literature_published_by_wgms)
- `contributors` WGMS scientific collaboration network of national correspondents and principal investigators as listed in the data (`INVESTIGATOR` column) and related publications.
- `disclaimer` The data may contain errors and inaccuracies. Hence, we strongly suggest performing data quality checks and, in case of ambiguities, to contact us as well as the investigators and institutions listed in the data (`INVESTIGATOR` and `SPONS_AGENCY` columns) and related publications.
- `languages` ['en']

## `GLACIER`

General (and presumably static) information about each glacier. When submitting a new glacier, assign a temporary `WGMS_ID` and use this as the `WGMS_ID` in all other table rows that correspond to this glacier.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. A list of codes is available at https://www.iso.org/obp/ui/#search/code.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True
    - `enum`: ['AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO', 'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM', 'CA', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 'KM', 'CD', 'CG', 'CK', 'CR', 'CI', 'HR', 'CU', 'CW', 'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'EC', 'EG', 'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO', 'MK', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 'PT', 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH', 'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'UM', 'US', 'UY', 'UZ', 'VU', 'VE', 'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW']

### `NAME`

  - `description`: The name of the glacier, written in capital letters (A-Z).

    In order to ensure global interoperability of our dataset, glacier names should only contain the following characters: A-Z (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z), 0-9 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), - (dash), . (period), : (colon), () (parentheses), / (forward slash), ' (apostrophe), and  (space). Characters which do not fall into the given range should be transliterated. If no Latin name exists, use the International Organization for Standardization (ISO) standards for transliteration (https://www.iso.org/ics/01.140.10/x/). If the Latin name contains accents, apply the following rules (Å → AA, Æ → AE, Ä → AE, ð → D, Ø → OE, œ → OE, Ö → OE, ß → SS, þ → TH, Ü → UE) and neglect any remaining accents.

    If a name is too long, a meaningful abbreviation should be used. In this case, the full name should be listed in `REMARKS`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True
    - `maxLength`: 60
    - `pattern`: `[0-9A-Z\-\.:\(\)/\'\+&,\*=_]+( [0-9A-Z\-\.:\(\)/\'\+&,\*=_]+)*`

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `GEN_LOCATION`

  - `description`: Refers to a large geographic entity (e.g. a large mountain range or large political subdivision) which gives a rough idea of the location of the glacier, without requiring the use of a map or an atlas. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Western Alps
  - `type`: string
  - `constraints`:
    - `maxLength`: 30
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPEC_LOCATION`

  - `description`: Refers to a more specific geographic location (e.g. a drainage basin or mountain subrange), which can be found easily on a small scale map of the country. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Rhone Basin
  - `type`: string
  - `constraints`:
    - `maxLength`: 30
    - `pattern`: `[^\s]+( [^\s]+)*`

### `LATITUDE`

  - `description`: Latitude in decimal degrees (°, WGS 84). Positive values indicate the northern hemisphere and negative values indicate the southern hemisphere. The point (`LATITUDE`, `LONGITUDE`) should be in the main channel in the upper part of the glacier ablation area.
  - `example`: 45.9926
  - `type`: number
  - `constraints`:
    - `required`: True
    - `minimum`: -90
    - `maximum`: 90

### `LONGITUDE`

  - `description`: Longitude in decimal degrees (°, WGS 84). Positive values indicate east of the zero meridian and negative values indicate west of the zero meridian. The point (`LATITUDE`, `LONGITUDE`) should be in the main channel in the upper part of the glacier ablation area.
  - `example`: 7.8803
  - `type`: number
  - `constraints`:
    - `required`: True
    - `minimum`: -180
    - `maximum`: 180

### `PRIM_CLASSIFIC`

  - `description`: Glacier primary classification per [*Perennial ice and snow masses* (UNESCO/IAHS, 1970)](https://www.wgms.ch/downloads/UNESCO_1970.pdf):

      - 0 (Other): Any type not listed below (please explain in `REMARKS`).
      - 1 (Continental ice sheet): Inundates areas of continental size.
      - 2 (Icefield): Ice masses of sheet or blanket type of a thickness that is insufficient to obscure the subsurface topography.
      - 3 (Ice cap): Dome-shaped ice masses with radial flow.
      - 4 (Outlet glacier): Drains an ice sheet, icefield or ice cap, usually of valley glacier form. The catchment area may not be easily defined.
      - 5 (Valley glacier): Flows down a valley. The catchment area is well defined.
      - 6 (Mountain glacier): Cirque, niche or crater type, hanging glacier. Includes ice aprons and groups of small units.
      - 7 (Glacieret and snowfield): Small ice masses of indefinite shape in hollows, river beds and on protected slopes, which has developed from snow drifting, avalanching, and/or particularly heavy accumulation in certain years. Usually no marked flow pattern is visible. In existence for at least two consecutive years.
      - 8 (Ice shelf): Floating ice sheet of considerable thickness attached to a coast nourished by a glacier(s). Snow accumulation on its surface or bottom freezing.
      - 9 (Rock glacier): Lava-stream-like debris mass containing ice in several possible forms and moving slowly downslope.

    Note: `PARENT_GLACIER` can be used to classify complex glacier systems – for example, ice caps with outlet glaciers and glaciers splitting into multiple glaciers over time.
  - `example`: 5
  - `type`: integer
  - `constraints`:
    - `enum`: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

### `FORM`

  - `description`: Glacier form per [*Perennial ice and snow masses* (UNESCO/IAHS, 1970)](https://www.wgms.ch/downloads/UNESCO_1970.pdf):

      - 0 (Other): Any type not listed below (please explain in `REMARKS`).
      - 1 (Compound basins): Two or more individual valley glaciers issuing from tributary valleys and coalescing (Fig. 1a).
      - 2 (Compound basin): Two or more individual accumulation basins feeding one glacier system (Fig. 1b).
      - 3 (Simple basin): Single accumulation area (Fig. 1c).
      - 4 (Cirque): Occupies a separate, rounded, steep-walled recess which it has formed on a mountain side (Fig. 1d).
      - 5 (Niche): Small glacier in a V-shaped gulley or depression on a mountain slope (Fig. 1e). More common than a further-developed cirque glacier.
      - 6 (Crater): Occurring in extinct or dormant volcanic craters.
      - 7 (Ice apron): Irregular, usually thin ice mass which adheres to a mountain slope or ridge.
      - 8 (Group): A number of similar ice masses occurring in close proximity and too small to be assessed individually.
      - 9 (Remnant): Inactive, usually small ice masses left by a receding glacier.

    <!-- ![](figures/form.jpg) -->
  - `example`: 1
  - `type`: integer
  - `constraints`:
    - `enum`: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

### `FRONTAL_CHARS`

  - `description`: Glacier front characteristics per [*Perennial ice and snow masses* (UNESCO/IAHS, 1970)](https://www.wgms.ch/downloads/UNESCO_1970.pdf):

      - 0 (Other): Any type not listed below (please explain in `REMARKS`).
      - 1 (Piedmont): Icefield formed on a lowland area by lateral expansion of one or a coalescence of several glaciers (Fig. 2a, 2b).
      - 2 (Expanded foot): Lobe or fan formed where the lower portion of the glacier leaves the confining wall of a valley and extends onto a less restrictive and more level surface (Fig. 2c).
      - 3 (Lobed): Ice sheet or ice cap outlet glacier lacking a calving terminus (Fig. 2d).
      - 4 (Calving): Terminus of a glacier sufficiently extended into sea or lake water to produce icebergs. Includes - for this inventory - dry land calving which would be recognisable from the “lowest glacier elevation".
      - 5: Coalescing, non-contributing (Fig. 2e).
      - 6: Irregular, mainly clean ice (mountain or valley glaciers).
      - 7: Irregular, debris-covered (mountain or valley glaciers).
      - 8: Single lobe, mainly clean ice (mountain or valley glaciers).
      - 9: Single lobe, debris-covered (mountain or valley glaciers).

    <!-- ![](figures/frontal_chars.jpg) -->
  - `example`: 6
  - `type`: integer
  - `constraints`:
    - `enum`: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

### `EXPOS_ACC_AREA`

  - `description`: Main orientation of the accumulation area using an 8-point compass.
  - `example`: NW
  - `type`: string
  - `constraints`:
    - `enum`: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

### `EXPOS_ABL_AREA`

  - `description`: Main orientation of the ablation area using an 8-point compass.
  - `example`: W
  - `type`: string
  - `constraints`:
    - `enum`: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

### `PARENT_GLACIER`

  - `description`: Parent glacier `WGMS_ID`. Used to link glaciers to their (former) parent glacier.
  - `example`: 789
  - `type`: integer
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 999999

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `GLACIER_REGION_CODE`

  - `description`: First-order region code per [*Glacier Regions* (GTN-G, 2017)](https://doi.org/10.5904/gtng-glacreg-2017-07).
  - `example`: CEU
  - `type`: string
  - `constraints`:
    - `required`: True
    - `enum`: ['ACN', 'ACS', 'ALA', 'ANT', 'ASC', 'ASE', 'ASN', 'ASW', 'CAU', 'CEU', 'GRL', 'ISL', 'NZL', 'RUA', 'SAN', 'SCA', 'SJM', 'TRP', 'WNA']

### `GLACIER_SUBREGION_CODE`

  - `description`: Second-order region code per [*Glacier Regions* (GTN-G, 2017)](https://doi.org/10.5904/gtng-glacreg-2017-07).
  - `example`: CEU-01
  - `type`: string
  - `constraints`:
    - `required`: True
    - `enum`: ['ACN-01', 'ACN-02', 'ACN-03', 'ACN-04', 'ACN-05', 'ACN-06', 'ACN-07', 'ACS-01', 'ACS-02', 'ACS-03', 'ACS-04', 'ACS-05', 'ACS-06', 'ACS-07', 'ACS-08', 'ACS-09', 'ALA-01', 'ALA-02', 'ALA-03', 'ALA-04', 'ALA-05', 'ALA-06', 'ANT-01', 'ANT-02', 'ANT-03', 'ANT-04', 'ANT-05', 'ANT-11', 'ANT-12', 'ANT-13', 'ANT-14', 'ANT-15', 'ANT-16', 'ANT-17', 'ANT-18', 'ANT-19', 'ANT-20', 'ANT-21', 'ANT-22', 'ANT-23', 'ANT-24', 'ANT-31', 'ASC-01', 'ASC-02', 'ASC-03', 'ASC-04', 'ASC-05', 'ASC-06', 'ASC-07', 'ASC-08', 'ASC-09', 'ASE-01', 'ASE-02', 'ASE-03', 'ASN-01', 'ASN-02', 'ASN-03', 'ASN-04', 'ASN-05', 'ASN-06', 'ASN-07', 'ASW-01', 'ASW-02', 'ASW-03', 'CAU-01', 'CAU-02', 'CEU-01', 'CEU-02', 'GRL-01', 'GRL-11', 'ISL-01', 'NZL-01', 'RUA-01', 'RUA-02', 'RUA-03', 'SAN-01', 'SAN-02', 'SCA-01', 'SCA-02', 'SCA-03', 'SJM-01', 'SJM-02', 'TRP-01', 'TRP-02', 'TRP-03', 'TRP-04', 'WNA-01', 'WNA-02', 'WNA-03', 'WNA-04', 'WNA-05']

## `GLACIER_ID_LUT`

Links glaciers in this database (`GLACIER.WGMS_ID`) to glacier identifiers in other databases.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `PSFG_ID`

  - `description`: Glacier ID in the publications of the Permanent Service for the Fluctuations of Glaciers (PSFG), a predecessor of the WGMS. The ID was assigned by the national correspondents following existing glacier inventories. It consists of 6-7 characters: a 2-character political unit, a 4-character zero-padded integer, and an optional tag.
  - `example`: CH0016
  - `type`: string
  - `constraints`:
    - `minLength`: 6
    - `maxLength`: 7
    - `pattern`: `[A-Z]{2}[0-9]{4}[0-9A-Z]?`

### `WGI_ID`

  - `description`: Glacier ID in the World Glacier Inventory (https://nsidc.org/data/g01130/versions/1). The ID is constructed from the following elements:

    - 2-character political unit
    - 1-character continent code
    - 4-character drainage code
    - 2-character free position code
    - 3-character local glacier code
  - `example`: CH4N01356003
  - `type`: string
  - `constraints`:
    - `minLength`: 12
    - `maxLength`: 12
    - `pattern`: `[A-Z]{2}[1-7][0-9A-Z]{9}`

### `GLIMS_ID`

  - `description`: Glacier ID in the Global Land Ice Measurements from Space database (https://www.glims.org/MapsAndDocs/). The ID has the format `GxxxxxxEyyyyyΘ`, where `xxxxxx` is longitude east of the Greenwich meridian in millidegrees, `yyyyy` is north or south latitude in millidegrees, and `Θ` is N or S depending on the hemisphere.
  - `example`: G007880E45990N
  - `type`: string
  - `constraints`:
    - `minLength`: 14
    - `maxLength`: 14
    - `pattern`: `G[0-9]{6}E[0-9]{5}[NS]`

### `RGI50_ID`

  - `description`: Glacier ID in the Randolph Glacier Inventory 5.0 (https://nsidc.org/data/nsidc-0770/versions/5). The ID has the format `RGI50-rr.nnnnn`, where `rr` is the first-order region (zero-padded), and `nnnnn` is an arbitrary numeric code (which is not necessarily the same across RGI versions).
  - `example`: RGI50-11.02773
  - `type`: string
  - `constraints`:
    - `minLength`: 14
    - `maxLength`: 14
    - `pattern`: `RGI50-[0-1][0-9].[0-9]{5}`

### `RGI60_ID`

  - `description`: Glacier ID in the Randolph Glacier Inventory 6.0 (https://nsidc.org/data/nsidc-0770/versions/6). The ID has the format `RGI60-rr.nnnnn`, where `rr` is the first-order region (zero-padded), and `nnnnn` is an arbitrary numeric code (which is not necessarily the same across RGI versions).
  - `example`: RGI60-11.02773
  - `type`: string
  - `constraints`:
    - `minLength`: 14
    - `maxLength`: 14
    - `pattern`: `RGI60-[0-1][0-9].[0-9]{5}`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `STATE`

Glacier length, area, and elevation range.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `YEAR`

  - `description`: Survey year.
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `maximum`: 2023
    - `required`: True

### `SURVEY_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19940906
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `HIGHEST_ELEVATION`

  - `description`: Highest elevation on the glacier (m).
  - `example`: 3370
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9000

### `MEDIAN_ELEVATION`

  - `description`: Elevation of the contour line (m) which cuts the glacier into two parts of equal area.
  - `example`: 2920
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9000

### `LOWEST_ELEVATION`

  - `description`: Lowest elevation on the glacier (m).
  - `example`: 2370
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9000

### `ELEVATION_UNC`

  - `description`: Estimated random error of reported elevations (m).
  - `example`: 10
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `LENGTH`

  - `description`: Maximum length of glacier (km) measured along the main flowline.
  - `example`: 6.2
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `LENGTH_UNC`

  - `description`: Estimated random error of reported length (km).
  - `example`: 0.005
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `AREA`

  - `description`: Glacier area (km²).
  - `example`: 2.55
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `AREA_UNC`

  - `description`: Estimated random error of reported area (km²).
  - `example`: 0.01
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `SURVEY_PLATFORM_METHOD`

  - `description`: Survey platform (first digit, lowercase):

      - t: Terrestrial
      - a: Airborne
      - s: Spaceborne
      - c: Combined (explain in `REMARKS`)
      - x: Unknown or other (explain in `REMARKS`)

    Survey method (second digit, uppercase):

      - R: Reconstructed (e.g. historical sources, geomorphic evidence, dating of moraines)
      - M: Derived from maps
      - G: Ground survey (e.g. GPS, tachymetry, tape measure)
      - P: Photogrammetry
      - L: Laser altimetry or scanning
      - Z: Radar altimetry or interferometry
      - C: Combined (explain in `REMARKS`)
      - X: Unknown or other (explain in `REMARKS`)
  - `example`: aP
  - `type`: string
  - `constraints`:
    - `enum`: ['tR', 'tM', 'tG', 'tP', 'tL', 'tZ', 'tC', 'tX', 'aR', 'aM', 'aG', 'aP', 'aL', 'aZ', 'aC', 'aX', 'sR', 'sM', 'sG', 'sP', 'sL', 'sZ', 'sC', 'sX', 'cR', 'cM', 'cG', 'cP', 'cL', 'cZ', 'cC', 'cX', 'xR', 'xM', 'xG', 'xP', 'xL', 'xZ', 'xC', 'xX']

### `INVESTIGATOR`

  - `description`: Names of the persons or agencies that performed the survey or processed the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Michael Zemp
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPONS_AGENCY`

  - `description`: Full name, abbreviation and address of the agencies that sponsored the survey or archived the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: World Glacier Monitoring Service (WGMS), University of Zurich, Wintherthurerstr. 190, 8057 Zurich, Switzerland
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REFERENCE`

  - `description`: References to publications related to the data or methods. Use a short format such as `Author et al. YYYY (URL)` if a canonical URL is available (e.g. https://doi.org/DOI). Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Author et al. YYYY (https://doi.org/DOI)
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `CHANGE`

Change in glacier thickness, area, and/or volume – typically from geodetic surveys.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `SURVEY_ID`

  - `description`: Numeric key identifying data records related to a specific glacier survey. This key is assigned by the WGMS in order to distinguish results from different surveys (and sources) for the same glacier and survey period.
  - `example`: 288
  - `type`: integer
  - `constraints`:
    - `minimum`: 1
    - `required`: True

### `YEAR`

  - `description`: Survey year.
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `maximum`: 2023
    - `required`: True

### `SURVEY_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19940906
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `REFERENCE_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19931002
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `LOWER_BOUND`

  - `description`: Lower boundary of the surface elevation band (m), or 9999 if referring to the entire glacier.
  - `example`: 2500
  - `type`: integer
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9999
    - `required`: True

### `UPPER_BOUND`

  - `description`: Upper boundary of the surface elevation band (m), or 9999 if referring to the entire glacier.
  - `example`: 2600
  - `type`: integer
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9999
    - `required`: True

### `AREA_SURVEY_YEAR`

  - `description`: Glacier area (km²) of the elevation band at the time of `SURVEY_DATE`.
  - `example`: 0.071
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `AREA_CHANGE`

  - `description`: Change in area (1000 m²) for the elevation band.
  - `example`: -19
  - `type`: number

### `AREA_CHANGE_UNC`

  - `description`: Estimated random error of `AREA_CHANGE` (1000 m²).
  - `example`: 0.1
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `THICKNESS_CHG`

  - `description`: Mean change in ice thickness (mm) for the elevation band.
  - `example`: -5976
  - `type`: number

### `THICKNESS_CHG_UNC`

  - `description`: Estimated random error of `THICKNESS_CHG` (mm).
  - `example`: 10
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `VOLUME_CHANGE`

  - `description`: Change in ice volume (1000 m³) for the elevation band.
  - `example`: -424
  - `type`: number

### `VOLUME_CHANGE_UNC`

  - `description`: Estimated random error of `VOLUME_CHANGE` (1000 m³).
  - `example`: 5
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `SD_PLATFORM_METHOD`

  - `description`: Survey platform (first digit, lowercase):

      - t: Terrestrial
      - a: Airborne
      - s: Spaceborne
      - c: Combined (explain in `REMARKS`)
      - x: Unknown or other (explain in `REMARKS`)

    Survey method (second digit, uppercase):

      - R: Reconstructed (e.g. historical sources, geomorphic evidence, dating of moraines)
      - M: Derived from maps
      - G: Ground survey (e.g. GPS, tachymetry, tape measure)
      - P: Photogrammetry
      - L: Laser altimetry or scanning
      - Z: Radar altimetry or interferometry
      - C: Combined (explain in `REMARKS`)
      - X: Unknown or other (explain in `REMARKS`)
  - `example`: aP
  - `type`: string
  - `constraints`:
    - `enum`: ['tR', 'tM', 'tG', 'tP', 'tL', 'tZ', 'tC', 'tX', 'aR', 'aM', 'aG', 'aP', 'aL', 'aZ', 'aC', 'aX', 'sR', 'sM', 'sG', 'sP', 'sL', 'sZ', 'sC', 'sX', 'cR', 'cM', 'cG', 'cP', 'cL', 'cZ', 'cC', 'cX', 'xR', 'xM', 'xG', 'xP', 'xL', 'xZ', 'xC', 'xX']

### `RD_PLATFORM_METHOD`

  - `description`: Survey platform (first digit, lowercase):

      - t: Terrestrial
      - a: Airborne
      - s: Spaceborne
      - c: Combined (explain in `REMARKS`)
      - x: Unknown or other (explain in `REMARKS`)

    Survey method (second digit, uppercase):

      - R: Reconstructed (e.g. historical sources, geomorphic evidence, dating of moraines)
      - M: Derived from maps
      - G: Ground survey (e.g. GPS, tachymetry, tape measure)
      - P: Photogrammetry
      - L: Laser altimetry or scanning
      - Z: Radar altimetry or interferometry
      - C: Combined (explain in `REMARKS`)
      - X: Unknown or other (explain in `REMARKS`)
  - `example`: tG
  - `type`: string
  - `constraints`:
    - `enum`: ['tR', 'tM', 'tG', 'tP', 'tL', 'tZ', 'tC', 'tX', 'aR', 'aM', 'aG', 'aP', 'aL', 'aZ', 'aC', 'aX', 'sR', 'sM', 'sG', 'sP', 'sL', 'sZ', 'sC', 'sX', 'cR', 'cM', 'cG', 'cP', 'cL', 'cZ', 'cC', 'cX', 'xR', 'xM', 'xG', 'xP', 'xL', 'xZ', 'xC', 'xX']

### `INVESTIGATOR`

  - `description`: Names of the persons or agencies that performed the survey or processed the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Michael Zemp
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPONS_AGENCY`

  - `description`: Full name, abbreviation and address of the agencies that sponsored the survey or archived the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: World Glacier Monitoring Service (WGMS), University of Zurich, Wintherthurerstr. 190, 8057 Zurich, Switzerland
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REFERENCE`

  - `description`: References to publications related to the data or methods. Use a short format such as `Author et al. YYYY (URL)` if a canonical URL is available (e.g. https://doi.org/DOI). Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Author et al. YYYY (https://doi.org/DOI)
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `FRONT_VARIATION`

Glacier length changes from in-situ and remote sensing measurements.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `YEAR`

  - `description`: Survey year.
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `maximum`: 2023
    - `required`: True

### `SURVEY_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19940906
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `REFERENCE_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19931002
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `FRONT_VARIATION`

  - `description`: Variation in the position of the glacier front (m) between `REFERENCE_DATE` and `SURVEY_DATE` (positive: advance, negative: retreat).
  - `example`: -17
  - `type`: number

### `FRONT_VAR_UNC`

  - `description`: Estimated random error of reported front variation (m).
  - `example`: 1
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `QUALITATIVE_VARIATION`

  - `description`: Qualitative front variation (in the absence of a quantitative measurement) between `REFERENCE_DATE` and `SURVEY_DATE`.

      - +X: Glacer in advance
      - -X: Glacier in retreat
      - ST: Glacier stationary
  - `example`: -X
  - `type`: string
  - `constraints`:
    - `enum`: ['+X', '-X', 'ST']

### `SURVEY_PLATFORM_METHOD`

  - `description`: Survey platform (first digit, lowercase):

      - t: Terrestrial
      - a: Airborne
      - s: Spaceborne
      - c: Combined (explain in `REMARKS`)
      - x: Unknown or other (explain in `REMARKS`)

    Survey method (second digit, uppercase):

      - R: Reconstructed (e.g. historical sources, geomorphic evidence, dating of moraines)
      - M: Derived from maps
      - G: Ground survey (e.g. GPS, tachymetry, tape measure)
      - P: Photogrammetry
      - L: Laser altimetry or scanning
      - Z: Radar altimetry or interferometry
      - C: Combined (explain in `REMARKS`)
      - X: Unknown or other (explain in `REMARKS`)
  - `example`: aP
  - `type`: string
  - `constraints`:
    - `enum`: ['tR', 'tM', 'tG', 'tP', 'tL', 'tZ', 'tC', 'tX', 'aR', 'aM', 'aG', 'aP', 'aL', 'aZ', 'aC', 'aX', 'sR', 'sM', 'sG', 'sP', 'sL', 'sZ', 'sC', 'sX', 'cR', 'cM', 'cG', 'cP', 'cL', 'cZ', 'cC', 'cX', 'xR', 'xM', 'xG', 'xP', 'xL', 'xZ', 'xC', 'xX']

### `INVESTIGATOR`

  - `description`: Names of the persons or agencies that performed the survey or processed the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Michael Zemp
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPONS_AGENCY`

  - `description`: Full name, abbreviation and address of the agencies that sponsored the survey or archived the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: World Glacier Monitoring Service (WGMS), University of Zurich, Wintherthurerstr. 190, 8057 Zurich, Switzerland
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REFERENCE`

  - `description`: References to publications related to the data or methods. Use a short format such as `Author et al. YYYY (URL)` if a canonical URL is available (e.g. https://doi.org/DOI). Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Author et al. YYYY (https://doi.org/DOI)
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `MASS_BALANCE_OVERVIEW`

Overview of glacier mass balance surveys.

When submitting a mass balance survey, ensure that the corresponding rows in `MASS_BALANCE` and `MASS_BALANCE_POINT` have the same `WGMS_ID` and `YEAR` as the survey.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `YEAR`

  - `description`: Calendar year associated with the last accumulation (winter) - ablation (summer) cycle.
    This is almost always the calendar year at the end of the measurement period
    unless the cycle extends only briefly into the following year
    (e.g. 2020-01-05 to 2021-01-17 ends in 2021 but is the 2020 hydrological year).
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `required`: True
    - `maximum`: 2023

### `TIME_SYSTEM`

  - `description`: Time measurement system for the measurement of annual mass balance:

      - FLO: Floating-date
      - FXD: Fixed-date
      - STR: Stratigraphic
      - COM: Combined - usually STR and FXD per Mayo et al. 1972 (https://doi.org/10.3189/S0022143000022449)
      - OTH: Other - please explain in `REMARKS`

    See Cogley et al. 2011 (https://doi.org/10.5167/uzh-53475) for details on the above time measurement systems. Please give floating dates for `BEGIN_PERIOD`, `END_PERIOD` and `END_WINTER` regardless of system and explain methodological details (e.g. fixed dates and correction methods) in `REMARKS`.

    Note that FLO was only introduced in 2011, so earlier mass balances based on the floating-date system are (at least theoretically) reported as OTH.
  - `example`: FLO
  - `type`: string
  - `constraints`:
    - `enum`: ['FLO', 'FXD', 'STR', 'COM', 'OTH']

### `BEGIN_PERIOD`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19930925
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `END_WINTER`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19940513
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `END_PERIOD`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 19940929
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `ELA_PREFIX`

  - `description`: Whether the equilibrium line altitude (ELA) was below ('<'), on (blank), or above ('>') the glacier.
  - `type`: string
  - `constraints`:
    - `enum`: ['<', '>']

### `ELA`

  - `description`: Mean elevation (m), averaged over the glacier, of the end-of-mass-balance-year equilibrium line. This should be the glacier minimum or maximum elevation if the ELA was below or above the glacier, respectively.
  - `example`: 2673
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9000

### `ELA_UNC`

  - `description`: Estimated random error of `ELA` (m).
  - `example`: 10
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `MIN_SITES_ACC`

  - `description`: Minimum number of sites at which measurements were taken in the accumulation area. Minimum and maximum values can be used to indicate that different numbers of measurements were carried out for winter and annual mass balance surveys or for different measurement types (e.g. snow pits versus snow probings).
  - `example`: 5
  - `type`: integer
  - `constraints`:
    - `minimum`: 0

### `MAX_SITES_ACC`

  - `description`: Maximum number of sites at which measurements were taken in the accumulation area. Minimum and maximum values can be used to indicate that different numbers of measurements were carried out for winter and annual mass balance surveys or for different measurement types (e.g. snow pits versus snow probings).
  - `example`: 41
  - `type`: integer
  - `constraints`:
    - `minimum`: 0

### `MIN_SITES_ABL`

  - `description`: Minimum number of measurement sites in the ablation area used for either the winter or annual mass balance surveys.
  - `example`: 17
  - `type`: integer
  - `constraints`:
    - `minimum`: 0

### `MAX_SITES_ABL`

  - `description`: Maximum number of measurement sites in the ablation area used for either the winter or annual mass balance surveys.
  - `example`: 71
  - `type`: integer
  - `constraints`:
    - `minimum`: 0

### `ACC_AREA`

  - `description`: Accumulation area (km²).
  - `example`: 5.112
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `ACC_AREA_UNC`

  - `description`: Estimated random error of `ACC_AREA` (km²).
  - `example`: 0.01
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `ABL_AREA`

  - `description`: Ablation area (km²).
  - `example`: 1.218
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `ABL_AREA_UNC`

  - `description`: Estimated random error of `ABL_AREA` (km²).
  - `example`: 0.01
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `AAR`

  - `description`: Accumulation area divided by the total glacier area, multiplied by 100 (%).
  - `example`: 81.0
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 100

### `INVESTIGATOR`

  - `description`: Names of the persons or agencies that performed the survey or processed the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Michael Zemp
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPONS_AGENCY`

  - `description`: Full name, abbreviation and address of the agencies that sponsored the survey or archived the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: World Glacier Monitoring Service (WGMS), University of Zurich, Wintherthurerstr. 190, 8057 Zurich, Switzerland
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REFERENCE`

  - `description`: References to publications related to the data or methods. Use a short format such as `Author et al. YYYY (URL)` if a canonical URL is available (e.g. https://doi.org/DOI). Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Author et al. YYYY (https://doi.org/DOI)
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `MASS_BALANCE`

Glacier mass balance measurements by elevation band.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `YEAR`

  - `description`: Calendar year associated with the last accumulation (winter) - ablation (summer) cycle.
    This is almost always the calendar year at the end of the measurement period
    unless the cycle extends only briefly into the following year
    (e.g. 2020-01-05 to 2021-01-17 ends in 2021 but is the 2020 hydrological year).
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `required`: True
    - `maximum`: 2023

### `LOWER_BOUND`

  - `description`: Lower boundary of the surface elevation band (m), or 9999 if referring to the entire glacier.
  - `example`: 2500
  - `type`: integer
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9999
    - `required`: True

### `UPPER_BOUND`

  - `description`: Upper boundary of the surface elevation band (m), or 9999 if referring to the entire glacier.
  - `example`: 2600
  - `type`: integer
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9999
    - `required`: True

### `AREA`

  - `description`: Area of the elevation band (km²).
  - `example`: 0.608
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `WINTER_BALANCE`

  - `description`: Mass balance (mm w.e. ~ kg m⁻²) over the winter (accumulation) season – from `BEGIN_PERIOD` to `END_WINTER`.
  - `example`: 1050
  - `type`: number

### `WINTER_BALANCE_UNC`

  - `description`: Estimated random error of `WINTER_BALANCE` (mm w.e.).
  - `example`: 50
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `SUMMER_BALANCE`

  - `description`: Mass balance (mm w.e. ~ kg m⁻²) over the summer (ablation) season – from `END_WINTER` to `END_PERIOD`.
  - `example`: -1920
  - `type`: number

### `SUMMER_BALANCE_UNC`

  - `description`: Estimated random error of `SUMMER_BALANCE` (mm w.e.).
  - `example`: 20
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `ANNUAL_BALANCE`

  - `description`: Mass balance (mm w.e. ~ kg m⁻²) over the hydrological year – from `BEGIN_PERIOD` to `END_PERIOD`.
  - `example`: -870
  - `type`: number

### `ANNUAL_BALANCE_UNC`

  - `description`: Estimated random error of `ANNUAL_BALANCE` (mm w.e.).
  - `example`: 30
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `MASS_BALANCE_POINT`

Glacier mass balance measured at specific points (e.g. stakes or pits).

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `YEAR`

  - `description`: Calendar year associated with the last accumulation (winter) - ablation (summer) cycle.
    This is almost always the calendar year at the end of the measurement period
    unless the cycle extends only briefly into the following year
    (e.g. 2020-01-05 to 2021-01-17 ends in 2021 but is the 2020 hydrological year).
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `required`: True
    - `maximum`: 2023

### `POINT_ID`

  - `description`: Identifier used for the point in the original study.
  - `example`: P123
  - `type`: string
  - `constraints`:
    - `required`: True

### `FROM_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 20030925
  - `type`: string
  - `constraints`:
    - `required`: True
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `TO_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `REMARKS`.
  - `example`: 20040515
  - `type`: string
  - `constraints`:
    - `required`: True
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `POINT_LAT`

  - `description`: Latitude in decimal degrees (°, WGS 84). Positive values indicate the northern hemisphere and negative values indicate the southern hemisphere.
  - `example`: 46.8709
  - `type`: number
  - `constraints`:
    - `minimum`: -90
    - `maximum`: 90

### `POINT_LON`

  - `description`: Longitude in decimal degrees (°, WGS 84). Positive values indicate east of the zero meridian and negative values indicate west of the zero meridian.
  - `example`: 10.8261
  - `type`: number
  - `constraints`:
    - `minimum`: -180
    - `maximum`: 180

### `POINT_ELEVATION`

  - `description`: Glacier surface elevation (m).
  - `example`: 2550
  - `type`: number

### `POINT_BALANCE`

  - `description`: Mass balance (mm w.e.) between `FROM_DATE` and `TO_DATE`.
  - `example`: 3500
  - `type`: number

### `POINT_BALANCE_UNCERTAINTY`

  - `description`: Estimated random error of `POINT_BALANCE` (mm w.e.).
  - `example`: 100
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `DENSITY`

  - `description`: Mean (measured or estimated) glacier density (kg m⁻³) used to convert thickness change (mm) to mass balance (mm w.e.). If multiple density values were used (e.g. for snow and ice), they should be described in `REMARKS`.
  - `example`: 400
  - `type`: number
  - `constraints`:
    - `minimum`: 1
    - `maximum`: 1000

### `DENSITY_UNCERTAINTY`

  - `description`: Estimated random error of `DENSITY` (kg m⁻³).
  - `example`: 100
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 1000

### `BALANCE_CODE`

  - `description`: Whether and how the point balance was used in the calculation of glacier-wide balances:

      - BW: Winter balance (`MASS_BALANCE.WINTER_BALANCE`)
      - BS: Summer balance (`MASS_BALANCE.SUMMER_BALANCE`)
      - BA: Annual balance (`MASS_BALANCE.ANNUAL_BALANCE`)
      - IN: Index point not used for glacier-wide balance calculations
  - `example`: BW
  - `type`: string
  - `constraints`:
    - `enum`: ['BW', 'BS', 'BA', 'IN']
    - `required`: True

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `SPECIAL_EVENT`

Extraordinary events concerning glacier hazards and dramatic glacier changes.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `EVENT_ID`

  - `description`: Unique identifier (assigned by the WGMS).
  - `example`: 123
  - `type`: integer
  - `constraints`:
    - `unique`: True
    - `minimum`: 1
    - `required`: True

### `EVENT_DATE`

  - `description`: Date formatted as YYYYMMDD (4-digit year, 2-digit month, and 2-digit day). Use '99' to designate unknown day or month (e.g. 20100199, 20109999) and make a note in `EVENT_DESCRIPTION`.

    For events spanning multiple days, the date of the main event should be given and the sequence of events further described in `EVENT_DESCRIPTION`.
  - `example`: 20000908
  - `type`: string
  - `constraints`:
    - `pattern`: `(1[0-9]{3}|20[0-1][0-9]|202[0-3])(0[1-9]|1[0-2]|99)(0[1-9]|[1-2][0-9]|3[0-1]|99)`

### `ET_SURGE`

  - `description`: Whether a surge was involved.
  - `type`: boolean

### `ET_CALVING`

  - `description`: Whether calving was involved.
  - `type`: boolean

### `ET_FLOOD`

  - `description`: Whether a flood (e.g. glacial-lake outburst flood, debris flow) was involved.
  - `type`: boolean

### `ET_AVALANCHE`

  - `description`: Whether an ice avalanche was involved.
  - `type`: boolean

### `ET_TECTONIC`

  - `description`: Whether tectonics (e.g. earthquake, volcanic eruption) were involved.
  - `type`: boolean

### `ET_OTHER`

  - `description`: Whether any other event types were involved.
  - `type`: boolean

### `EVENT_DESCRIPTION`

  - `description`: Summary description of the event sequence - including for example the type and scale of the damage, measures taken to mitigate glacier hazards, and studies carried out in connection with the event. Quantitative information should be included whenever possible.

      - Surge: Date and location of onset, duration, flow velocity, discharge anomalies and periodicity
      - Calving: Rate of retreat, iceberg discharge, flow velocity and water depth at calving front
      - Flood: Volume, mechanism, peak discharge, sediment load, reach and propagation velocity of flood wave or flow front
      - Ice avalanche: Volume, runout distance, overall slope (ratio of vertical drop height to horizontal runout distance) of path
      - Tectonics: Volumes, runout distances and overall slopes (ratio of vertical drop height to horizontal runout distance) of rockfall on glacier surface, amount of geothermal melting in craters, etc.
  - `example`: On 8 September 2000, a rock fall of about 0.1 million m3 started from 2000–2200 m on the west face of Mättenberg, above Findelen Glacier. The rockfall reached and destroyed the trail leading to the Schreckhornhütte.
  - `type`: string

### `INVESTIGATOR`

  - `description`: Names of the persons or agencies that performed the survey or processed the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Michael Zemp
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPONS_AGENCY`

  - `description`: Full name, abbreviation and address of the agencies that sponsored the survey or archived the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: World Glacier Monitoring Service (WGMS), University of Zurich, Wintherthurerstr. 190, 8057 Zurich, Switzerland
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REFERENCE`

  - `description`: References to publications related to the data or methods. Use a short format such as `Author et al. YYYY (URL)` if a canonical URL is available (e.g. https://doi.org/DOI). Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Author et al. YYYY (https://doi.org/DOI)
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `RECONSTRUCTION_SERIES`

Overview of reconstructed glacier length change series.

When submitting a new series, assign a temporary `REC_SERIES_ID` and use this as the `REC_SERIES_ID` for all corresponding entries in `RECONSTRUCTION_FRONT_VARIATION`.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `REC_SERIES_ID`

  - `description`: Reconstruction series identifier (assigned by the WGMS).
  - `example`: 42
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 1

### `INVESTIGATOR`

  - `description`: Names of the persons or agencies that performed the survey or processed the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Michael Zemp
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `SPONS_AGENCY`

  - `description`: Full name, abbreviation and address of the agencies that sponsored the survey or archived the data. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: World Glacier Monitoring Service (WGMS), University of Zurich, Wintherthurerstr. 190, 8057 Zurich, Switzerland
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REFERENCE`

  - `description`: References to publications related to the data or methods. Use a short format such as `Author et al. YYYY (URL)` if a canonical URL is available (e.g. https://doi.org/DOI). Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Author et al. YYYY (https://doi.org/DOI)
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

## `RECONSTRUCTION_FRONT_VARIATION`

Glacier length changes reconstructed from historic records and geologic dating.

### `POLITICAL_UNIT`

  - `description`: Two-character code (ISO 3166 Alpha-2) of the country in which the glacier is located. Must match `GLACIER.POLITICAL_UNIT` for the corresponding `WGMS_ID`.
  - `example`: CH
  - `type`: string
  - `constraints`:
    - `required`: True

### `NAME`

  - `description`: The name of the glacier. Must match `GLACIER.NAME` for the corresponding `WGMS_ID`.
  - `example`: FINDELEN
  - `type`: string
  - `constraints`:
    - `required`: True

### `WGMS_ID`

  - `description`: Integer key identifying glaciers in the Fluctuations of Glaciers (FoG) database. For new glacier entries, this key is assigned by the WGMS.
  - `example`: 389
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 0
    - `maximum`: 999999

### `REC_SERIES_ID`

  - `description`: Reconstruction series identifier (assigned by the WGMS).
  - `example`: 42
  - `type`: integer
  - `constraints`:
    - `required`: True
    - `minimum`: 1

### `YEAR`

  - `description`: Survey year.
  - `example`: 2004
  - `type`: year
  - `constraints`:
    - `required`: True
    - `maximum`: 2023

### `YEAR_UNC`

  - `description`: Estimated random error of `YEAR` (years).
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `REFERENCE_YEAR`

  - `description`: Reference year.
  - `example`: 1904
  - `type`: year
  - `constraints`:
    - `maximum`: 2023

### `REF_YEAR_UNC`

  - `description`: Estimated maximum error of `REFERENCE_YEAR` (years).
  - `example`: 3
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `FRONT_VARIATION`

  - `description`: Variation in the position of the glacier front (m) from `REFERENCE_YEAR` to `YEAR` (positive: advance, negative: retreat).
  - `example`: -230
  - `type`: number

### `QUALITATIVE_VARIATION`

  - `description`: Qualitative front variation (in the absence of a quantitative measurement) between `REFERENCE_DATE` and `SURVEY_DATE`.

      - +X: Glacer in advance
      - -X: Glacier in retreat
      - ST: Glacier stationary
  - `example`: -X
  - `type`: string
  - `constraints`:
    - `enum`: ['+X', '-X', 'ST']

### `FRONT_VAR_POS_UNC`

  - `description`: Estimated maximum positive error for `FRONT_VARIATION` (m). `FRONT_VARIATION` plus `FRONT_VAR_POS_UNC` should mark the maximum possible front variation.
  - `example`: 10
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `FRONT_VAR_NEG_UNC`

  - `description`: Estimated maximum negative error for `FRONT_VARIATION` (m). `FRONT_VARIATION` plus `FRONT_VAR_NEG_UNC` should mark the minimum possible front variation.
  - `example`: 10
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `LOWEST_ELEVATION`

  - `description`: Lowest elevation on the glacier (m).
  - `example`: 2370
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9000

### `HIGHEST_ELEVATION`

  - `description`: Highest elevation on the glacier (m).
  - `example`: 3370
  - `type`: number
  - `constraints`:
    - `minimum`: 0
    - `maximum`: 9000

### `ELEVATION_UNC`

  - `description`: Estimated random error of reported elevations (m).
  - `type`: number
  - `constraints`:
    - `minimum`: 0

### `MORAINE_DEFINED_MAX`

  - `description`: Condition of the moraine used to determine maximum glacier length.

      - MMP: Moraine mainly preserved
      - MPE: Moraine partly eroded
      - MME: Moraine mainly eroded

    If another object was used (e.g. a large boulder or a building), the condition codes can be used but the object should be described in `REMARKS`.
  - `example`: MPE
  - `type`: string
  - `constraints`:
    - `enum`: ['MMP', 'MPE', 'MME']

### `METHOD_CODE`

  - `description`: Method(s) used to reconstruct glacier length in `YEAR`.

      - PAI: Oil painting
      - DRA: Drawing
      - PRT: Print
      - PHO: Photograph
      - MAP: Map
      - WRS: Written source
      - HIS: Other historical source (specify in `METHOD_REMARKS`)
      - RAD: Radiocarbon date
      - DEN: Dendrochronology
      - COM: Combination of multiple methods (specify in `METHOD_REMARKS`)
      - OTH: Other (specify in `METHOD_REMARKS`)
  - `example`: COM
  - `type`: string
  - `constraints`:
    - `enum`: ['PAI', 'DRA', 'PRT', 'PHO', 'MAP', 'WRS', 'HIS', 'RAD', 'DEN', 'COM', 'OTH']

### `METHOD_REMARKS`

  - `description`: Description of the method(s) used, e.g. relative date (REL) using weathering rind thickness, lichenometry or Schmidt hammer rebound.
  - `example`: PHO & WRS
  - `type`: string

### `REMARKS`

  - `description`: Any important information or comments not included elsewhere. Cannot contain leading ( \*), trailing (\* ), or consecutive (\* &nbsp; \*) spaces.
  - `example`: Example data. Should not be used for science.
  - `type`: string
  - `constraints`:
    - `pattern`: `[^\s]+( [^\s]+)*`

