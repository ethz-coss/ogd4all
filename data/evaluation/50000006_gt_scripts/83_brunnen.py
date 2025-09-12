# Q: Wieviele Brunnen hat es in einem Umkreis von {{num_meters}} um die Adresse Poststrasse 1?
# Relevant datasets: ['Brunnen']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
from shapely.geometry import Point
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_meters", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Brunnen: ['wvz.wvz_brunnen']

### Solution ###
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
gdf_brunn = gpd.read_file("./data/opendata/50000006/extracted/brunnen.gpkg", layer="wvz.wvz_brunnen")

location = geolocator.geocode("Poststrasse 1, Zürich")
crs_gesch = gdf_brunn.crs
pt = gpd.GeoSeries(
        [Point(location.longitude, location.latitude)],
        crs="EPSG:4326" # geopy returns WGS84 (EPSG:4326), so we need to convert it
     ).to_crs(crs_gesch)

buffer = pt.buffer(int(args.num_meters))

hits = gdf_brunn[gdf_brunn.intersects(buffer.iloc[0])]
if hits.empty:
    print(f"Keine Brunnen im Umkreis von {args.num_meters} m.")
else:
    print(len(hits))