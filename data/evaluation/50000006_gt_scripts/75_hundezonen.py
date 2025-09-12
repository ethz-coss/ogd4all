# Q: Was ist die Objekt ID der nächste Hundefreilaufzone zu {{addr}}?
# Relevant datasets: ['Hundezonen']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Hundezonen: ['gsz.hundezone']

### Solution ###
# Suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

gdf_hunde = gpd.read_file("./data/opendata/50000006/extracted/hundezonen.gpkg", layer="gsz.hundezone")
gdf_hunde = gdf_hunde[gdf_hunde["einschraenkungen"] == "Hundefreilaufzone"]

location = geolocator.geocode(f"{args.addr.strip()}, Zürich")
addr_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
addr_location = addr_location.to_crs(epsg=2056)

gdf_hunde["distance"] = gdf_hunde.geometry.distance(addr_location.iloc[0])
closest_zone = gdf_hunde.loc[gdf_hunde["distance"].idxmin()]
print(closest_zone["objectid"])