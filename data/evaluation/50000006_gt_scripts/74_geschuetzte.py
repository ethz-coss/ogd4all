# Q: Wieviele geschützte Bäume hat es in einem 100 Meter Radius um {{addr}}?
# Relevant datasets: ['Geschützte Einzelbäume']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
from shapely.geometry import Point
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Geschützte Einzelbäume: ['gsz.geschuetzteeinzelbaeume']

### Solution ###
gdf_gesch = gpd.read_file("./data/opendata/50000006/extracted/geschuetzte_einzelbaeume.gpkg", layer="gsz.geschuetzteeinzelbaeume")

location = geolocator.geocode(f"{args.addr.strip()}, Zürich")
crs_gesch = gdf_gesch.crs
pt = gpd.GeoSeries(
        [Point(location.longitude, location.latitude)],
        crs="EPSG:4326" # geopy returns WGS84 (EPSG:4326), so we need to convert it
     ).to_crs(crs_gesch)

buffer_100m = pt.buffer(100)

# Find protected trees whose footprint intersects the buffer
hits = gdf_gesch[gdf_gesch.intersects(buffer_100m.iloc[0])]
if hits.empty:
    print("Kein geschützter Baum im 100 m Umkreis.")
else:
    print(len(hits))