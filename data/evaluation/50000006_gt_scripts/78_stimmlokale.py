# Q: Was ist die Adresse des nächsten Stimmlokals zu {{addr}}
# Relevant datasets: ['Stimmlokale']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Stimmlokale: ['stzh.poi_stimmlokal_att', 'stzh.poi_stimmlokal_view']

### Solution ###
gdf_stimm = gpd.read_file("./data/opendata/50000006/extracted/stimmlokale.gpkg", layer="stzh.poi_stimmlokal_view")

location = geolocator.geocode(f"{args.addr.strip()}, Zürich")
addr_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
addr_location = addr_location.to_crs(epsg=2056)

gdf_stimm["distance"] = gdf_stimm.geometry.distance(addr_location.iloc[0])
closest_stimmkreis = gdf_stimm.loc[gdf_stimm["distance"].idxmin()]
print(closest_stimmkreis["adresse"])