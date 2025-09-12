# Q: Was ist die {{id_type}} des nächstgelegenen Notwasserbrunnens zur Adresse Förrlibuckstrasse 110 in Zürich?
# Relevant datasets: ['Brunnen']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("id_type", type=str)
args = parser.parse_args()

geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Brunnen: ['wvz.wvz_brunnen']

### Solution ###
# Suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

gdf_brunn = gpd.read_file("./data/opendata/50000006/extracted/brunnen.gpkg", layer="wvz.wvz_brunnen")
gdf_brunn = gdf_brunn[gdf_brunn['art'] =="Notwasserbrunnen"]

location = geolocator.geocode("Förrlibuckstrasse 110, Zürich")
esri_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
esri_location = esri_location.to_crs(epsg=2056)
gdf_brunn['distance_to_esri'] = gdf_brunn['geometry'].apply(lambda x: x.distance(esri_location.iloc[0]))
closest_fountain = gdf_brunn.nsmallest(1, 'distance_to_esri')
if args.id_type.strip() == "Objekt ID":
    print(int(closest_fountain['objectid'].iloc[0]))
elif args.id_type.strip() == "Brunnennummer":
    print(int(closest_fountain['brunnennummer'].iloc[0]))
else:
    raise ValueError(f"Unknown id_type: {args.id_type.strip()}")