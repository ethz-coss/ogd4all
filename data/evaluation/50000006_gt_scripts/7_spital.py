# Q: Welches Spital ist dem Hauptgebäude der ETH Zürich am nächsten?
# Relevant datasets: ['Spital']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os

# Layers in dataset Spital: ['stzh.poi_spital_view']
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")

### Solution ###
# Challenge: geopy geocoder uses WGS84 (EPSG:4326), but the data is in CH1903+ (EPSG:2056)
gdf_spita = gpd.read_file("./data/opendata/50000006/extracted/spital.gpkg", layer="stzh.poi_spital_view")

location = geolocator.geocode("Hauptgebäude ETH Zürich")
eth_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
eth_location = eth_location.to_crs(epsg=2056)
gdf_spita['distance'] = gdf_spita['geometry'].apply(lambda x: x.distance(eth_location.iloc[0]))
closest_spital = gdf_spita.nsmallest(1, 'distance')
print(closest_spital['name'].iloc[0])