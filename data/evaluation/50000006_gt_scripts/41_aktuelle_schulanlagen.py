# Q: Was ist der Projektbeschrieb des Tiefbauprojekts {{location_desc}}?
# Relevant datasets: ['Aktuelle Tiefbauprojekte im öffentlichen Grund']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("location_desc", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Aktuelle Tiefbauprojekte im öffentlichen Grund: ['stzh.aer_baustellen_a']

### Solution ###
# Challenge: understand what it means to be "next to" something
gdf_aktue = gpd.read_file("./data/opendata/50000006/extracted/aktuelle_tiefbauprojekte_im_oeffentlichen_grund.gpkg", layer="stzh.aer_baustellen_a")

# location_desc options: ["neben der Schule im Herrlig", "neben dem Park Zürichhorn", "beim Beckenhof"
if args.location_desc.strip() == "neben der Schule im Herrlig":
    location = geolocator.geocode("Schule im Herrlig, Zürich")
elif args.location_desc.strip() == "beim Schiffsteg Wollishofen":
    location = geolocator.geocode("Schiffsteg Wollishofen, Zürich")
elif args.location_desc.strip() == "beim Beckenhof":
    location = geolocator.geocode("Beckenhof, Zürich")
else:
    raise ValueError(f"Unknown location description: {args.location_desc}")
location_crs = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
location_crs = location_crs.to_crs(epsg=2056)
gdf_aktue['distance'] = gdf_aktue['geometry'].apply(lambda x: x.distance(location_crs.iloc[0]))
closest_projekt = gdf_aktue.nsmallest(1, 'distance')
print(closest_projekt['projektbeschrieb'].iloc[0])