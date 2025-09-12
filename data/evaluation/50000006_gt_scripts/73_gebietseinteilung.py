# Q: Zu welcher römisch-katholischen Kirchgemeinde gehört die Adresse {{addr}}
# Relevant datasets: ['Gebietseinteilung der römisch-katholischen Kirchgemeinden']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Gebietseinteilung der römisch-katholischen Kirchgemeinden: ['kirchezh.kirchgemeinde_roem_kath_a']

### Solution ###
gdf_gebie = gpd.read_file("./data/opendata/50000006/extracted/gebietseinteilung_der_roemisch_katholischen_kirchgemeinden.gpkg", layer="kirchezh.kirchgemeinde_roem_kath_a")

location = geolocator.geocode(f"{args.addr.strip()}, Zürich")
addr_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
addr_location = addr_location.to_crs(epsg=2056)

filtered_gdf = gdf_gebie[gdf_gebie.geometry.apply(lambda x: x.contains(addr_location.iloc[0]) if x is not None else False)]
print(filtered_gdf["bezeichnung"].iloc[0] if not filtered_gdf.empty else "No matching zone found")