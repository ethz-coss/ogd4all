# Q: Ist die thermische Regeneration von Erdwärmesonden zulässig bei der Adresse {{addr}}?
# Relevant datasets: ['Einsatz Luft-Wasser-Wärmepumpen']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Einsatz Luft-Wasser-Wärmepumpen: ['eb.einsatz_luft_wasser_waermepump']

### Solution ###
gdf_einsa = gpd.read_file("./data/opendata/50000006/extracted/einsatz_luft_wasser_waermepumpen.gpkg", layer="eb.einsatz_luft_wasser_waermepump")

location = geolocator.geocode(f"{args.addr.strip()}, Zürich")
addr_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
addr_location = addr_location.to_crs(epsg=2056)

filtered_gdf = gdf_einsa[gdf_einsa.geometry.apply(lambda x: x.contains(addr_location.iloc[0]) if x is not None else False)]
print(filtered_gdf["thermische_regeneration"].iloc[0] if not filtered_gdf.empty else "No matching zone found")