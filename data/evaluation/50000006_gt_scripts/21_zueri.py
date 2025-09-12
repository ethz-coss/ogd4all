# Q: Wo ist das nächste rollstuhlgängige, öffentliche WC beim Seebad Enge?
# Relevant datasets: ['Züri WC']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os

geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")

# Layers in dataset Züri WC: ['stzh.poi_zueriwc_rs_att', 'stzh.poi_zueriwc_att', 'stzh.poi_zueriwc_view', 'stzh.poi_zueriwc_mobil_view', 'stzh.poi_zueriwc_rs_view', 'stzh.poi_zueriwc_mobil_rs_view']

### Solution ###
# Challenge: Data layers are not documented well, RS means rollstuhlgängig, mobil refers to a mobile WC, and view layer needs to be used as att has no geometry
gdf_zueri = gpd.read_file("./data/opendata/50000006/extracted/zueri_wc.gpkg", layer="stzh.poi_zueriwc_rs_view")

location = geolocator.geocode("Seebad Enge, Zürich")
seebad_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
seebad_location = seebad_location.to_crs(epsg=2056)
gdf_zueri['distance'] = gdf_zueri['geometry'].apply(lambda x: x.distance(seebad_location.iloc[0]))
closest_spital = gdf_zueri.nsmallest(1, 'distance')
print(closest_spital['standort'].iloc[0])