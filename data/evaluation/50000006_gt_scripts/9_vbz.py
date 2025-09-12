# Q: Welche VBZ-Verkaufsstelle ist am nächsten beim Esri R&D Center in Zürich?
# Relevant datasets: ['VBZ-Verkaufsstelle']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os

geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")

# Layers in dataset VBZ-Verkaufsstelle: ['stzh.poi_vbzverkaufsstelle_view']

### Solution ###
gdf_vbz_v = gpd.read_file("./data/opendata/50000006/extracted/vbz_verkaufsstelle.gpkg", layer="stzh.poi_vbzverkaufsstelle_view")

location = geolocator.geocode("Esri R&D Center Zürich")
esri_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
esri_location = esri_location.to_crs(epsg=2056)
gdf_vbz_v['distance'] = gdf_vbz_v['geometry'].apply(lambda x: x.distance(esri_location.iloc[0]))
closest_vbz = gdf_vbz_v.nsmallest(1, 'distance')
print(closest_vbz['name'].iloc[0])