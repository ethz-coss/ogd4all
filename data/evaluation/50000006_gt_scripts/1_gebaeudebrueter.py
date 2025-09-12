# Q: Nester welcher Vogelarten wurden bei Gebäuden in einem Umkreis von 250 Metern von der Schiessanlage Albisgüetli gesichtet?
# Relevant datasets: ['Gebäudebrüter']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
from shapely.geometry import Point
import os

geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Gebäudebrüter: ['gsz.gebaeudebrueter_inventar']

### Solution ###
gdf_gebae = gpd.read_file("./data/opendata/50000006/extracted/gebaeudebrueter.gpkg", layer="gsz.gebaeudebrueter_inventar")

# Locate Turbinenplatz
location = geolocator.geocode("Schiessplatz Albisgüetli, Zürich")

crs_buildings = gdf_gebae.crs
pt = gpd.GeoSeries(
        [Point(location.longitude, location.latitude)],
        crs="EPSG:4326" # geopy returns WGS84 (EPSG:4326), so we need to convert it to the same CRS as the buildings
     ).to_crs(crs_buildings)

buffer_250m = pt.buffer(250)

# Find buildings whose footprint intersects the buffer
hits = gdf_gebae[gdf_gebae.intersects(buffer_250m.iloc[0])]
if hits.empty:
    print("Kein Gebäudebrüter-Nachweis im 250 m Umkreis.")
else:
    vogelarten = hits["vogelarten"].dropna().unique()
    print(", ".join(vogelarten.tolist()))