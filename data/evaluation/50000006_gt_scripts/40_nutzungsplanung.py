# Q: In was für einem Bauzonen Typ liegt {{addr}}?
# Relevant datasets: ['Nutzungsplanung - kommunale Bau- und Zonenordnung (BZO)']

import geopandas as gpd
from geopy.geocoders import Nominatim, GoogleV3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr", type=str)
args = parser.parse_args()
geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")
# Layers in dataset Nutzungsplanung - kommunale Bau- und Zonenordnung (BZO): ['afs.bzo_zone_laermvorbelast_v', 'afs.bzo_proj_eg_nutzung_flaeche_v', 'afs.bzo_baumschutz_v', 'afs.bzo_proj_zone_wohnanteil_v', 'afs.bzo_zone_es_v', 'afs.bzo_hochhausgebiet_v', 'afs.bzo_planungszone_v', 'afs.bzo_proj_zone_innenl_f_v', 'afs.bzo_quez_linie_v', 'afs.bzo_gp_v', 'afs.bzo_zone_innenl_f_v', 'afs.bzo_kernzone_v', 'afs.bzo_gewaesserabstandsl_v', 'afs.bzo_zone_ffz_v', 'afs.bzo_proj_zone_v', 'afs.bzo_zone_v', 'afs.bzo_aussichtsschutz_v', 'afs.bzo_proj_gp_pflicht_v', 'afs.bzo_proj_zone_es_v', 'afs.bzo_proj_waldabstandsl_v', 'afs.bzo_eg_nutzung_linie_v', 'afs.bzo_proj_sbv_v', 'afs.bzo_zone_erhoehte_az_v', 'afs.bzo_zone_wohnanteil_v', 'afs.bzo_sbv_v', 'afs.bzo_quez_v', 'afs.bzo_proj_kernzone_v', 'afs.bzo_waldabstandsl_v', 'afs.bzo_proj_revisionsgebiet_v', 'afs.bzo_gp_pflicht_v', 'afs.bzo_proj_gp_v', 'afs.bzo_eg_nutzung_flaeche_v', 'afs.bzo_proj_eg_nutzung_linie_v']

### Solution ###
gdf_nutzu = gpd.read_file("./data/opendata/50000006/extracted/nutzungsplanung_kommunale_bau_und_zonenordnung_bzo_.gpkg", layer="afs.bzo_zone_v")

location = geolocator.geocode(f"{args.addr.strip()}, Zürich")
building_location = gpd.GeoSeries(gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
building_location = building_location.to_crs(epsg=2056)
zone = gdf_nutzu[gdf_nutzu.geometry.contains(building_location.union_all())]
if not zone.empty:
    print(zone['typ'].values[0])