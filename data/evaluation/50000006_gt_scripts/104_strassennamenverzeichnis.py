# Q: Kreuzen sich die Strassen {{street_name_tuple}}? Falls ja, gib auch die Koordinaten des Kreuzungspunkts an (EPSG:2056).
# Relevant datasets: ['Strassennamenverzeichnis']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("street_name_tuple", type=str)
args = parser.parse_args()
# Layers in dataset Strassennamenverzeichnis: ['geoz.sv_str_verz', 'geoz.sv_snb_beschluesse', 'geoz.sv_str_lin']

### Solution ###
gdf_stras = gpd.read_file("./data/opendata/50000006/extracted/strassennamenverzeichnis.gpkg", layer="geoz.sv_str_lin")

street1, street2 = args.street_name_tuple.split(" und ")
street1_geometry = gdf_stras[gdf_stras["str_name"] == street1].geometry.union_all()
street2_geometry = gdf_stras[gdf_stras["str_name"] == street2].geometry.union_all()

pt = street1_geometry.intersection(street2_geometry)
if pt.is_empty:
    print("Nein.")
else:
    print(f"Ja, ({pt.x}, {pt.y})")