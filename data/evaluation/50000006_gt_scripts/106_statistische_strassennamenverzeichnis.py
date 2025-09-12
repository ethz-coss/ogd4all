# Q: Verläuft die Strasse {{street_name}} vollständig innerhalb des statistischen Quartiers Fluntern?
# Relevant datasets: ['Statistische Quartiere', 'Strassennamenverzeichnis']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("street_name", type=str)
args = parser.parse_args()
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']
# Layers in dataset Strassennamenverzeichnis: ['geoz.sv_str_verz', 'geoz.sv_snb_beschluesse', 'geoz.sv_str_lin']

### Solution ###
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")
gdf_stras = gpd.read_file("./data/opendata/50000006/extracted/strassennamenverzeichnis.gpkg", layer="geoz.sv_str_lin")

quar = gdf_stati[gdf_stati["qname"] == "Fluntern"]
stras_geometry = gdf_stras[gdf_stras["str_name"] == args.street_name.strip()].geometry.union_all()

if quar.geometry.contains(stras_geometry).all():
    print("Ja")
else:
    print("Nein")