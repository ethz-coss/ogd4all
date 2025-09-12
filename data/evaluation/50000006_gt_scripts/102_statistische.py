# Q: Grenzen die statistischen Quartiere {{quartier_tuple}} aneinander?
# Relevant datasets: ['Statistische Quartiere']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("quartier_tuple", type=str)
args = parser.parse_args()
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']

### Solution ###
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")

quartier1, quartier2 = args.quartier_tuple.split(" und ")
gdf1 = gdf_stati[gdf_stati["qname"] == quartier1]
gdf2 = gdf_stati[gdf_stati["qname"] == quartier2]

if gdf1.geometry.iloc[0].touches(gdf2.geometry.iloc[0]):
    print("Ja")
else:
    print("Nein")
