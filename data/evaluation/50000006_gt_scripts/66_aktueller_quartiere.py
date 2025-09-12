# Q: Wieviele Hunde hatte es im statistischen Quartier {{quartier_lang}} pro Quadratkilometer im Jahr 2023? Runde auf eine Nachkommastelle.
# Relevant datasets: ['Hundebestände der Stadt Zürich, seit 2014', 'Statistische Quartiere']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("quartier_lang", type=str)
args = parser.parse_args()
# Layers in dataset Hundebestände der Stadt Zürich, seit 2014: ['kul100od1001']
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']

### Solution ###
gdf_kul10 = gpd.read_file("./data/opendata/50000006/extracted/kul100od1001.csv", layer="kul100od1001", encoding="utf-8")
gdf_quart = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")

gdf_kul10_filtered = gdf_kul10[gdf_kul10["StichtagDatJahr"] == "2023"]
gdf_kul10_filtered = gdf_kul10_filtered[gdf_kul10_filtered["QuarLang"] == args.quartier_lang.strip()]
gdf_kul10_filtered["AnzHunde"] = gdf_kul10_filtered["AnzHunde"].astype(int)
num_dogs = gdf_kul10_filtered.groupby("QuarLang").agg({"AnzHunde": "sum"}).reset_index()["AnzHunde"].values[0]
quart_area = gdf_quart[gdf_quart["qname"] == args.quartier_lang.strip()].area.values[0]
print(f"{round(num_dogs / (quart_area / 1e6), 1)} Hunde/km²")