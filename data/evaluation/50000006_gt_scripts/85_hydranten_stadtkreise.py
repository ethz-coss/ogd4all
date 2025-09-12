# Q: Was ist das durchschnittliche Verlegejahr (auf ganze Zahl gerundet) von Hydranten im Stadtkreis {{stadtkreis_num}}?
# Relevant datasets: ['Hydranten', 'Stadtkreise']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stadtkreis_num", type=str)
args = parser.parse_args()
# Layers in dataset Hydranten: ['wvz.wvz_hydranten']
# Layers in dataset Stadtkreise: ['stzh.adm_stadtkreise_v', 'stzh.adm_stadtkreise_a', 'stzh.adm_stadtkreise_beschr_p']

### Solution ###
gdf_hydra = gpd.read_file("./data/opendata/50000006/extracted/hydranten.gpkg", layer="wvz.wvz_hydranten")
gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtkreise.gpkg", layer="stzh.adm_stadtkreise_a")

relevant_kreis = gdf_stadt[gdf_stadt["bezeichnung"] == f"Kreis {args.stadtkreis_num.strip()}"]

gdf_hydra = gdf_hydra[gdf_hydra.geometry.within(relevant_kreis.geometry.values[0])]
avg_year = gdf_hydra["verlegejahr"].mean()
print(round(avg_year))