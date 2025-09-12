# Q: Was ist die summierte Fläche der Stadtkreise {{kreis_list}} in km², gerundet auf eine Nachkommastelle?
# Relevant datasets: ['Stadtkreise']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("kreis_list", type=str)
args = parser.parse_args()
# Layers in dataset Stadtkreise: ['stzh.adm_stadtkreise_v', 'stzh.adm_stadtkreise_a', 'stzh.adm_stadtkreise_beschr_p']

### Solution ###
gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtkreise.gpkg", layer="stzh.adm_stadtkreise_v")

kreis_list = args.kreis_list.split(",")
kreis_list = [int(k) for k in kreis_list]

gdf_stadt = gdf_stadt[gdf_stadt["knr"].isin(kreis_list)]
print(f"{round(gdf_stadt.area.sum() / 1e6, 1)} km²")