# Q: Wie viele Brunnen gibt es im Stadtkreis {{kreis_nr}}?
# Relevant datasets: ['Brunnen']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("kreis_nr", type=str)
args = parser.parse_args()
# Layers in dataset Brunnen: ['wvz.wvz_brunnen']

### Solution ###
# Suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

gdf_brunn = gpd.read_file("./data/opendata/50000006/extracted/brunnen.gpkg", layer="wvz.wvz_brunnen")

gdf_brunn_stadtkreis_5 = gdf_brunn[gdf_brunn['stadtkreis'] == args.kreis_nr.strip()]
print(gdf_brunn_stadtkreis_5.shape[0])


# Alternative solution
# gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtkreise.gpkg", layer="stzh.adm_stadtkreise_a")
# relevant_kreis = gdf_stadt[gdf_stadt["bezeichnung"] == f"Kreis {args.kreis_nr.strip()}"]
# gdf_brunn_filtered = gdf_brunn[gdf_brunn.geometry.within(relevant_kreis.geometry.values[0])]
# print(gdf_brunn_filtered.shape[0])