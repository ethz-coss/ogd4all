# Q: Wieviele Stützzmauern gibt es, die komplett innerhalb der statistischen Zone '{{stat_zone}}' liegen?
# Relevant datasets: ['Statistische Zonen', 'Kunstbauteninventar']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stat_zone", type=str)
args = parser.parse_args()
# Layers in dataset Statistische Zonen: ['stzh.adm_statzonen_map', 'stzh.adm_statzonen_v', 'stzh.adm_statzonen_beschr_p']
# Layers in dataset Kunstbauteninventar: ['taz.view_kuba_linien', 'taz.view_kuba_flaechen']

### Solution ###
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_zonen.gpkg", layer="stzh.adm_statzonen_v")
gdf_kunst = gpd.read_file("./data/opendata/50000006/extracted/kunstbauteninventar.gpkg", layer="taz.view_kuba_linien")

gdf_stati_zone = gdf_stati[gdf_stati['stzname'] == args.stat_zone.strip()]
gdf_kunst_stuetz = gdf_kunst[gdf_kunst['kategorie'] == 'Stützmauer']
gdf_stuetz_eth = gpd.sjoin(gdf_kunst_stuetz, gdf_stati_zone, how="inner", predicate="within")
print(gdf_stuetz_eth.shape[0])