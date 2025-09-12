# Q: Wie viele Prozent der öffentlichen Spielplätze liegen innerhalb von {{num_meters}} Metern um Schulanlagen (Stand 2024)? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['POI öffentliche Spielplätze', 'Schulanlagen']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_meters", type=str)
args = parser.parse_args()
# Layers in dataset POI öffentliche Spielplätze: ['stzh.poi_spielplatz_att', 'stzh.poi_spielplatz_view']
# Layers in dataset Schulanlagen: ['stzh.poi_kinderhort_att', 'stzh.poi_volksschule_att', 'stzh.poi_kindergarten_att', 'stzh.poi_kindergarten_view', 'stzh.poi_kinderhort_view', 'stzh.poi_volksschule_view', 'ssd.schulanlagen_archiv', 'stzh.poi_sonderschule_view']

### Solution ###
# Challenge: understand it should use archive and filter for 2024
gdf_poi_o = gpd.read_file("./data/opendata/50000006/extracted/poi_oeffentliche_spielplaetze.gpkg", layer="stzh.poi_spielplatz_view")
gdf_schul = gpd.read_file("./data/opendata/50000006/extracted/schulanlagen.gpkg", layer="ssd.schulanlagen_archiv")

num_playgrounds = gdf_poi_o.shape[0]
gdf_schul = gdf_schul[gdf_schul["jahresstand"] == 2024]
gdf_poi_o["dist_min"] = gdf_poi_o.geometry.apply(lambda x: gdf_schul.distance(x).min())
gdf_poi_o_within_300m = gdf_poi_o[gdf_poi_o["dist_min"] <= int(args.num_meters)]
print(round((gdf_poi_o_within_300m.shape[0] / num_playgrounds) * 100, 2))