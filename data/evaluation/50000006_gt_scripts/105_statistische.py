# Q: Liegt das statistische Quartier {{quar_lang}} im Stadtkreis 7?
# Relevant datasets: ['Statistische Quartiere']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("quar_lang", type=str)
args = parser.parse_args()
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']

### Solution ###
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")

quar = gdf_stati[gdf_stati["qname"] == args.quar_lang.strip()]
if int(quar.iloc[0]["knr"]) == 7:
    print("Ja")
else:
    print("Nein")