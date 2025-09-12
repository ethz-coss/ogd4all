# Q: Wieviele Stimmlokale hat es im Stimmkreis '{{stimmkreis_lang}}'
# Relevant datasets: ['Stimmlokale', 'Stimmkreise']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stimmkreis_lang", type=str)
args = parser.parse_args()
# Layers in dataset Stimmlokale: ['stzh.poi_stimmlokal_att', 'stzh.poi_stimmlokal_view']
# Layers in dataset Stimmkreise: ['stzh.adm_zaehlkreise_beschr_p', 'stzh.adm_zaehlkreise_a']

### Solution ###
gdf_stimm = gpd.read_file("./data/opendata/50000006/extracted/stimmlokale.gpkg", layer="stzh.poi_stimmlokal_view")
gdf_zaehl = gpd.read_file("./data/opendata/50000006/extracted/stimmkreise.gpkg", layer="stzh.adm_zaehlkreise_a")

gdf_zaehl_filtered = gdf_zaehl[gdf_zaehl["bezeichnung"] == args.stimmkreis_lang.strip()]
gdf_stimm_filtered = gdf_stimm[gdf_stimm.geometry.within(gdf_zaehl_filtered.geometry.iloc[0])]

print(len(gdf_stimm_filtered))