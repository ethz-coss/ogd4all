# Q: In welchem statistischen Quartier gibt es die meisten Wertstoffsammelstellen für {{sammel_type}}? Wieviele sind es?
# Relevant datasets: ['Sammelstelle', 'Statistische Quartiere']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sammel_type", type=str)
args = parser.parse_args()
# Layers in dataset Sammelstelle: ['stzh.poi_sammelstelle_att', 'stzh.poi_sammelstelle_view']
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']

### Solution ###
gdf_samme = gpd.read_file("./data/opendata/50000006/extracted/sammelstelle.gpkg", layer="stzh.poi_sammelstelle_view")
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")

if args.sammel_type.strip() == "Textilien":
    filter_col = 'textilien'
elif args.sammel_type.strip() == "Glas":
    filter_col = 'glas'
elif args.sammel_type.strip() == "Kleinmetalle":
    filter_col = 'metall'
else:
    raise ValueError(f"Unsupported sammel_type: {args.sammel_type}")
gdf_samme = gdf_samme[gdf_samme[filter_col] == 'X']
merged = gpd.sjoin(gdf_samme, gdf_stati[['qnr', 'qname', 'geometry']], how='inner', predicate='within')
merged = merged.groupby(['qnr', 'qname']).size().reset_index(name='sammelstelle_count')
merged = gdf_stati.merge(merged, on=['qnr', 'qname'], how='left').fillna({'sammelstelle_count': 0})
max_row = merged.sort_values('sammelstelle_count', ascending=False).iloc[0]
print(f"{max_row['qname']} mit {int(max_row['sammelstelle_count'])} Sammelstellen für {args.sammel_type}")