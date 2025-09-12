# Q: Wie heisst die statistische Zone mit der {{sort}} Dichte an Bäumen in der Stadt Zürich? Bitte gib auch die Dichte an in Bäumen pro Quadratkilometern, auf eine ganze Zahl gerundet.
# Relevant datasets: ['Statistische Zonen', 'Baumkataster']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sort", type=str)
args = parser.parse_args()
# Layers in dataset Statistische Zonen: ['stzh.adm_statzonen_map', 'stzh.adm_statzonen_v', 'stzh.adm_statzonen_beschr_p']
# Layers in dataset Baumkataster: ['gsz.baumstandorte_k', 'gsz.baumkataster_kronendurchmesser', 'gsz.baumkataster_baumstandorte']

### Solution ###
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_zonen.gpkg", layer="stzh.adm_statzonen_v")
gdf_baumk = gpd.read_file("./data/opendata/50000006/extracted/baumkataster.gpkg", layer="gsz.baumkataster_baumstandorte")

gdf_stati['area'] = gdf_stati.geometry.to_crs(epsg=2056).area / 1e6 # CRS 2056 uses meters
merged = gpd.sjoin(gdf_baumk, gdf_stati[['stznr', 'stzname', 'geometry']], how='inner', predicate='within')
merged = merged.groupby(['stznr', 'stzname']).size().reset_index(name='tree_count')
merged = gdf_stati.merge(merged, on=['stznr', 'stzname'], how='left').fillna({'tree_count': 0})

merged['tree density'] = merged['tree_count'] / merged['area']
if args.sort.strip().lower() not in ['tiefsten', 'höchsten']:
    raise ValueError("Argument 'sort' must be either 'tiefsten' or 'höchsten'.")
ascending = True if args.sort.strip().lower() == 'tiefsten' else False
desired_density_row = merged.sort_values('tree density', ascending=ascending).iloc[0]

print(f"{desired_density_row['stzname']} mit {round(desired_density_row['tree density'])} Bäumen pro Quadratkilometer")