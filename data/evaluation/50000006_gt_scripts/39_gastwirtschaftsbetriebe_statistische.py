# Q: Wie heist die statistische Zone mit der höchsten Dichte an Gastwirtschaftsbetrieben, und wie hoch ist sie? Runde auf eine Nachkommastelle und nutze Gastwirtschaftsbetrieb/km² als Einheit.
# Relevant datasets: ['Gastwirtschaftsbetriebe', 'Statistische Zonen']

import geopandas as gpd


# Layers in dataset Gastwirtschaftsbetriebe: ['stp.gastwirtschaftsbetriebe']
# Layers in dataset Statistische Zonen: ['stzh.adm_statzonen_map', 'stzh.adm_statzonen_v', 'stzh.adm_statzonen_beschr_p']

### Solution ###
gdf_gastw = gpd.read_file("./data/opendata/50000006/extracted/gastwirtschaftsbetriebe.gpkg", layer="stp.gastwirtschaftsbetriebe")
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_zonen.gpkg", layer="stzh.adm_statzonen_v")

gdf_stati['area'] = gdf_stati.geometry.to_crs(epsg=2056).area / 1e6 # CRS 2056 uses meters
merged = gpd.sjoin(gdf_gastw, gdf_stati[['stznr', 'stzname', 'geometry']], how='inner', predicate='within')
merged = merged.groupby(['stznr', 'stzname']).size().reset_index(name='rest_count')
merged = gdf_stati.merge(merged, on=['stznr', 'stzname'], how='left').fillna({'rest_count': 0})
merged['rest density'] = merged['rest_count'] / merged['area']
max_density_row = merged.sort_values('rest density', ascending=False).iloc[0]
print(f"{max_density_row['stzname']} mit {round(max_density_row['rest density'], 1)} Gastwirtschaftsbetrieben pro Quadratkilometer")