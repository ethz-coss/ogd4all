# Q: Wie heisst das statistische Quartier ohne öffentlichen Spielplätze?
# Relevant datasets: ['POI öffentliche Spielplätze', 'Statistische Quartiere']

import geopandas as gpd


# Layers in dataset POI öffentliche Spielplätze: ['stzh.poi_spielplatz_att', 'stzh.poi_spielplatz_view']
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']

### Solution ###
gdf_poi_o = gpd.read_file("./data/opendata/50000006/extracted/poi_oeffentliche_spielplaetze.gpkg", layer="stzh.poi_spielplatz_view")
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")

gdf_stati['area'] = gdf_stati.geometry.to_crs(epsg=2056).area / 1e6
merged = gpd.sjoin(gdf_poi_o, gdf_stati[['qnr', 'qname', 'geometry']], how='inner', predicate='within')
merged = merged.groupby(['qnr', 'qname']).size().reset_index(name='spielplatz_count')
merged = gdf_stati.merge(merged, on=['qnr', 'qname'], how='left').fillna({'spielplatz_count': 0})
merged_no_playgrounds = merged[merged['spielplatz_count'] == 0]
print(merged_no_playgrounds['qname'].str.cat(sep=', '))