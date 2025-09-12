# Q: Wieviele laufende Hochbauprojekte enthalten formal unter Schutz gestellte Denkmale in ihrem Gebiet?
# Relevant datasets: ['AHB-Hochbauprojekte in Ausfuehrung OGD', 'Denkmalpflege-Inventar']

import geopandas as gpd


# Layers in dataset AHB-Hochbauprojekte in Ausfuehrung OGD: ['ahb.ahb_projekte_ogd']
# Layers in dataset Denkmalpflege-Inventar: ['afs.denkmalpflege_inventar_p']

### Solution ###
gdf_ahb_h = gpd.read_file("./data/opendata/50000006/extracted/ahb_hochbauprojekte_in_ausfuehrung_ogd.gpkg", layer="ahb.ahb_projekte_ogd")
gdf_denkm = gpd.read_file("./data/opendata/50000006/extracted/denkmalpflege_inventar.gpkg", layer="afs.denkmalpflege_inventar_p")

gdf_denkm = gdf_denkm[gdf_denkm['unterschut'] == 'Ja']
gdf_ahb_h = gdf_ahb_h[gdf_ahb_h['geometry'].intersects(gdf_denkm.geometry.union_all())]
print(gdf_ahb_h.shape[0])