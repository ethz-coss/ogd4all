# Q: Wieviele Kilometer sogenannter 'erholungsorientierter Fusswege' sind markiert als zu Fuss nicht begehbar? Runde auf ganze Kilometer.
# Relevant datasets: ['Fuss- und Velowegnetz']

import geopandas as gpd


# Layers in dataset Fuss- und Velowegnetz: ['taz_mm.tbl_routennetz', 'taz_mm.tbl_routennetz_abbiegeverbote']

### Solution ###
gdf_fuss_ = gpd.read_file("./data/opendata/50000006/extracted/fuss_und_velowegnetz.gpkg", layer="taz_mm.tbl_routennetz")

gdf_fuss = gdf_fuss_[gdf_fuss_["map_fuss"] == 1]
gdf_fuss_begehbar = gdf_fuss[gdf_fuss["fuss"] == 1]
print(round((gdf_fuss.length.sum() - gdf_fuss_begehbar.length.sum()) / 1000))