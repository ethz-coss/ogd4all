# Q: Wieviele Kilometer der Strassen in Zürich verfügen über einen beidseitigen Velostreifen? Runde auf ganze Kilometer.
# Relevant datasets: ['Fuss- und Velowegnetz']

import geopandas as gpd


# Layers in dataset Fuss- und Velowegnetz: ['taz_mm.tbl_routennetz', 'taz_mm.tbl_routennetz_abbiegeverbote']

### Solution ###
gdf_fuss_ = gpd.read_file("./data/opendata/50000006/extracted/fuss_und_velowegnetz.gpkg", layer="taz_mm.tbl_routennetz")

gdf_fuss_both = gdf_fuss_[gdf_fuss_["velostreifen"] == "BOTH"]
print(round(gdf_fuss_both.length.sum() / 1000))