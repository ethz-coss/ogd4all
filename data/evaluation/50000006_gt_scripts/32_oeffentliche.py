# Q: Wieviele Strassen-, Wegleuchten und ähnliche gibt es gesamthaft in Zürich?
# Relevant datasets: ['Öffentliche Beleuchtung der Stadt Zürich']

import geopandas as gpd


# Layers in dataset Öffentliche Beleuchtung der Stadt Zürich: ['ewz.ewz_brennstelle_p']

### Solution ###
gdf_oeffe = gpd.read_file("./data/opendata/50000006/extracted/oeffentliche_beleuchtung_der_stadt_zuerich.gpkg", layer="ewz.ewz_brennstelle_p")

print(gdf_oeffe.shape[0])