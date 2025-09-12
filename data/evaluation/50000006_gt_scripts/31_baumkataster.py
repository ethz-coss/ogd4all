# Q: Wie viele Bäume sind im Baumkataster der Stadt Zürich erfasst?
# Relevant datasets: ['Baumkataster']

import geopandas as gpd


# Layers in dataset Baumkataster: ['gsz.baumstandorte_k', 'gsz.baumkataster_kronendurchmesser', 'gsz.baumkataster_baumstandorte']

### Solution ###
gdf_baumk = gpd.read_file("./data/opendata/50000006/extracted/baumkataster.gpkg", layer="gsz.baumkataster_baumstandorte")
print(gdf_baumk.shape[0])