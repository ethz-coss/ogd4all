# Q: Was ist der deutsche Name der häufigsten Baumart in der Stadt Zürich? Wie viele Bäume dieser Art gibt es in der Stadt?
# Relevant datasets: ['Baumkataster']

import geopandas as gpd

# Layers in dataset Baumkataster: ['gsz.baumstandorte_k', 'gsz.baumkataster_kronendurchmesser', 'gsz.baumkataster_baumstandorte']

### Solution ###
gdf_baumk = gpd.read_file("./data/opendata/50000006/extracted/baumkataster.gpkg", layer="gsz.baumkataster_baumstandorte")
# Group by baumnamedeu and count occurrences
baum_counts = gdf_baumk['baumnamedeu'].value_counts()
most_common_tree = baum_counts.idxmax()
print(f"{most_common_tree}: {baum_counts.max()}")