# Q: Was ist die häufigste Fällursache für Bäume? Wie oft tritt sie auf?
# Relevant datasets: ['Baumersatz']

import geopandas as gpd

# Layers in dataset Baumersatz: ['gsz.baumersatz']

### Solution ###
# Challenge: faellgrund and faellursache are very similar columns, faellursache is a subcategory of faellgrund.
gdf_baume = gpd.read_file("./data/opendata/50000006/extracted/baumersatz.gpkg", layer="gsz.baumersatz")
faellursache_counts = gdf_baume["faellursache"].value_counts()
most_common = faellursache_counts.idxmax()
count = faellursache_counts.max()
print(f"Die häufigste Fällursache ist '{most_common}' und sie tritt {count} mal auf.")