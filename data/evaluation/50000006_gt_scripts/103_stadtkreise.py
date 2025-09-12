# Q: Welche Stadtkreise grenzen an den Kreis {{num_kreis}} an?
# Relevant datasets: ['Stadtkreise']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_kreis", type=str)
args = parser.parse_args()
# Layers in dataset Stadtkreise: ['stzh.adm_stadtkreise_v', 'stzh.adm_stadtkreise_a', 'stzh.adm_stadtkreise_beschr_p']

### Solution ###
gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtkreise.gpkg", layer="stzh.adm_stadtkreise_v")

kreis = gdf_stadt[gdf_stadt["knr"] == int(args.num_kreis)]
touching = gdf_stadt[gdf_stadt.geometry.touches(kreis.geometry.iloc[0])]
touching_kreise = sorted(touching["knr"].tolist())
print(", ".join(map(str, touching_kreise)))