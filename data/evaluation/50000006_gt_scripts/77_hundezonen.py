# Q: Auf wieviel Quadratkilometern der Stadt Zürich gilt {{area_type}}? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['Hundezonen']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("area_type", type=str)
args = parser.parse_args()
# Layers in dataset Hundezonen: ['gsz.hundezone']

### Solution ###
# Suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
gdf_hunde = gpd.read_file("./data/opendata/50000006/extracted/hundezonen.gpkg", layer="gsz.hundezone")

if args.area_type.strip() == "Hundeverbot":
    einschraenkungen = "Hundeverbot"
elif "Leinenpflicht" in args.area_type.strip():
    einschraenkungen = "Leinenpflicht"
else:
    raise ValueError("Unknown area type")

gdf_hunde = gdf_hunde[gdf_hunde["einschraenkungen"] == einschraenkungen]
area = gdf_hunde.geometry.area.sum() / 1e6  # Convert from m² to km²
print(f"{round(area, 2)} km²")