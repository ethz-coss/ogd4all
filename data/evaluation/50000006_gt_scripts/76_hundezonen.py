# Q: Gilt die Leinenpflicht {{addr_desc}} den ganzen Tag?
# Relevant datasets: ['Hundezonen']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr_desc", type=str)
args = parser.parse_args()
# Layers in dataset Hundezonen: ['gsz.hundezone']

### Solution ###
# Suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
gdf_hunde = gpd.read_file("./data/opendata/50000006/extracted/hundezonen.gpkg", layer="gsz.hundezone")

if args.addr_desc.strip() == "beim Zürichhorn":
    zonenname = "Zürichhorn"
elif args.addr_desc.strip() == "bei der Landiwiese":
    zonenname = "Landiwiese"
elif args.addr_desc.strip() == "im Belvoirpark":
    zonenname = "Belvoirpark"
else:
    raise ValueError("Unknown address description")

gdf_filtered = gdf_hunde[gdf_hunde["zonenname"] == zonenname]
print(f"Nein, nur von {gdf_filtered["zusatzbedingung"].iloc[0]}.")