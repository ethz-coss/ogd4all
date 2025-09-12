# Q: Wieviele automatische Zählstellen für Fussgänger und Velos in Zürich sind aktuell im Einsatz?
# Relevant datasets: ['Standorte der automatischen Fuss- und Velozählungen']

import geopandas as gpd


# Layers in dataset Standorte der automatischen Fuss- und Velozählungen: ['taz.view_eco_standorte']

### Solution ###
# Challenge: Has to read from documentation that null value for bis means current
# Suppress runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

gdf_stand = gpd.read_file("./data/opendata/50000006/extracted/standorte_der_automatischen_fuss_und_velozaehlungen.gpkg", layer="taz.view_eco_standorte")

aktuelle_zaehler = gdf_stand[gdf_stand["bis"].isnull()]
print(aktuelle_zaehler.shape[0])