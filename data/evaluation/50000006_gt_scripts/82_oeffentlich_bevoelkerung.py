# Q: Wieviele Einwohner (Stand 2024, wirtschaftliche Wohnbevölkerung) kommen in der Stadt Zürich auf einen öffentlichen Strassenparkplatz? Runde auf eine ganze Zahl.
# Relevant datasets: ['Öffentlich zugängliche Strassenparkplätze OGD', 'Bevölkerung der Stadt Zürich']

import geopandas as gpd

# Layers in dataset Öffentlich zugängliche Strassenparkplätze OGD: ['taz.view_pp_ogd']
# Layers in dataset Bevölkerung der Stadt Zürich: ['bev324od3243']

### Solution ###
gdf_oeffe = gpd.read_file("./data/opendata/50000006/extracted/oeffentlich_zugaengliche_strassenparkplaetze_ogd.gpkg", layer="taz.view_pp_ogd")
gdf_bev32 = gpd.read_file("./data/opendata/50000006/extracted/bev324od3243.csv", layer="bev324od3243", encoding="utf-8")

bev_count = int(gdf_bev32[gdf_bev32["StichtagDatJahr"] == '2024']['AnzBestWir'].values[0])
parkplaetze_count = gdf_oeffe.shape[0]

print(f"{round(bev_count / parkplaetze_count)} Einwohner pro Parkplatz")