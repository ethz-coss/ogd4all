# Q: Welches ist der flächenmässig grösste Stimmkreis der Stadt Zürich? Gib auch die Fläche an, in Quadratkilometern auf eine Nachkommastelle gerundet.
# Relevant datasets: ['Stimmkreise']

import geopandas as gpd


# Layers in dataset Stimmkreise: ['stzh.adm_zaehlkreise_beschr_p', 'stzh.adm_zaehlkreise_a']

### Solution ###
gdf_stimm = gpd.read_file("./data/opendata/50000006/extracted/stimmkreise.gpkg", layer="stzh.adm_zaehlkreise_a")

gdf_stimm["area_m2"] = gdf_stimm.geometry.area
gdf_stimm["area_km2"] = gdf_stimm["area_m2"] / 1e6
gdf_stimm = gdf_stimm.sort_values(by="area_km2", ascending=False)
print(f"{gdf_stimm.iloc[0]['bezeichnung']} mit {round(gdf_stimm.iloc[0]['area_km2'], 1)} km²")