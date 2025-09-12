# Q: Was ist die Objekt ID der flächenmässig grössten Liegenschaft in Zürich Ende 2023?
# Relevant datasets: ['Amtliche Vermessungsdaten Stadt Zürich Jahresendstand 2023']

import geopandas as gpd


# Layers in dataset Amtliche Vermessungsdaten Stadt Zürich Jahresendstand 2023: ['geoz_2023.av_bo_bbnachfuehrung', 'geoz_2023.av_ei_allegebaeude_a_v', 'geoz_2023.av_geb_strassenstueck_ap', 'geoz_2023.av_ge_gemeindegrenze_boundary', 'geoz_2023.av_ro_linienelement', 'geoz_2023.av_bo_gebaeudenummer_t', 'geoz_2023.av_bo_boflaeche_l', 'geoz_2023.av_ge_gemeindegrenze_a', 'geoz_2023.av_bo_objektname_t', 'geoz_2023.av_li_projliegenschaft_a', 'geoz_2023.av_li_selbstrecht_a', 'geoz_2023.av_no_flurname_uep5_t', 'geoz_2023.av_geb_strassennamen_uep5_t', 'geoz_2023.av_geb_strassenstueck', 'geoz_2023.av_ge_hoheitsgrenzpunkt_t', 'geoz_2023.av_bo_allegebaeude_a_v', 'geoz_2023.av_fi_lfp1_t', 'geoz_2023.av_ei_linienelement', 'geoz_2023.av_bo_projboflaeche_a', 'geoz_2023.av_fi_hfp1', 'geoz_2023.av_ei_objektname_uep5_t', 'geoz_2023.av_ei_objektnummer_t', 'geoz_2023.av_geb_gebaeudeadresse_t', 'geoz_2023.av_fi_lfp2_t', 'geoz_2023.av_bo_projgebaeudenummer_t', 'geoz_2023.av_fi_hfp3_t', 'geoz_2023.av_fi_hfp1_t', 'geoz_2023.av_fi_hfp3', 'geoz_2023.av_ge_hoheitsgrenzpunkt', 'geoz_2023.av_fi_lfp1', 'geoz_2023.av_li_grenzpunkt_t', 'geoz_2023.av_bo_objektname_uep5_t', 'geoz_2023.av_geb_kurzstrnamen_uep5_t', 'geoz_2023.av_li_projselbstrecht_a', 'geoz_2023.av_plz_einteilung_a', 'geoz_2023.av_geb_gebaeudeadresse_uep5_t', 'geoz_2023.av_fi_lfp3', 'geoz_2023.av_bo_boflaeche_a', 'geoz_2023.av_fi_hfp2_t', 'geoz_2023.av_li_projgrundstueck_t', 'geoz_2023.av_fi_lfp3_t', 'geoz_2023.av_be_bezirksgrenzabschnitt', 'geoz_2023.av_li_grenzpunkt', 'geoz_2023.av_geb_strassennamen_t', 'geoz_2023.av_li_grundstueck_t', 'geoz_2023.av_no_flurname_t', 'geoz_2023.av_li_liegenschaft_a', 'geoz_2023.av_ei_objektname_t', 'geoz_2023.av_bo_boflaechesymbol', 'geoz_2023.av_ei_flaechenelement_a', 'geoz_2023.av_fi_lfp2', 'geoz_2023.av_fi_hfp2']

### Solution ###
# Challenge: recognize area is available as a column in the dataset, but can also be solved by using the geometry column and calculating the area
gdf_amtli = gpd.read_file("./data/opendata/50000006/extracted/amtliche_vermessungsdaten_stadt_zuerich_jahresendstand_2023.gpkg", layer="geoz_2023.av_li_liegenschaft_a")

gdf_amtli = gdf_amtli.sort_values(by="flaechenmass", ascending=False)
largest_property = gdf_amtli.iloc[0]
print(largest_property["objid"])