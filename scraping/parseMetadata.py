import xml.etree.ElementTree as ET
import os
import json
import argparse
import logging

"""
Parse XML metadata file from Geocat and extract the following information:
- Title
- Keywords
- Metadata date
- Update frequency
- Purpose
- Abstract
- Identification date
- Identification date type
- Responsible party
- Contact person
- Reference system
- Geographic bounding box
- Available file formats
- Download URL
- Classes
- Attributes

Saves the extracted information as a JSON file in folder "processed"

:param path: The path to the XML file to be processed
"""
def process_file(path):
    # Dictionary with processed output
    dict = {}

    # Define namespaces
    namespaces = {
        'gmd': 'http://www.isotc211.org/2005/gmd',
        'gco': 'http://www.isotc211.org/2005/gco',
        'che': 'http://www.geocat.ch/2008/che',
    }

    tree = ET.parse(path, parser=ET.XMLParser(encoding="iso-8859-1"))
    root = tree.getroot()

    # -- Title --
    # Find the <gco:CharacterString> element inside <gmd:hierarchyLevelName>
    title_elem = root.find(
        ".//gmd:hierarchyLevelName/gco:CharacterString", 
        namespaces
    )
    if title_elem is not None and title_elem.text:
        dict["title"] = title_elem.text
    else:
        logging.info("The title element was not found or is empty.")

    # -- Keywords --
    # Find the <gco:CharacterString> element inside <gmd:descriptiveKeywords>
    keyword_elem = root.find(
        ".//gmd:descriptiveKeywords/gmd:MD_Keywords/gmd:keyword/gco:CharacterString", 
        namespaces
    )
    if keyword_elem is not None and keyword_elem.text:
        # Split the text on commas and remove any extra whitespace
        keywords = [kw.strip() for kw in keyword_elem.text.split(',')]
        dict["keywords"] = keywords # Note: many metadata files don't have keywords, so we don't logging.info a warning if this is the case

    ## -- Date --
    # Find the <gmd:dateStamp> element and extract gco:DateTime value
    date_elem = root.find('.//gmd:dateStamp/gco:DateTime', namespaces)
    if date_elem is not None and date_elem.text:
        dict["metadata_date"] = date_elem.text
    else:
        logging.info(f"{dict['title']} Date element not found in XML.")

    ## -- Update frequency --
    update_elem = root.find('.//gmd:maintenanceAndUpdateFrequency/gmd:MD_MaintenanceFrequencyCode', namespaces)
    if update_elem is not None and 'codeListValue' in update_elem.attrib:
        dict["updateFrequency"] = update_elem.attrib['codeListValue'].strip()
    else:
        logging.info(f"{dict['title']} Update frequency not found in XML.")

    ## -- Purpose --
    purpose_elem = root.find('.//gmd:purpose', namespaces)
    purpose_texts = []
    if purpose_elem is not None:
        # Look for all <gmd:LocalisedCharacterString>
        for text_elem in purpose_elem.findall('.//gmd:LocalisedCharacterString', namespaces):
            if text_elem.text:
                purpose_texts.append(text_elem.text.strip())
        # If none found, try direct <gco:CharacterString> children
        if not purpose_texts:
            for text_elem in purpose_elem.findall('.//gco:CharacterString', namespaces):
                if text_elem.text:
                    purpose_texts.append(text_elem.text.strip())
        if purpose_texts:
            dict["purpose"] = " ".join(purpose_texts)
        else:
            logging.info(f"{dict['title']} Purpose text not found in XML.")
    else:
        logging.info(f"{dict['title']} Purpose element not found in XML.")

    ## -- Abstract --
    abstract_elem = root.find('.//gmd:abstract/gco:CharacterString', namespaces)
    if abstract_elem is not None and abstract_elem.text:
        dict["abstract"] = abstract_elem.text.strip()
    else:
        logging.info(f"{dict['title']} Abstract not found in XML.")
        dict["abstract"] = None


    ## -- Date and Interaction --
    id_info_elem = root.find('.//gmd:identificationInfo/che:CHE_MD_DataIdentification', namespaces)
    if id_info_elem is not None:
        # Look for the first gco:Date inside CHE_MD_DataIdentification
        date_elem = id_info_elem.find('.//gco:Date', namespaces)
        dict["identificationDate"] = date_elem.text.strip() if date_elem is not None and date_elem.text else None
        if dict["identificationDate"] is None:
            logging.info(f"{dict['title']} Identification date not found in XML.")
        # To find the corresponding date type, look within the same CI_Date block
        ci_date_elem = id_info_elem.find('.//gmd:CI_Date', namespaces)
        if ci_date_elem is not None:
            date_type_elem = ci_date_elem.find('.//gmd:dateType/gmd:CI_DateTypeCode', namespaces)
            dict["identificationDateType"] = date_type_elem.attrib["codeListValue"].strip() if date_type_elem is not None and "codeListValue" in date_type_elem.attrib else None
            if dict["identificationDateType"] is None:
                logging.info(f"{dict['title']} Identification date type not found in XML.")
        else:
            logging.info(f"{dict['title']} CI_Date element not found in CHE_MD_DataIdentification.")
    else:
        logging.info(f"{dict['title']} CHE_MD_DataIdentification not found in XML.")

    ## -- Responsible Party and Person --
    contact_elem = root.find('.//gmd:contact/che:CHE_CI_ResponsibleParty', namespaces)
    if contact_elem is not None:
        # Get name of responsible party
        org_elem = contact_elem.find('.//gmd:organisationName/gco:CharacterString', namespaces)
        if org_elem is not None and org_elem.text:
            dict["responsibleParty"] = org_elem.text.strip()
        else:
            logging.info(f"{dict['title']} responsibleParty (organisationName) not found in XML.")

        # Get contact person by combining first and last name
        first_name_elem = contact_elem.find('.//che:individualFirstName/gco:CharacterString', namespaces)
        last_name_elem = contact_elem.find('.//che:individualLastName/gco:CharacterString', namespaces)
        first_name = first_name_elem.text.strip() if first_name_elem is not None and first_name_elem.text else ""
        last_name = last_name_elem.text.strip() if last_name_elem is not None and last_name_elem.text else ""
        if first_name or last_name:
            dict["contactPerson"] = f"{first_name} {last_name}".strip()
        else:
            logging.info(f"{dict['title']} contactPerson (first or last name) not found in XML.")
    else:
        logging.info(f"{dict['title']} Contact information not found in XML.")

    ## -- Reference System --
    # For now, just concatenate all text
    refsys_elem = root.find('.//gmd:referenceSystemInfo', namespaces)
    if refsys_elem is not None:
        refsys_texts = [elem.text.strip() for elem in refsys_elem.iter() if elem.text and elem.text.strip()]
        dict["referenceSystem"] = " ".join(refsys_texts)
    else:
        logging.info(f"{dict['title']} Reference system info not found in XML.")

    ## -- Geographic Bounding Box --
    # Find the <gmd:EX_GeographicBoundingBox> element
    bbox_elem = root.find('.//gmd:EX_GeographicBoundingBox', namespaces)
    # Save west, east, south, north bounds
    bbox = {}
    if bbox_elem is not None:
        for bound in ['westBoundLongitude', 'eastBoundLongitude', 'southBoundLatitude', 'northBoundLatitude']:
            bound_elem = bbox_elem.find(f'.//gmd:{bound}/gco:Decimal', namespaces)
            if bound_elem is not None and bound_elem.text:
                bbox[bound] = float(bound_elem.text)
            else:
                logging.info(f"{bound} not found in XML.")
    else:
        logging.info(f"{dict['title']} Bounding box not found in XML.")
    dict["boundingBox"] = bbox

    ## -- Available File Formats --
    distribution_formats = []
    distribution_format_elems = root.findall('.//gmd:distributionInfo/gmd:MD_Distribution/gmd:distributionFormat', namespaces)
    if distribution_format_elems:
        for df in distribution_format_elems:
            name_elem = df.find('.//gmd:MD_Format/gmd:name/gco:CharacterString', namespaces)
            if name_elem is not None and name_elem.text:
                distribution_formats.append(name_elem.text.strip())
        dict["distributionFormats"] = distribution_formats # Note: many geodata files have no distribution formats since "Datensatz mit Nutzungsbeschränkungen"

    ## -- Download URL --
    download_url_elem = root.find('.//gmd:distributionInfo/gmd:MD_Distribution//gmd:URL', namespaces)
    if download_url_elem is not None and download_url_elem.text:
        dict["downloadURL"] = download_url_elem.text.strip()

    ## -- Attributes --
    # Find the <gmd:contentInfo> element
    content_info_elem = root.find('.//gmd:contentInfo/che:CHE_MD_FeatureCatalogueDescription', namespaces)
    if content_info_elem is not None:
        # Parse includedWithDataset as a boolean
        included_elem = content_info_elem.find('.//gmd:includedWithDataset/gco:Boolean', namespaces)
        if included_elem is not None and included_elem.text:
            try:
                descriptions_included = bool(int(included_elem.text.strip()))
            except ValueError:
                descriptions_included = None
                logging.info(f"{dict['title']} unable to convert includedWithDataset value to boolean.")
        else:
            logging.info(f"{dict['title']} includedWithDataset not found in contentInfo.")
            descriptions_included = None

        # Parse class information under che:class/che:CHE_MD_Class elements
        classes = []
        for class_elem in content_info_elem.findall('.//che:class/che:CHE_MD_Class', namespaces):
            class_name_elem = class_elem.find('.//che:name/gco:CharacterString', namespaces)
            class_desc_elem = class_elem.find('.//che:description/gco:CharacterString', namespaces)
            class_info = {
            "name": class_name_elem.text.strip() if class_name_elem is not None and class_name_elem.text else None,
            "description": class_desc_elem.text.strip() if class_desc_elem is not None and class_desc_elem.text else None
            }

            # FOR EVERY CLASS:
            # Parse attributes under che:attribute elements
            class_attributes = []
            for attr_elem in class_elem.findall('.//che:attribute', namespaces):
                attr_name_elem = attr_elem.find('.//che:name/gco:CharacterString', namespaces)
                attr_desc_elem = attr_elem.find('.//che:description/gco:CharacterString', namespaces)
                attr = {
                    "name": attr_name_elem.text.strip() if attr_name_elem is not None and attr_name_elem.text else None,
                    "description": attr_desc_elem.text.strip() if attr_desc_elem is not None and attr_desc_elem.text else None
                }
                class_attributes.append(attr)

            class_info["attributes"] = class_attributes
            classes.append(class_info)

        if not classes:
            logging.info(f"{dict['title']} No class information found in contentInfo.")


        dict["contentInfo"] = {
            "descriptionsIncludedWithDataset": descriptions_included,
            "classes": classes,
        }
    else:
        logging.info(f"{dict['title']} Content info not found in XML.")

    # Save the dictionary as JSON in subfolder processed with same name as XML file
    input_dir = os.path.dirname(path)
    output_dir = os.path.join(input_dir, "../processed")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(dict, json_file, ensure_ascii=False, indent=3)

"""
Process all XML files in a directory

:param directory: The directory containing the XML files to be processed
"""
def process_directory(directory):
    xml_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                process_file(file_path)
                xml_count += 1
    logging.info(f"Processed {xml_count} XML files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse XML metadata from Geocat to JSON")
    parser.add_argument("--groupOwner", type=str, default="50000006",help="The groupOwner id where XML data should be parsed.")
    parser.add_argument("--verbose", action="store_true", help="Print more information during processing.")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO if args.verbose else logging.WARNING)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_dir = os.path.join(script_dir, f"../data/metadata/{args.groupOwner}/raw")
    process_directory(xml_dir)

