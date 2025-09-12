import requests
import os
import argparse
import logging

"""
Get metadata records from the Geocat API for a specific groupOwner.
:param groupOwner: The groupOwner to filter metadata records by.
:param max_hits: The maximum number of hits to return.

:return: List of indexed search hits as returned by the Geocat API
"""
def get_groupowner_hits(groupOwner, max_hits=500):
    # URL for the msearch endpoint (adjust if necessary)
    url = "https://www.geocat.ch/geonetwork/srv/api/search/records/_search"

    # HTTP headers; note that the API expects JSON content.
    headers = {
        "Content-Type": "application/json"
    }

    # JSON body for the POST request, bool query with filter for groupOwner and ignoring all template data
    payload = {
        "query": {
            "bool": {
                "must": [],
                "should": [],
                "filter": [
                    {
                    "terms": {
                        "isTemplate": [
                            "n"
                        ]
                    }
                    },
                    {
                    "query_string": {
                        "query": f"groupOwner:(\"{groupOwner}\")"
                    }
                    }
                ]
            }
        },
        "from": 0,
        "size": max_hits
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx/5xx)
        data = response.json()

        # Log the number of hits
        logging.info(f"Number of hits for groupOwner {groupOwner}: {data['hits']['total']['value']}")

        return data['hits']['hits']
    except requests.exceptions.RequestException as e:
        logging.error("An error occurred in get_groupowner_hits:", e)
        return []

"""
Download metadata records from the Geocat API and save them to files at scriptlocation/../data/metadata/groupOwner/raw.
:param hits: List of search hits to download, as returned by get_groupowner_hits

:return: None
"""
def download_metadata(groupOwner, hits):
    headers = {
        "Accept": "application/xml"
    }

    skip_count = 0
    # Metadata should be saved in ../data/metadata/groupOwner/raw relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f"../data/metadata/{groupOwner}/raw")
    os.makedirs(save_dir, exist_ok=True)

    for hit in hits:
        metadata_id = hit['_id']
        metadata_url = f"https://www.geocat.ch/geonetwork/srv/api/records/{metadata_id}"

        file_path = os.path.join(save_dir, f"{metadata_id}.xml")
        if os.path.exists(file_path):
            skip_count += 1
            continue

        try:
            response = requests.get(metadata_url, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx/5xx)
            
            with open(file_path, "w") as f:
                f.write(response.text)

        except requests.exceptions.RequestException as e:
            logging.error("An error occurred in download_metadata:", e)

    logging.info(f"Downloaded {len(hits) - skip_count} metadata records, skipped {skip_count} existing records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download metadata records from the Geocat API.")
    parser.add_argument("--groupOwner", type=str, default="50000006",help="The groupOwner id to filter metadata records by.")
    parser.add_argument("--max_hits", type=int, default=500, help="The maximum number of hits to return (default: 500).")
    parser.add_argument("--verbose", action="store_true", help="Print more information during processing.")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO if args.verbose else logging.WARNING)

    hits = get_groupowner_hits(groupOwner=args.groupOwner, max_hits=args.max_hits)
    download_metadata(args.groupOwner, hits)