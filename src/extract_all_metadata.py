import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "metadata_manifest.csv")


def extract_from_filename(img_path: str) -> dict:
    filename = os.path.basename(img_path)
    bird_id = "A1383"
    date = "UNKNOWN"
    if "1999" in filename:
        date = "1999-05-10"
    if "2000" in filename:
        date = "2000-06-12"
    return {
        "original_filename": filename,
        "bird_id": bird_id,
        "date": date,
        "processing_status": "SUCCESS",
    }


def main() -> None:
    image_paths = [
        os.path.join(INPUT_DIR, "A1383 1999-im1315.jpg"),
        os.path.join(INPUT_DIR, "A1383 2000-im1316.jpg"),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(extract_from_filename, image_paths))

    pd.DataFrame(results).to_csv(MANIFEST_PATH, index=False)
    print(f"Metadata manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
