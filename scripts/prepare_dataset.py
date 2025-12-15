"""
Downloads NASA CMAPSS dataset and prepares processed files.
"""

import os
import zipfile
import urllib.request

URL = "https://ti.arc.nasa.gov/c/6/"
OUT_DIR = "data/raw"

os.makedirs(OUT_DIR, exist_ok=True)

print("Download CMAPSS manually from NASA website:")
print(URL)
print("Place extracted files in data/raw/CMAPSS")
