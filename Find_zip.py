# find_zip.py - Quick environment diagnostic
# Run this FIRST before inspect_cama.py
# It tells you exactly where Python is looking and what files exist

import os
import sys

print("=" * 60)
print("ENVIRONMENT DIAGNOSTIC")
print("=" * 60)

# 1. Where is Python running FROM
print(f"\n1. Python is running from this folder:")
print(f"   {os.getcwd()}")

# 2. What files are in that folder
print(f"\n2. Files in that folder:")
for f in sorted(os.listdir(".")):
    size = os.path.getsize(f)
    if size > 1024*1024:
        size_str = f"{size/(1024*1024):.1f} MB"
    elif size > 1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size} bytes"
    print(f"   {f:<45} {size_str}")

# 3. Specifically check for ZIP
print(f"\n3. ZIP file check:")
if os.path.exists("ACPA_CAMAData.zip"):
    size = os.path.getsize("ACPA_CAMAData.zip")
    print(f"   FOUND — ACPA_CAMAData.zip ({size/(1024*1024):.1f} MB)")
else:
    print(f"   NOT FOUND — ACPA_CAMAData.zip is not in this folder")
    # Search common locations
    print(f"\n4. Searching for any .zip files nearby...")
    search_paths = [
        ".",
        "..",
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
    ]
    found_any = False
    for path in search_paths:
        try:
            for f in os.listdir(path):
                if f.endswith(".zip"):
                    full = os.path.join(path, f)
                    size = os.path.getsize(full)
                    print(f"   Found: {full}  ({size/(1024*1024):.1f} MB)")
                    found_any = True
        except:
            pass
    if not found_any:
        print("   No .zip files found in common locations")

# 4. Python version
print(f"\n5. Python version: {sys.version}")
print("=" * 60)