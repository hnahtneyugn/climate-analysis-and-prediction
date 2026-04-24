import io
import os
import zipfile
import xarray as xr
from pathlib import Path

LAT_MIN, LAT_MAX = 10, 21
LON_MIN, LON_MAX = 103, 110
TIME_START, TIME_END = "1980-01-01", "2024-12-31"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "../data/raw"
PROCESSED_DIR  = BASE_DIR / "../data/processed"
MERGED_DIR = BASE_DIR / "../data/merged"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

def unzip_file(zip_path: str, folder_processed_path: str):
    out_path = ""
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]

    with zipfile.ZipFile(zip_path, "r") as z:
        nc_files = [f for f in z.namelist() if f.endswith(".nc")]

        for nc in nc_files:
            unique_name = f"{zip_name}_{os.path.basename(nc)}"
            out_path = os.path.join(folder_processed_path, unique_name)

            if os.path.exists(out_path):
                continue

            with z.open(nc) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

    return out_path

def merge_nc_files(folder_name, nc_paths):
    print(f"Starting merging netcdf that belongs to variable {folder_name}!")

    # Merge netcdf files in data/processed, and write the new merged netcdf file into data/merged
    ds = xr.open_mfdataset(
        paths=nc_paths,
        combine='nested',
        concat_dim='valid_time',
        preprocess=lambda ds: ds.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX),
            valid_time=slice(TIME_START, TIME_END)
        )
    )

    nc_merged_path = os.path.join(MERGED_DIR, f"{folder_name}_merged.nc")
    ds.to_netcdf(nc_merged_path)
    ds.close()

    print(f"Saved merged netcdf file of variable {folder_name}")

def main():
    for dir in os.listdir(RAW_DIR):
        nc_paths = []

        processed_path = os.path.join(PROCESSED_DIR, dir)   # Create dir for merged netcdf files
        os.makedirs(processed_path, exist_ok=True)      

        print(f"Unzipping for variable {dir}")
        # Unzip and merge netcdf files in each folder
        raw_folder_path = os.path.join(RAW_DIR, dir)
        for zip_name in os.listdir(raw_folder_path):
            zip_path = os.path.join(raw_folder_path, zip_name)
            nc_processed_path = unzip_file(zip_path=zip_path, folder_processed_path=processed_path)
            nc_paths.append(nc_processed_path)

        print(f"Finished unzipping for variable {dir}")

        merge_nc_files(folder_name=dir, nc_paths=nc_paths)

        # Cleanup in data/processed
        for path in nc_paths:
            os.remove(path)

if __name__ == "__main__":
    main()
