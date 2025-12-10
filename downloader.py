import os
import pystac_client
import planetary_computer
import rioxarray



DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LOCATIONS = {
    "norfolk": {
        "point": {"type": "Point", "coordinates": [-76.305, 36.885]},
        "date_range": "2020-01-01/2023-12-31",
        "rgb_out": os.path.join(DATA_DIR, "train_rgb_nir.tif"),
        "dem_out": os.path.join(DATA_DIR, "train_elevation.tif"),
    },
    "miami": {
        "point": {"type": "Point", "coordinates": [-80.1918, 25.7617]},
        "date_range": "2020-01-01/2023-12-31",
        "rgb_out": os.path.join(DATA_DIR, "miami_rgb.tif"),
        "dem_out": os.path.join(DATA_DIR, "miami_dem.tif"),
    },
}


def download_city(city_name: str, config: dict) -> None:

    print(f"\n=== Downloading {city_name.capitalize()} ===")

    # Connect to catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # Find NAIP tile
    print("Searching NAIP tile...")
    search_naip = catalog.search(
        collections=["naip"],
        intersects=config["point"],
        datetime=config["date_range"],
        limit=1,
    )
    naip_items = list(search_naip.item_collection())
    if not naip_items:
        print(f"⚠ No NAIP tile found for {city_name}")
        return

    naip_item = naip_items[0]
    print(f"  Found NAIP tile: {naip_item.id}")

    # Load RGB-NIR raster
    print("Loading NAIP raster...")
    naip_ds = rioxarray.open_rasterio(naip_item.assets["image"].href)

    # Find DEM tile
    print("Searching DEM tile...")
    search_dem = catalog.search(
        collections=["3dep-seamless"],
        bbox=naip_item.bbox,
        limit=1
    )
    dem_items = list(search_dem.item_collection())
    if not dem_items:
        print(f"⚠ No DEM tile found for {city_name}")
        return

    dem_item = dem_items[0]

    print("Loading DEM raster...")
    dem_ds = rioxarray.open_rasterio(dem_item.assets["data"].href)

    print("Aligning DEM to NAIP resolution...")
    dem_aligned = dem_ds.rio.reproject_match(naip_ds)

    
    print("Saving output files...")
    naip_ds.astype("float32").rio.to_raster(config["rgb_out"])
    dem_aligned.astype("float32").rio.to_raster(config["dem_out"])

    print(f"  Saved:\n    {config['rgb_out']}\n    {config['dem_out']}")
    print(f"=== {city_name.capitalize()} complete ===\n")


def main():
    for city_name, cfg in LOCATIONS.items():
        download_city(city_name, cfg)


if __name__ == "__main__":
    main()
