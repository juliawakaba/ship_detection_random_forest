import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import os
    from pathlib import Path
    import laspy
    import numpy as np
    from scipy.interpolate import griddata
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.plot import show as rshow
    from affine import Affine
    import scipy.ndimage
    import geopandas as gpd
    from shapely.strtree import STRtree
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from rasterstats import zonal_stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import pandas as pd
    import nickyspatial as ns


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ship Detection in Harbor Imagery
    ### Object-Based Image Analysis · RGB + nDSM: Belgium 2011

    - **Segmentation**: Done on eCognition using Multiresolution Segmentation on ortho photo RGB + nDSM
    - **Feature extraction**: Done to obtain spectral means, NDWI, shape metrics, height stats
    - **Classification**: Random Forest trained on digitised sample points
    - **Further Spatial Classification**; Classified the detected ships as SHIP_WATER vs SHIP_DOCK using adjacency rules
    - **Export**: GeoPackage with ship vessel attributes
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Install Dependencies
    Run this while inside your virtual environment:

    ```bash
    pip install nickyspatial rasterio scikit-learn geopandas scipy affine laspy rasterstats shapely tabulate
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setting the File Paths

    `SAMPLES_PATH` is the training points GeoPackage digitised in QGIS
    (needs a field called **`class`** with values: `SHIP`, `WATER`, `DOCK`).
    """)
    return


@app.cell
def _():
    # Set project root
    BASE_DIR = Path(__file__).resolve().parent

    RGB_PATH      = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315135_56865_UTM31N.tif"
    LAS_PATH      = BASE_DIR / "data_store" / "BE_LIDAR_27032011_315135_56865_UTM31N.las"
    DSM_PATH      = BASE_DIR / "data_store" / "BE_DSM_27032011_315135_56865_UTM31N.tif"
    DTM_PATH      = BASE_DIR / "data_store" / "BE_DTM_315135_56865_generated.tif"
    nDSM_PATH     = BASE_DIR / "data_store" / "BE_nDSM_315135_56865.tif"
    SEGMENTS_PATH = BASE_DIR / "ecognition_segmentation" / "segmentation_315135" / "300_315135_segmentation.shp"
    SAMPLES_PATH  = BASE_DIR / "315135_samples" / "315135_sample_points.gpkg"
    OUTPUT_PATH   = BASE_DIR / "output" / "ships_detected.gpkg"
    return (
        BASE_DIR,
        DSM_PATH,
        DTM_PATH,
        LAS_PATH,
        OUTPUT_PATH,
        RGB_PATH,
        SAMPLES_PATH,
        SEGMENTS_PATH,
        nDSM_PATH,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate DTM from LiDAR

    Derives the bare-earth DTM from the `.las` point cloud.
    Falls back to 10th percentile if no ground points found.
    If interpolation is noisy (harbour is flat), switches to a flat DTM at 40.70m.
    This step is skipped automatically if file already exists.


    this process takes a few seconds, I just had to give it some time like less than 1 minute to run
    """)
    return


@app.cell
def _():
    def generate_dtm(

        las_path: Path,

        dsm_path: Path,

        dtm_path: Path,

        interp_scale: float = 0.1,

        flat_fallback_elev: float = 40.70,

        noise_std_threshold: float = 1.0,

        max_ground_points: int = 200_000,

    ) -> str:

        """Derive a bare-earth DTM from a LiDAR point cloud, matched to

        dsm_path's grid, and write it to dtm_path. Returns a description of

        the method actually used."""

        _las = laspy.read(las_path)

        _gx = _las.x[_las.classification == 2]

        _gy = _las.y[_las.classification == 2]

        _gz = _las.z[_las.classification == 2]


        if len(_gx) == 0:

            _z_thresh = np.percentile(_las.z, 10)

            _mask2    = _las.z <= _z_thresh

            _gx, _gy, _gz = _las.x[_mask2], _las.y[_mask2], _las.z[_mask2]


        with rasterio.open(dsm_path) as _s:

            _H, _W = _s.shape

            _left, _bottom, _right, _top = _s.bounds

            _profile = _s.profile


        if len(_gx) > max_ground_points:

            _idx = np.random.choice(len(_gx), max_ground_points, replace=False)

            _gx, _gy, _gz = _gx[_idx], _gy[_idx], _gz[_idx]


        _H_lo, _W_lo = max(int(_H * interp_scale), 2), max(int(_W * interp_scale), 2)

        _xs_lo = np.linspace(_left, _right, _W_lo)

        _ys_lo = np.linspace(_top, _bottom, _H_lo)

        _gx_lo, _gy_lo = np.meshgrid(_xs_lo, _ys_lo)


        _dtm_lo = griddata(

            np.column_stack([_gx, _gy]), _gz,

            (_gx_lo, _gy_lo), method="linear", fill_value=np.nan

        )


        _nan_mask_lo = np.isnan(_dtm_lo)

        if _nan_mask_lo.any():

            _fill_idx = scipy.ndimage.distance_transform_edt(

                _nan_mask_lo, return_distances=False, return_indices=True

            )

            _dtm_filled_lo = _dtm_lo[tuple(_fill_idx)]

        else:

            _dtm_filled_lo = _dtm_lo


        _dtm_std = float(np.std(_dtm_filled_lo))

        if _dtm_std > noise_std_threshold:

            _dtm_final  = np.full((_H, _W), fill_value=flat_fallback_elev, dtype=np.float32)

            _method_str = f"Flat DTM at {flat_fallback_elev}m (interpolation noisy)"

        else:

            _zoom_factors = (_H / _dtm_filled_lo.shape[0], _W / _dtm_filled_lo.shape[1])

            _dtm_final = scipy.ndimage.zoom(_dtm_filled_lo, _zoom_factors, order=1).astype(np.float32)

            if _dtm_final.shape != (_H, _W):

                _tmp = np.full((_H, _W), float(_dtm_final.mean()), dtype=np.float32)

                _h_c, _w_c = min(_H, _dtm_final.shape[0]), min(_W, _dtm_final.shape[1])

                _tmp[:_h_c, :_w_c] = _dtm_final[:_h_c, :_w_c]

                _dtm_final = _tmp

            _method_str = "Interpolated from LiDAR (low-res, upsampled)"


        _profile.update(dtype="float32", count=1, compress="lzw")

        dtm_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dtm_path, "w", **_profile) as _dst:

            _dst.write(_dtm_final, 1)


        return f"{_method_str}. Range: {_dtm_final.min():.2f} to {_dtm_final.max():.2f} m"


    def generate_ndsm(

        dsm_path: Path,

        dtm_path: Path,

        ndsm_path: Path,

        clip_min: float = 0.0,

        clip_max: float = 30.0,

    ) -> str:

        """Compute nDSM = clip(DSM - DTM, clip_min, clip_max), matched to

        dsm_path's grid, and write it to ndsm_path."""

        with rasterio.open(dsm_path) as _s:

            _dsm = _s.read(1).astype(np.float32)

            _profile = _s.profile

        with rasterio.open(dtm_path) as _s:

            _dtm = _s.read(1, out_shape=_dsm.shape, resampling=Resampling.bilinear).astype(np.float32)


        _ndsm = np.clip(_dsm - _dtm, clip_min, clip_max)

        _profile.update(dtype="float32", count=1)

        ndsm_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(ndsm_path, "w", **_profile) as _dst:

            _dst.write(_ndsm, 1)


        return f"nDSM range: {_ndsm.min():.2f} to {_ndsm.max():.2f} m"

    return generate_dtm, generate_ndsm


@app.cell
def _():
    gen_dtm_button = mo.ui.run_button(label="Generate DTM from LiDAR (just run once)")
    gen_dtm_button
    return (gen_dtm_button,)


@app.cell
def _(DSM_PATH, DTM_PATH, LAS_PATH, gen_dtm_button, generate_dtm):
    mo.stop(

        DTM_PATH.exists() and not gen_dtm_button.value,

        mo.md("DTM already exists. Click the button to regenerate.")

    )

    mo.stop(

        not DTM_PATH.exists() and not gen_dtm_button.value,

        mo.md("DTM not found. Click the button to generate DTM from LiDAR.")

    )

    _msg = generate_dtm(LAS_PATH, DSM_PATH, DTM_PATH)

    mo.md(f"Generated DTM has been saved. {_msg}")
    return


@app.cell
def _():
    # mo.stop(
    #     DTM_PATH.exists() and not gen_dtm_button.value,
    #     mo.md("DTM already exists. Click the button to regenerate.")
    # )
    # mo.stop(
    #     not DTM_PATH.exists() and not gen_dtm_button.value,
    #     mo.md("DTM not found. Click the button to generate DTM from LiDAR.")
    # )

    # _las = laspy.read(LAS_PATH)
    # _gx = _las.x[_las.classification == 2]
    # _gy = _las.y[_las.classification == 2]
    # _gz = _las.z[_las.classification == 2]

    # if len(_gx) == 0:
    #     _z_thresh = np.percentile(_las.z, 10)
    #     _mask2    = _las.z <= _z_thresh
    #     _gx, _gy, _gz = _las.x[_mask2], _las.y[_mask2], _las.z[_mask2]

    # with rasterio.open(DSM_PATH) as _s:
    #     # Only the shape/bounds/profile are needed here — reading the full band
    #     # (the original `_s.read(1)`) cost an extra full-resolution array load
    #     # for no reason, since it was never used below.
    #     _H, _W         = _s.shape
    #     _ref_transform = _s.transform
    #     _ref_crs       = _s.crs
    #     _left, _bottom, _right, _top = _s.bounds
    #     _profile       = _s.profile

    # if len(_gx) > 200_000:
    #     _idx = np.random.choice(len(_gx), 200_000, replace=False)
    #     _gx, _gy, _gz = _gx[_idx], _gy[_idx], _gz[_idx]

    # # --- Interpolate at reduced resolution, then upsample to the full DSM grid ---
    # # The harbour DTM is essentially flat (per the 2c/2d diagnostics), so spending
    # # millions of griddata query points on it buys no real accuracy. Interpolating
    # # at 10% resolution and upsampling cuts the dominant cost by ~100x.
    # _INTERP_SCALE = 0.1
    # _H_lo, _W_lo  = max(int(_H * _INTERP_SCALE), 2), max(int(_W * _INTERP_SCALE), 2)

    # _xs_lo = np.linspace(_left, _right, _W_lo)
    # _ys_lo = np.linspace(_top, _bottom, _H_lo)
    # _gx_lo, _gy_lo = np.meshgrid(_xs_lo, _ys_lo)

    # _dtm_lo = griddata(
    #     np.column_stack([_gx, _gy]), _gz,
    #     (_gx_lo, _gy_lo), method="linear", fill_value=np.nan
    # )

    # _nan_mask_lo = np.isnan(_dtm_lo)
    # if _nan_mask_lo.any():
    #     # Distance-transform nearest-fill instead of a second full griddata call —
    #     # one fast pass instead of a second Qhull-based interpolation.
    #     _fill_idx = scipy.ndimage.distance_transform_edt(
    #         _nan_mask_lo, return_distances=False, return_indices=True
    #     )
    #     _dtm_filled_lo = _dtm_lo[tuple(_fill_idx)]
    # else:
    #     _dtm_filled_lo = _dtm_lo

    # _dtm_std = float(np.std(_dtm_filled_lo))
    # if _dtm_std > 1.0:
    #     _ground_elev = 40.70
    #     _dtm_final   = np.full((_H, _W), fill_value=_ground_elev, dtype=np.float32)
    #     _method_str  = "Flat DTM at 40.70m (interpolation noisy)"
    # else:
    #     _zoom_factors = (_H / _dtm_filled_lo.shape[0], _W / _dtm_filled_lo.shape[1])
    #     _dtm_final = scipy.ndimage.zoom(_dtm_filled_lo, _zoom_factors, order=1).astype(np.float32)
    #     # zoom's output shape can be off by a pixel due to rounding — pad/crop to
    #     # exactly match the DSM grid, since rasterio requires an exact shape match.
    #     if _dtm_final.shape != (_H, _W):
    #         _tmp = np.full((_H, _W), float(_dtm_final.mean()), dtype=np.float32)
    #         _h_c, _w_c = min(_H, _dtm_final.shape[0]), min(_W, _dtm_final.shape[1])
    #         _tmp[:_h_c, :_w_c] = _dtm_final[:_h_c, :_w_c]
    #         _dtm_final = _tmp
    #     _method_str = "Interpolated from LiDAR (low-res, upsampled)"

    # _profile.update(dtype="float32", count=1, compress="lzw")
    # DTM_PATH.parent.mkdir(parents=True, exist_ok=True)
    # with rasterio.open(DTM_PATH, "w", **_profile) as _dst:
    #     _dst.write(_dtm_final, 1)

    # mo.md(f"Generated DTM has been saved. Method: {_method_str}. Range: {_dtm_final.min():.2f} to {_dtm_final.max():.2f} m")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2c · DSM Value Diagnostics
    """)
    return


@app.cell
def _(DSM_PATH):
    with rasterio.open(DSM_PATH) as _s:
        _dsm_d  = _s.read(1)
        _nodata = _s.nodata
    _valid = _dsm_d[_dsm_d != _nodata] if _nodata is not None else _dsm_d[_dsm_d > -9999]
    mo.md(
        f"DSM min: {_dsm_d.min():.2f} | max: {_dsm_d.max():.2f} | "
        f"valid min: {_valid.min():.2f} | 5th pct: {np.percentile(_valid,5):.2f} | "
        f"median: {np.percentile(_valid,50):.2f}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2d · DSM / DTM / nDSM Preview
    """)
    return


@app.cell
def _(DSM_PATH, DTM_PATH):
    with rasterio.open(DTM_PATH) as _s:
        _dtm_check = _s.read(1)
    with rasterio.open(DSM_PATH) as _s:
        _dsm_check = _s.read(1)
    _ndsm_check = np.clip(_dsm_check - _dtm_check, 0, 15)

    _fig, _ax3 = plt.subplots(1, 3, figsize=(18, 5))
    _ax3[0].imshow(_dsm_check, cmap="terrain"); _ax3[0].set_title("DSM"); _ax3[0].axis("off")

    _ax3[1].imshow(_dtm_check, cmap="terrain", vmin=40, vmax=50)
    _ax3[1].set_title("DTM (generated flat at ~40.7m)"); _ax3[1].axis("off")

    _im = _ax3[2].imshow(_ndsm_check, cmap="hot", vmin=0, vmax=5)
    _ax3[2].set_title("nDSM = DSM - DTM (0-5m scale)"); _ax3[2].axis("off")
    plt.colorbar(_im, ax=_ax3[2], fraction=0.046, label="Height (m)")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(DTM_PATH):
    def _():
        import rasterio
        import numpy as np

        with rasterio.open(DTM_PATH) as src:
            dtm = src.read(1)

        print(f"DTM std: {np.std(dtm):.2f}")
        print(f"DTM min: {dtm.min():.2f}")
        return print(f"DTM max: {dtm.max():.2f}")


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2e · Export nDSM for eCognition
    """)
    return


@app.cell
def _():
    # with rasterio.open(DSM_PATH) as _s:
    #     _dsm_e  = _s.read(1).astype(np.float32)
    #     _prof_e = _s.profile
    # with rasterio.open(DTM_PATH) as _s:
    #     _dtm_e  = _s.read(1).astype(np.float32)
    # _ndsm_raw = np.clip(_dsm_e - _dtm_e, 0, 30)
    # _prof_e.update(dtype="float32", count=1)
    # with rasterio.open(nDSM_PATH, "w", **_prof_e) as _dst:
    #     _dst.write(_ndsm_raw, 1)
    # mo.md(f"nDSM exported to {nDSM_PATH}. Range: {_ndsm_raw.min():.2f} to {_ndsm_raw.max():.2f} m")
    return


@app.cell
def _(DSM_PATH, DTM_PATH, generate_ndsm, nDSM_PATH):
    _msg = generate_ndsm(DSM_PATH, DTM_PATH, nDSM_PATH)

    mo.md(f"nDSM exported to {nDSM_PATH}. {_msg}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 · Load Rasters and Compute nDSM

    Open ortophoto RGB, read the bands and get the transform, CRS, and pixel dimensions from it

    Open the DSM and DTM separately and reads each one resampled to that same (H, W) shape using bilinear interpolation. This matters because the DSM/DTM files, to ensure same resolution as orthophoto

    dsm - dtm gives height above ground (the nDSM) and is clipped to a 0–30m range

    The height values then get rescaled from meters (0–30) into the 0–255 uint8 range, since the RGB bands are already uint8 and you can't stack mismatched dtypes into one array cleanly

    Stack R, G, B and the normalised DSM into a single 4-band array.

    The DSM was acquired without mobile objects (containers, cars, boats).
    Ships therefore appear as holes (near-zero height) in the nDSM.
    """)
    return


@app.cell
def _(DSM_PATH, DTM_PATH, RGB_PATH):
    with rasterio.open(RGB_PATH) as _s:
        rgb       = _s.read()
        transform = _s.transform
        crs       = _s.crs
        H, W      = rgb.shape[1], rgb.shape[2]
    with rasterio.open(DSM_PATH) as _s:
        dsm = _s.read(1, out_shape=(H, W), resampling=Resampling.bilinear).astype(np.float32)
    with rasterio.open(DTM_PATH) as _s:
        dtm = _s.read(1, out_shape=(H, W), resampling=Resampling.bilinear).astype(np.float32)
    ndsm_raw    = np.clip(dsm - dtm, 0, 30)
    ndsm_scaled = ((ndsm_raw / 30.0) * 255).astype(np.uint8)
    image_data  = np.vstack([rgb, ndsm_scaled[np.newaxis]])
    mo.md(f"Stack shape: {image_data.shape} (bands, H, W) | CRS: {crs}")
    return image_data, transform


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 4 · Quick Visual Preview
    """)
    return


@app.cell
def _(image_data):
    _band_labels = ["Red", "Green", "Blue", "nDSM (scaled 0-255)"]
    _cmaps       = ["Reds", "Greens", "Blues", "gray"]
    _fig4, _ax4  = plt.subplots(1, 4, figsize=(18, 4))
    for _i, _ax in enumerate(_ax4):
        _ax.imshow(image_data[_i], cmap=_cmaps[_i])
        _ax.set_title(_band_labels[_i])
        _ax.axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### True-colour RGB
    """)
    return


@app.cell
def _(image_data):
    _fig_rgb, _ax_rgb = plt.subplots(figsize=(10, 10))
    _ax_rgb.imshow(image_data[:3].transpose(1, 2, 0))
    _ax_rgb.set_title("RGB - Tile 315135_56865")
    _ax_rgb.axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Downsample for Fast Prototyping

    Scale factor 0.25 runs ~16x faster. Set to 1.0 for the final run.
    """)
    return


@app.cell
def _():
    scale_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.25,
        show_value=True, label="Scale factor", debounce=True
    )
    scale_slider
    return (scale_slider,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load eCognition Segments

    Segments created in eCognition using Multiresolution Segmentation
    (scale ~300, shape 0.2, compactness 0.5, nDSM weight 3).
    Exported as Shapefile.
    """)
    return


@app.cell
def _(SEGMENTS_PATH):
    segments_raw = gpd.read_file(SEGMENTS_PATH)
    mo.md(f"Loaded {len(segments_raw)} segments. CRS: {segments_raw.crs}")
    return (segments_raw,)


@app.cell
def _(segments_raw):
    mo.md(f"""
    Empty geometries: {segments_raw.geometry.is_empty.sum()} | "
        f"Valid: {segments_raw.geometry.is_valid.sum()} | "
        f"Types: {segments_raw.geometry.geom_type.value_counts().to_dict()} | "
        f"Bounds: {segments_raw.total_bounds.tolist()}
    """)
    return


@app.cell
def _(RGB_PATH, segments_raw):
    _fig_seg, _ax_seg = plt.subplots(figsize=(10, 10))
    with rasterio.open(RGB_PATH) as _s:
        from rasterio.plot import show as _rshow
        _rshow(_s, ax=_ax_seg)
    segments_raw.boundary.plot(ax=_ax_seg, color="blue", linewidth=0.3)
    _ax_seg.set_title(f"eCognition segments ({len(segments_raw)} polygons)")
    _ax_seg.axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Feature Extraction via Zonal Statistics

    Build the data RF will be trained on

    The four zonal_stats calls compute the mean and standard deviation of pixel values falling inside each segment polygon, one call per source: red, green, blue bands of the orthophoto, and the nDSM. This gives us the spectral and height signal each segment carries.

    The mean gives the average color/height

    std gives texture/variability within the segment (a uniform water patch should have low std; a cluttered deck or dock edge should have higher std).

    Water absorbs more red and reflects more green than most other surfaces, so this G–R difference is a genuine water signature, we calculate NDWI = (G - R) / (G + R + eps) is normally computed from green and near-infrared but there is no NIR band available (just RGB + height), so this is a green/red proxy standing in for it under the same name.
    """)
    return


@app.cell
def _(RGB_PATH, nDSM_PATH, segments_raw):
    _stats = ["mean", "std"]
    _r = pd.DataFrame(zonal_stats(segments_raw, str(RGB_PATH), band=1, stats=_stats, prefix="R_"))
    _g = pd.DataFrame(zonal_stats(segments_raw, str(RGB_PATH), band=2, stats=_stats, prefix="G_"))
    _b = pd.DataFrame(zonal_stats(segments_raw, str(RGB_PATH), band=3, stats=_stats, prefix="B_"))
    _h = pd.DataFrame(zonal_stats(segments_raw, str(nDSM_PATH), band=1, stats=_stats, prefix="H_"))
    segments = pd.concat([segments_raw, _r, _g, _b, _h], axis=1)
    segments = segments.rename(columns={
        "R_mean": "band_1_mean", "G_mean": "band_2_mean",
        "B_mean": "band_3_mean", "H_mean": "band_4_mean",
        "R_std":  "band_1_std",  "G_std":  "band_2_std",
        "B_std":  "band_3_std",  "H_std":  "band_4_std",
    })
    for _col in ["NDWI", "area", "perimeter", "compactness", "elongation"]:
        if _col in segments.columns:
            segments = segments.drop(columns=[_col])
    segments["NDWI"] = (
        (segments["band_2_mean"] - segments["band_1_mean"]) /
        (segments["band_2_mean"] + segments["band_1_mean"] + 1e-6)
    )
    def get_elongation(geom):
        try:
            _mrr = geom.minimum_rotated_rectangle
            _c   = list(_mrr.exterior.coords)
            _e0  = np.hypot(_c[1][0]-_c[0][0], _c[1][1]-_c[0][1])
            _e1  = np.hypot(_c[2][0]-_c[1][0], _c[2][1]-_c[1][1])
            return max(_e0, _e1) / (min(_e0, _e1) + 1e-6)
        except:
            return np.nan
    segments["area"]        = segments.geometry.area
    segments["perimeter"]   = segments.geometry.length
    segments["compactness"] = (4 * np.pi * segments["area"]) / (segments["perimeter"] ** 2)
    segments["elongation"]  = segments.geometry.apply(get_elongation)
    mo.md(f"Segments: {len(segments)} | Columns: {segments.columns.tolist()}")
    return get_elongation, segments


@app.cell
def _(segments):
    mo.md(str(segments[["NDWI","area","compactness","elongation"]].describe().round(2)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8 · Load Training Samples

    Digitise points in QGIS and save as a GeoPackage.
    Each point must have a field `class` with one of: SHIP, WATER, DOCK.
    Aim for at least 15-20 points per class.
    """)
    return


@app.cell
def _(SAMPLES_PATH):
    sample_points = gpd.read_file(SAMPLES_PATH)
    _counts = sample_points["class"].value_counts()
    _s = " | ".join([f"{k}: {v}" for k, v in _counts.items()])
    mo.md(f"Sample counts: {_s}")
    return (sample_points,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 9 · NickySpatial Layer and Training Spatial Join
    """)
    return


@app.cell
def _(segments):
    seg_layer = ns.Layer(name="ECognition_Segments", layer_type="segmentation")
    seg_layer.objects = segments.copy()
    manager = ns.LayerManager()
    manager.add_layer(seg_layer)
    mo.md(f"NickySpatial layer ready. {len(seg_layer.objects)} segments.")
    return (seg_layer,)


@app.cell
def _():
    selected_features = [
        "band_1_mean", "band_2_mean", "band_3_mean", "band_4_mean",
        "band_1_std",  "band_2_std",  "band_3_std",  "band_4_std",
        "NDWI", "area", "perimeter", "compactness", "elongation"
    ]
    mo.md(f"{len(selected_features)} features: {selected_features}")
    return (selected_features,)


@app.cell
def _(sample_points, seg_layer):
    object_samples = gpd.sjoin(
        seg_layer.objects,
        sample_points[["class", "geometry"]],
        how="inner",
        predicate="intersects"
    )
    _c = object_samples["class"].value_counts()
    _s = " | ".join([f"{k}: {v}" for k, v in _c.items()])
    mo.md(f"Matched {len(object_samples)} segments. Classes: {_s}")
    return (object_samples,)


@app.cell
def _(object_samples):
    mo.md(f"""
    object_samples columns: {object_samples.columns.tolist()}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 9b · Visualise Training Sample Distribution
    """)
    return


@app.cell
def _(image_data, scale_slider, transform):
    _sf      = scale_slider.value

    ds_image = scipy.ndimage.zoom(image_data, zoom=(1, _sf, _sf), order=0) if _sf < 1.0 else image_data

    ds_transform = transform * Affine.scale(1 / _sf)

    mo.md(f"Working resolution: {ds_image.shape}")
    return ds_image, ds_transform


@app.cell
def _(ds_image, ds_transform, object_samples):
    CLASS_COLORS = {
        "SHIP":  "#e63946",
        "BOAT":  "#f4a261",
        "WATER": "#457b9d",
        "DOCK":  "#a8dadc",
    }
    _sample_lyr = ns.Layer(name="Samples", layer_type="classification")
    _sample_lyr.objects = object_samples
    ns.plot_sample(
        _sample_lyr,
        image_data=ds_image,
        rgb_bands=(0, 1, 2),
        transform=ds_transform,
        class_field="class",
        class_color=CLASS_COLORS,
    )
    return (CLASS_COLORS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10 · Random Forest Classification

    class_weight balanced compensates for having many more WATER segments than SHIP.
    OOB score gives an honest accuracy estimate without a separate test set.
    """)
    return


@app.cell
def _():
    n_trees_slider = mo.ui.slider(
        start=50, stop=500, step=50, value=200,
        show_value=True, label="Number of trees", debounce=True
    )
    n_trees_slider
    return (n_trees_slider,)


@app.cell
def _(n_trees_slider, object_samples, selected_features):
    X_train = object_samples[selected_features].fillna(0)
    y_train = object_samples["class"]
    clf = RandomForestClassifier(
        n_estimators=n_trees_slider.value,
        oob_score=True,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    _oob_pred = clf.classes_[clf.oob_decision_function_.argmax(axis=1)]
    _report   = classification_report(y_train, _oob_pred, zero_division=0)
    mo.md(f"OOB accuracy: {clf.oob_score_:.3f}\n\n```\n{_report}\n```")
    return (clf,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Feature Importance
    """)
    return


@app.cell
def _(clf, selected_features):
    _imp = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
    _fig_imp, _ax_imp = plt.subplots(figsize=(9, 3))
    _imp.plot.bar(ax=_ax_imp, color="#e63946", edgecolor="white")
    _ax_imp.set_title(f"Feature Importances (OOB: {clf.oob_score_:.3f})")
    _ax_imp.set_ylabel("Importance")
    _ax_imp.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 11 · Classify All Segments
    """)
    return


@app.cell
def _(clf, seg_layer, selected_features):
    seg_layer.objects["classification"] = clf.predict(
        seg_layer.objects[selected_features].fillna(0)
    )
    classified_layer = ns.Layer(name="RF_Classification", layer_type="classification")
    classified_layer.objects = seg_layer.objects.copy()
    _c = classified_layer.objects["classification"].value_counts()
    _s = " | ".join([f"{k}: {v}" for k, v in _c.items()])
    mo.md(f"Classification complete: {_s}")
    return (classified_layer,)


@app.cell
def _(CLASS_COLORS, classified_layer):
    ns.plot_classification(
        classified_layer,
        class_field="classification",
        class_color=CLASS_COLORS,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Further Spatial Classification: SHIP_WATER vs SHIP_DOCK

    SHIP segments are sub-classified using spatial adjacency:
    - SHIP_WATER: vessel touching WATER (floating freely in open water)
    - SHIP_DOCK: vessel touching DOCK (berthed at quayside)
    """)
    return


@app.cell
def _(classified_layer):
    gdf = classified_layer.objects.copy().reset_index(drop=True)

    def subclassify_ships(idx, _gdf, _tree):
        _row = _gdf.iloc[idx]
        if _row["classification"] != "SHIP":
            return _row["classification"]
        _candidates       = _tree.query(_row.geometry, predicate="touches")
        _neighbours       = _gdf.iloc[_candidates]
        _neighbour_classes = _neighbours["classification"].values
        if "WATER" in _neighbour_classes:
            return "SHIP_WATER"
        elif "DOCK" in _neighbour_classes:
            return "SHIP_DOCK"
        else:
            return "SHIP"

    _tree = STRtree(gdf.geometry)
    gdf["classification"] = [subclassify_ships(i, gdf, _tree) for i in range(len(gdf))]
    _c = gdf["classification"].value_counts()
    _s = " | ".join([f"{k}: {v}" for k, v in _c.items()])
    mo.md(f"Sub-classification complete: {_s}")
    return (gdf,)


@app.cell
def _(gdf):
    EXTENDED_COLORS = {
        "SHIP_WATER": "#e63946",
        "SHIP_DOCK":  "#f4a261",
        "SHIP":       "#ffb3b3",
        "WATER":      "#457b9d",
        "DOCK":       "#a8dadc",
    }
    _refined = ns.Layer(name="Ship_Subclassified", layer_type="classification")
    _refined.objects = gdf.copy()
    ns.plot_classification(_refined, class_field="classification", class_color=EXTENDED_COLORS)
    return (EXTENDED_COLORS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Commented alternative using NickySpatial EnclosedByRuleSet

    This approach was also tried but STRtree adjacency worked better for this dataset.
    Kept here for reference.
    """)
    return


@app.cell
def _():
    # Alternative NickySpatial approach (commented out - STRtree used instead)
    # classified_layer.objects = classified_layer.objects.reset_index(drop=True)
    # classified_layer.objects["segment_id"] = classified_layer.objects.index
    # enclosed = ns.EnclosedByRuleSet()
    # ship_water = enclosed.execute(source_layer=classified_layer,
    #     class_column_name="classification", class_value_a="SHIP", class_value_b="WATER",
    #     new_class_name="SHIP_WATER", layer_manager=manager, layer_name="ship_water")
    # ship_dock = enclosed.execute(source_layer=ship_water,
    #     class_column_name="classification", class_value_a="SHIP", class_value_b="DOCK",
    #     new_class_name="SHIP_DOCK", layer_manager=manager, layer_name="ship_dock")
    mo.md("""
    Alternative EnclosedByRuleSet approach removed as it did not give me good enough results.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 13 · Extract Vessel Attributes

    | Attribute | Method |
    |---|---|
    | length_m | Long axis of minimum rotated bounding box |
    | width_m | Short axis of minimum rotated bounding box |
    | orientation | Angle of long axis from east (degrees) |
    | aspect_ratio | length / width |
    | ndsm_mean | Mean nDSM value (~3m on water, ~8m at dock) |
    | size_class | Large >50m / Medium 20-50m / Small <20m |
    """)
    return


@app.cell
def _(gdf):
    def oriented_bbox_attrs(geom):
        try:
            _mrr = geom.minimum_rotated_rectangle
            _c   = list(_mrr.exterior.coords)
            _dx0 = _c[1][0]-_c[0][0]; _dy0 = _c[1][1]-_c[0][1]
            _dx1 = _c[2][0]-_c[1][0]; _dy1 = _c[2][1]-_c[1][1]
            _e0, _e1 = np.hypot(_dx0, _dy0), np.hypot(_dx1, _dy1)
            _length, _width = max(_e0, _e1), min(_e0, _e1)
            _angle = np.degrees(
                np.arctan2(_dy0, _dx0) if _e0 >= _e1 else np.arctan2(_dy1, _dx1)
            ) % 180
            return _length, _width, _angle
        except:
            return np.nan, np.nan, np.nan

    _vessel_cls = ["SHIP_WATER", "SHIP_DOCK", "SHIP"]
    vessels = gdf[gdf["classification"].isin(_vessel_cls)].copy()
    _attrs  = vessels.geometry.apply(oriented_bbox_attrs)
    vessels[["length_m", "width_m", "orientation"]] = pd.DataFrame(_attrs.tolist(), index=vessels.index)
    vessels["aspect_ratio"] = vessels["length_m"] / vessels["width_m"].replace(0, np.nan)
    vessels["ndsm_mean"]    = vessels["band_4_mean"]

    def size_class(l):
        if l > 50:   return "Large vessel"
        elif l > 20: return "Medium vessel"
        else:        return "Small boat"

    vessels["size_class"] = vessels["length_m"].apply(size_class)
    _dcols = ["classification", "size_class", "length_m", "width_m", "aspect_ratio", "orientation", "ndsm_mean"]
    mo.md(f"{len(vessels)} vessel segments. Preview (first 10):\n\n" + vessels[_dcols].round(2).head(10).to_html())
    return oriented_bbox_attrs, vessels


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Vessel Count by Class
    """)
    return


@app.cell
def _(EXTENDED_COLORS, gdf):
    _cls = ["SHIP_WATER", "SHIP_DOCK", "SHIP"]
    _c   = gdf["classification"].value_counts()
    _vc  = _c[_c.index.isin(_cls)]
    _fig_c, _ax_c = plt.subplots(figsize=(6, 4))
    _vc.plot.bar(ax=_ax_c, color=[EXTENDED_COLORS.get(c, "gray") for c in _vc.index], edgecolor="white")
    _ax_c.set_title("Vessel Count by Class")
    _ax_c.set_ylabel("Number of segments")
    _ax_c.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Size Classification Map
    """)
    return


@app.cell
def _(gdf, vessels):
    gdf["size_class"] = gdf.index.map(vessels["size_class"])
    gdf["size_class"] = gdf["size_class"].fillna("")

    def combined_class(row):
        if row["classification"] in ["SHIP_WATER", "SHIP_DOCK"] and row["size_class"] != "":
            return row["classification"] + "_" + row["size_class"].replace(" ", "_")
        return row["classification"]

    gdf["combined_class"] = gdf.apply(combined_class, axis=1)
    COMBINED_COLORS = {
        "SHIP_WATER_Large_vessel":  "#F58027",
        "SHIP_WATER_Medium_vessel": "#E4F527",
        "SHIP_WATER_Small_boat":    "#BE27F5",
        "SHIP_DOCK_Large_vessel":   "#b35000",
        "SHIP_DOCK_Medium_vessel":  "#f4a261",
        "SHIP_DOCK_Small_boat":     "#27F52E",
        "SHIP":  "#F52749",
        "WATER": "#457b9d",
        "DOCK":  "#a8dadc",
    }
    _c = gdf["combined_class"].value_counts()
    _s = " | ".join([f"{k}: {v}" for k, v in _c.items()])
    mo.md(f"Size class counts: {_s}")
    return (COMBINED_COLORS,)


@app.cell
def _(COMBINED_COLORS, gdf):
    _size_lyr = ns.Layer(name="Ship_SizeClass", layer_type="classification")
    _size_lyr.objects = gdf.copy()
    _size_lyr.objects["classification"] = gdf["combined_class"]
    ns.plot_classification(_size_lyr, class_field="classification", class_color=COMBINED_COLORS)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 14 · Attribute Distributions
    """)
    return


@app.cell
def _(vessels):
    _fig_d, _ax_d = plt.subplots(1, 3, figsize=(15, 4))
    _ax_d[0].hist(vessels["length_m"].dropna(), bins=20, color="#e63946", edgecolor="white")
    _ax_d[0].set_xlabel("Length (m)"); _ax_d[0].set_title("Vessel Length Distribution")
    _ax_d[1].hist(vessels["aspect_ratio"].dropna(), bins=20, color="#f4a261", edgecolor="white")
    _ax_d[1].set_xlabel("Aspect Ratio"); _ax_d[1].set_title("Aspect Ratio Distribution")
    _ax_d[2].hist(vessels["ndsm_mean"].dropna(), bins=20, color="#457b9d", edgecolor="white")
    _ax_d[2].set_xlabel("nDSM mean (m)"); _ax_d[2].set_title("nDSM Mean")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(vessels):
    _fig_sc, _ax_sc = plt.subplots(figsize=(8, 5))
    for _cls, _grp in vessels.groupby("classification"):
        _col = {"SHIP_WATER": "#e63946", "SHIP_DOCK": "#f4a261", "SHIP": "#ffb3b3"}.get(_cls, "gray")
        _ax_sc.scatter(_grp["length_m"], _grp["aspect_ratio"], label=_cls,
                       color=_col, edgecolors="white", s=60, alpha=0.8)
    _ax_sc.set_xlabel("Length (m)"); _ax_sc.set_ylabel("Aspect Ratio")
    _ax_sc.set_title("Length vs Aspect Ratio by Vessel Class")
    _ax_sc.legend()
    plt.tight_layout()
    plt.gca()
    return


app._unparsable_cell(
    r"""
    _sw = vessels[vessels["classification"] == "SHIP_WATER"]["ndsm_mean"].dropna()
    _sd = vessels[vessels["classification"] == "SHIP_DOCK"]["ndsm_mean"].dropna()

    _colors = vessels["classification"].map({"SHIP_WATER": "#e63946", "SHIP_DOCK": "#f4a261", "SHIP": "#ffb3b3"})
    _zoomed = vessels[(vessels["length_m"] <= 30) & (vessels["aspect_ratio"] <= 10)]
    _cz     = _zoomed["classification"].map({"SHIP_WATER": "#e63946", "SHIP_DOCK": "#f4a261", "SHIP": "#ffb3b3"})

    _fig_h, _ax_h = plt.subplots(1, 3, figsize=(15, 4))
    _vc = vessels["classification"].value_counts()

    _ax_h[0].bar(_vc.index, _vc.values,
                 color=["#e63946" if "WATER" in c else "#f4a261" if "DOCK" in c else "#ffb3b3" for c in _vc.index])
    _ax_h[0].set_title("Vessel Count by Class"); _ax_h[0].set_ylabel("Count")


    _ax_h[1].scatter(_zoomed["length_m"], _zoomed["aspect_ratio"], c=_cz, edgecolors="white", s=80, alpha=0.9)
    _ax_h[1].set_xlabel("Boat Length (m)"); _ax_h[1].set_ylabel("Aspect Ratio")
    _ax_h[1].set_title(f"Zoomed: 0-30m, AR<=10 ({len(_zoomed)}/{len(vessels)} vessels)")
    _ax_h[1].set_xlim(0, 30); _ax_h[1].set_ylim(0, 10)
    _ax_h[1].legend(handles=[
        mpatches.Patch(facecolor="#e63946", label="SHIP_WATER"),
        mpatches.Patch(facecolor="#f4a261", label="SHIP_DOCK"),
        mpatches.Patch(facecolor="#ffb3b3", label="SHIP"),
    ])


    _ax_h[2].hist(_sw, bins=8, color="#e63946", alpha=0.8, label="SHIP_WATER", edgecolor="white")
    _ax_h[2].hist(_sd, bins=8, color="#f4a261", alpha=0.8, label="SHIP_DOCK",  edgecolor="white")
    _ax_h[2].set_xlabel("nDSM mean (m)"); _ax_h[2].set_ylabel("Count")
    _ax_h[2].set_title("Height Distribution by Class (~3m=water, ~8m=dock)")
    _ax_h[2].set_xlim(0, 22); _ax_h[2].legend()j


    plt.tight_layout()
    plt.gca()
    """,
    name="_"
)


@app.cell
def _(oriented_bbox_attrs, vessels):
    attrs = vessels.geometry.apply(oriented_bbox_attrs)
    vessels[["length_m", "width_m", "orientation"]] = pd.DataFrame(
        attrs.tolist(), index=vessels.index
    )
    vessels["aspect_ratio"] = vessels["length_m"] / vessels["width_m"].replace(0, np.nan)
    vessels["ndsm_mean"]    = vessels["band_4_mean"]

    def get_size_class(l):
        if l > 50:   return "Large vessel"
        elif l > 20: return "Medium vessel"
        else:        return "Small boat"

    vessels["size_class"] = vessels["length_m"].apply(get_size_class)

    display_cols = ["classification", "size_class", "length_m",
                    "width_m", "aspect_ratio", "orientation", "ndsm_mean"]

    mo.ui.table(vessels[display_cols].round(2).reset_index(drop=True))
    return


@app.cell
def _(vessels):
    _sw_s = vessels[vessels["classification"] == "SHIP_WATER"]["ndsm_mean"].describe().round(2)

    _sd_s = vessels[vessels["classification"] == "SHIP_DOCK"]["ndsm_mean"].describe().round(2)

    mo.md(f"SHIP_WATER nDSM: {_sw_s.to_dict()}\n\nSHIP_DOCK nDSM: {_sd_s.to_dict()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive Vessel Explorer

    Adjust filters and the map updates reactively.
    """)
    return


@app.cell
def _():
    min_len = mo.ui.slider(start=0, stop=100, step=5, value=0,   show_value=True, label="Min length (m)", debounce=True)

    max_len = mo.ui.slider(start=0, stop=100, step=5, value=100, show_value=True, label="Max length (m)", debounce=True)

    cls_flt = mo.ui.multiselect(options=["SHIP_WATER", "SHIP_DOCK", "SHIP"],
                                 value=["SHIP_WATER", "SHIP_DOCK", "SHIP"], label="Vessel class")

    sz_flt  = mo.ui.multiselect(options=["Small boat", "Medium vessel", "Large vessel"],
                                 value=["Small boat", "Medium vessel", "Large vessel"], label="Size class")
    mo.vstack([mo.hstack([min_len, max_len]), mo.hstack([cls_flt, sz_flt])])
    return cls_flt, max_len, min_len, sz_flt


@app.cell
def _(cls_flt, max_len, min_len, sz_flt, vessels):
    filtered = vessels[
        (vessels["length_m"] >= min_len.value) &
        (vessels["length_m"] <= max_len.value) &
        (vessels["classification"].isin(cls_flt.value)) &
        (vessels["size_class"].isin(sz_flt.value))
    ]
    mo.md(f"{len(filtered)} vessel(s) match current filters.")
    return (filtered,)


@app.cell
def _(RGB_PATH, filtered):
    _VCOL = {"SHIP_WATER": "#e63946", "SHIP_DOCK": "#f4a261", "SHIP": "#ffb3b3"}
    _fig_m, _ax_m = plt.subplots(figsize=(10, 10))

    with rasterio.open(RGB_PATH) as _s:
        from rasterio.plot import show as _rs
        _rs(_s, ax=_ax_m)

    for _cls2, _col2 in _VCOL.items():
        _sub = filtered[filtered["classification"] == _cls2]
        if len(_sub) > 0:
            _sub.plot(ax=_ax_m, color=_col2, alpha=0.7, edgecolor="white", linewidth=0.5, label=_cls2)

    _ax_m.legend(handles=[mpatches.Patch(facecolor=v, label=k) for k, v in _VCOL.items()], loc="upper right")

    _ax_m.set_title(f"{len(filtered)} vessel(s) in a  filtered view"); _ax_m.axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(filtered):
    _dcols2 = ["classification", "size_class", "length_m", "width_m", "aspect_ratio", "orientation", "ndsm_mean"]
    mo.ui.table(filtered[_dcols2].round(2).reset_index(drop=True))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 16 · Export to GeoPackage

    Exports confirmed vessel polygons with all attributes.
    Open the gpkg in QGIS for cartography and final quality assessment.
    """)
    return


@app.cell
def _():
    btn_export = mo.ui.run_button(label="Export GeoPackage")
    btn_export
    return (btn_export,)


@app.cell
def _(OUTPUT_PATH, btn_export, vessels):
    mo.stop(not btn_export.value, mo.md("Click button to export."))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _export_cols = [
        "classification", "size_class", "length_m", "width_m",
        "aspect_ratio", "orientation", "ndsm_mean",
        "band_1_mean", "band_2_mean", "band_3_mean", "geometry"
    ]
    vessels[_export_cols].to_file(OUTPUT_PATH, driver="GPKG")
    mo.md(f"Exported {len(vessels)} vessels to {OUTPUT_PATH}")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Apply Model to New Tile (315130) without nDSM
    """)
    return


@app.cell
def _(BASE_DIR):
    RGB_PATH_NEW      = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315140_56865_UTM31N.tif"
    SEGMENTS_PATH_NEW      = BASE_DIR / "ecognition_segmentation" / "segmentation_315140" / "300_315140_segmentation.shp"
    return RGB_PATH_NEW, SEGMENTS_PATH_NEW


@app.cell
def _(RGB_PATH_NEW, SEGMENTS_PATH_NEW, get_elongation):
    _segs_n = gpd.read_file(SEGMENTS_PATH_NEW)
    _stats  = ["mean", "std"]
    _r = pd.DataFrame(zonal_stats(_segs_n, str(RGB_PATH_NEW), band=1, stats=_stats, prefix="R_"))
    _g = pd.DataFrame(zonal_stats(_segs_n, str(RGB_PATH_NEW), band=2, stats=_stats, prefix="G_"))
    _b = pd.DataFrame(zonal_stats(_segs_n, str(RGB_PATH_NEW), band=3, stats=_stats, prefix="B_"))
    segments_new = pd.concat([_segs_n, _r, _g, _b], axis=1)
    segments_new = segments_new.rename(columns={
        "R_mean": "band_1_mean", "G_mean": "band_2_mean",
        "B_mean": "band_3_mean",
        "R_std":  "band_1_std",  "G_std":  "band_2_std",
        "B_std":  "band_3_std",
    })
    # Fill nDSM with zeros since we don't have it
    segments_new["band_4_mean"] = 0.0
    segments_new["band_4_std"]  = 0.0
    segments_new["NDWI"]        = (segments_new["band_2_mean"] - segments_new["band_1_mean"]) / (segments_new["band_2_mean"] + segments_new["band_1_mean"] + 1e-6)
    segments_new["area"]        = segments_new.geometry.area
    segments_new["perimeter"]   = segments_new.geometry.length
    segments_new["compactness"] = (4 * np.pi * segments_new["area"]) / (segments_new["perimeter"] ** 2)
    segments_new["elongation"]  = segments_new.geometry.apply(get_elongation)
    mo.md(f"New tile: {len(segments_new)} segments ready.")
    return (segments_new,)


@app.cell
def _(EXTENDED_COLORS, clf, segments_new, selected_features):
    segments_new["classification"] = clf.predict(segments_new[selected_features].fillna(0))

    _gdf_n  = segments_new.copy().reset_index(drop=True)
    _tree_n = STRtree(_gdf_n.geometry)

    def _sub_n(idx, _g, _t):
        _row = _g.iloc[idx]
        if _row["classification"] != "SHIP":
            return _row["classification"]
        _nbs = _g.iloc[_t.query(_row.geometry, predicate="touches")]["classification"].values
        if "WATER" in _nbs:  return "SHIP_WATER"
        elif "DOCK" in _nbs: return "SHIP_DOCK"
        else:                return "SHIP"

    _gdf_n["classification"] = [_sub_n(i, _gdf_n, _tree_n) for i in range(len(_gdf_n))]
    _c = _gdf_n["classification"].value_counts()
    mo.md(" | ".join([f"**{k}**: {v}" for k, v in _c.items()]))

    _lyr_n = ns.Layer(name="Tile315130", layer_type="classification")
    _lyr_n.objects = _gdf_n
    ns.plot_classification(_lyr_n, class_field="classification", class_color=EXTENDED_COLORS)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Apply Model to New Tile (315130) with nDSM and attributes
    """)
    return


@app.cell
def _(BASE_DIR):
    RGB_PATH_315140      = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315140_56865_UTM31N.tif"
    DSM_PATH_315140       = BASE_DIR / "data_store" / "BE_DSM_27032011_315140_56865_UTM31N.tif"
    LAS_PATH_315140       = BASE_DIR / "data_store" / "BE_LIDAR_27032011_315140_56865_UTM31N.las"
    DTM_PATH_315140       = BASE_DIR / "data_store" / "BE_DTM_315140_56865_generated.tif"
    nDSM_PATH_315140      = BASE_DIR / "data_store" / "BE_nDSM_315140_56865.tif"
    SEGMENTS_PATH_315140  = BASE_DIR / "ecognition_segmentation" / "segmentation_315140" / "300_315140_segmentation.shp"
    return (
        DSM_PATH_315140,
        DTM_PATH_315140,
        LAS_PATH_315140,
        RGB_PATH_315140,
        SEGMENTS_PATH_315140,
        nDSM_PATH_315140,
    )


@app.cell
def _():
    gen_dtm_button_new = mo.ui.run_button(label="Generate DTM for tile 315140 (run once)")

    gen_dtm_button_new
    return (gen_dtm_button_new,)


@app.cell
def _(
    DSM_PATH_315140,
    DTM_PATH_315140,
    LAS_PATH_315140,
    gen_dtm_button_new,
    generate_dtm,
):
    mo.stop(
        DTM_PATH_315140.exists() and not gen_dtm_button_new.value,
        mo.md("DTM for tile 315140 already exists. Click the button to regenerate.")
    )
    mo.stop(
        not DTM_PATH_315140.exists() and not gen_dtm_button_new.value,
        mo.md("DTM for tile 315140 not found. Click the button to generate it from LiDAR.")
    )
    _msg = generate_dtm(LAS_PATH_315140, DSM_PATH_315140, DTM_PATH_315140)
    mo.md(f"Generated DTM for tile 315140. {_msg}")
    return


@app.cell
def _(DSM_PATH_315140, DTM_PATH_315140, generate_ndsm, nDSM_PATH_315140):
    _msg = generate_ndsm(DSM_PATH_315140, DTM_PATH_315140, nDSM_PATH_315140)
    mo.md(f"nDSM exported for tile 315140 to {nDSM_PATH_315140}. {_msg}")
    return


@app.cell
def _(RGB_PATH_315140, SEGMENTS_PATH_315140, get_elongation, nDSM_PATH_315140):
    _segs_n = gpd.read_file(SEGMENTS_PATH_315140)
    _stats  = ["mean", "std"]
    _r = pd.DataFrame(zonal_stats(_segs_n, str(RGB_PATH_315140), band=1, stats=_stats, prefix="R_"))
    _g = pd.DataFrame(zonal_stats(_segs_n, str(RGB_PATH_315140), band=2, stats=_stats, prefix="G_"))
    _b = pd.DataFrame(zonal_stats(_segs_n, str(RGB_PATH_315140), band=3, stats=_stats, prefix="B_"))
    _h = pd.DataFrame(zonal_stats(_segs_n, str(nDSM_PATH_315140), band=1, stats=_stats, prefix="H_"))
    segments_315140 = pd.concat([_segs_n, _r, _g, _b, _h], axis=1)
    segments_315140 = segments_315140.rename(columns={
        "R_mean": "band_1_mean", "G_mean": "band_2_mean",
        "B_mean": "band_3_mean", "H_mean": "band_4_mean",
        "R_std":  "band_1_std",  "G_std":  "band_2_std",
        "B_std":  "band_3_std",  "H_std":  "band_4_std",
    })
    segments_315140["NDWI"] = (
        (segments_315140["band_2_mean"] - segments_315140["band_1_mean"]) /
        (segments_315140["band_2_mean"] + segments_315140["band_1_mean"] + 1e-6)
    )
    segments_315140["area"]        = segments_315140.geometry.area
    segments_315140["perimeter"]   = segments_315140.geometry.length
    segments_315140["compactness"] = (4 * np.pi * segments_315140["area"]) / (segments_315140["perimeter"] ** 2)
    segments_315140["elongation"]  = segments_315140.geometry.apply(get_elongation)
    mo.md(f"New tile: {len(segments_315140)} segments ready (real nDSM height feature, not zero-filled).")
    return (segments_315140,)


@app.cell
def _(EXTENDED_COLORS, clf, segments_315140, selected_features):
    segments_315140["classification"] = clf.predict(segments_315140[selected_features].fillna(0))

    _gdf_n  = segments_315140.copy().reset_index(drop=True)
    _tree_n = STRtree(_gdf_n.geometry)

    def _sub_n(idx, _g, _t):
        _row = _g.iloc[idx]
        if _row["classification"] != "SHIP":
            return _row["classification"]
        _nbs = _g.iloc[_t.query(_row.geometry, predicate="touches")]["classification"].values
        if "WATER" in _nbs:  return "SHIP_WATER"
        elif "DOCK" in _nbs: return "SHIP_DOCK"
        else:                return "SHIP"

    _gdf_n["classification"] = [_sub_n(i, _gdf_n, _tree_n) for i in range(len(_gdf_n))]
    _c = _gdf_n["classification"].value_counts()
    mo.md(" | ".join([f"**{k}**: {v}" for k, v in _c.items()]))

    _lyr_n = ns.Layer(name="Tile315140", layer_type="classification")
    _lyr_n.objects = _gdf_n
    ns.plot_classification(_lyr_n, class_field="classification", class_color=EXTENDED_COLORS)
    return


if __name__ == "__main__":
    app.run()
