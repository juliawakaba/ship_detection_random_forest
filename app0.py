import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", layout_file="layouts/app0.grid.json")

with app.setup(hide_code=True):
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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Start settings

    - Selecting method for classification
    - Selecting tile
    - Setting files paths
    """)
    return


@app.cell
def _():
    model_selector = mo.ui.dropdown(
        options=["Geoai", "Random Forest"],
        value="Random Forest",
        label="Select Model",
    )

    model_selector
    return (model_selector,)


@app.cell
def _():
    BASE_DIR = Path(__file__).resolve().parent

    tile_selector = mo.ui.dropdown(
        options=["Tile 35", "Tile 40"],
        value="Tile 35",
        label="Select Tile",
    )

    tile_selector
    return BASE_DIR, tile_selector


@app.cell
def _(BASE_DIR, tile_selector):
    if tile_selector.value == "Tile 35":
        RGB_PATH      = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315135_56865_UTM31N.tif"
        LAS_PATH      = BASE_DIR / "data_store" / "BE_LIDAR_27032011_315135_56865_UTM31N.las"
        DSM_PATH      = BASE_DIR / "data_store" / "BE_DSM_27032011_315135_56865_UTM31N.tif"
        DTM_PATH      = BASE_DIR / "data_store" / "BE_DTM_315135_56865_generated.tif"
        nDSM_PATH     = BASE_DIR / "data_store" / "BE_nDSM_315135_56865.tif"
        SEGMENTS_PATH = BASE_DIR / "ecognition_segmentation" / "segmentation_315135" / "300_315135_segmentation.shp"
        SAMPLES_PATH  = BASE_DIR / "315135_samples" / "315135_sample_points.gpkg"
    else: 
        RGB_PATH      = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315140_56865_UTM31N.tif"
        DSM_PATH       = BASE_DIR / "data_store" / "BE_DSM_27032011_315140_56865_UTM31N.tif"
        LAS_PATH       = BASE_DIR / "data_store" / "BE_LIDAR_27032011_315140_56865_UTM31N.las"
        DTM_PATH       = BASE_DIR / "data_store" / "BE_DTM_315140_56865_generated.tif"
        nDSM_PATH      = BASE_DIR / "data_store" / "BE_nDSM_315140_56865.tif"
        SEGMENTS_PATH  = BASE_DIR / "ecognition_segmentation" / "segmentation_315140" / "300_315140_segmentation.shp"
        SAMPLES_PATH  = BASE_DIR / "315135_samples" / "315135_sample_points.gpkg"

    OUTPUT_PATH   = BASE_DIR / "output" / "ships_detected.gpkg"
    return DSM_PATH, DTM_PATH, LAS_PATH, RGB_PATH, SEGMENTS_PATH, nDSM_PATH


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # OBIA method

    ## Object-Based Image Analysis · RGB + nDSM: Belgium 2011

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
    ## Generate DTM from LiDAR

    Derives the bare-earth DTM from the `.las` point cloud.
    Falls back to 10th percentile if no ground points found.
    If interpolation is noisy (harbour is flat), switches to a flat DTM at 40.70m.
    This step is skipped automatically if file already exists.


    this process takes a few seconds, I just had to give it some time like less than 1 minute to run
    """)
    return


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )
    return


@app.cell(hide_code=True)
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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

    return (generate_dtm,)


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("A method that does not require a DTM was selected")
    )

    gen_dtm_button = mo.ui.run_button(label="Generate DTM from LiDAR (just run once)")
    gen_dtm_button
    return (gen_dtm_button,)


@app.cell
def _(
    DSM_PATH,
    DTM_PATH,
    LAS_PATH,
    gen_dtm_button,
    generate_dtm,
    model_selector,
):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Rasters and Compute nDSM

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
    ## True-colour RGB image preview
    """)
    return


@app.cell
def _(image_data, tile_selector):
    _fig_rgb, _ax_rgb = plt.subplots(figsize=(10, 10))
    _ax_rgb.imshow(image_data[:3].transpose(1, 2, 0))
    _ax_rgb.set_title("RGB - " + tile_selector.value)
    _ax_rgb.axis("off")
    plt.tight_layout()
    plt.gca()
    return


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
def _(SEGMENTS_PATH, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    segments_raw = gpd.read_file(SEGMENTS_PATH)
    mo.md(f"Loaded {len(segments_raw)} segments. CRS: {segments_raw.crs}")
    return (segments_raw,)


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
def _(RGB_PATH, model_selector, nDSM_PATH, segments_raw):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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
    return get_elongation, segments


@app.cell
def _(model_selector, segments):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    seg_layer = ns.Layer(name="ECognition_Segments", layer_type="segmentation")
    seg_layer.objects = segments.copy()
    manager = ns.LayerManager()
    manager.add_layer(seg_layer)
    return (seg_layer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trainning the RF model
    The trainning is done on tile 35
    """)
    return


@app.cell
def _(BASE_DIR, get_elongation, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    _seg_path_135  = BASE_DIR / "ecognition_segmentation" / "segmentation_315135" / "300_315135_segmentation.shp"
    _rgb_path_135  = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315135_56865_UTM31N.tif"
    _ndsm_path_135 = BASE_DIR / "data_store" / "BE_nDSM_315135_56865.tif"
    _segs_135 = gpd.read_file(_seg_path_135)
    _stats = ["mean", "std"]
    _r = pd.DataFrame(zonal_stats(_segs_135, str(_rgb_path_135), band=1, stats=_stats, prefix="R_"))
    _g = pd.DataFrame(zonal_stats(_segs_135, str(_rgb_path_135), band=2, stats=_stats, prefix="G_"))
    _b = pd.DataFrame(zonal_stats(_segs_135, str(_rgb_path_135), band=3, stats=_stats, prefix="B_"))
    _h = pd.DataFrame(zonal_stats(_segs_135, str(_ndsm_path_135), band=1, stats=_stats, prefix="H_"))
    segments_train = pd.concat([_segs_135, _r, _g, _b, _h], axis=1)
    segments_train = segments_train.rename(columns={
        "R_mean": "band_1_mean", "G_mean": "band_2_mean",
        "B_mean": "band_3_mean", "H_mean": "band_4_mean",
        "R_std":  "band_1_std",  "G_std":  "band_2_std",
        "B_std":  "band_3_std",  "H_std":  "band_4_std",
    })
    segments_train["NDWI"] = (
        (segments_train["band_2_mean"] - segments_train["band_1_mean"]) /
        (segments_train["band_2_mean"] + segments_train["band_1_mean"] + 1e-6)
    )
    segments_train["area"]        = segments_train.geometry.area
    segments_train["perimeter"]   = segments_train.geometry.length
    segments_train["compactness"] = (4 * np.pi * segments_train["area"]) / (segments_train["perimeter"] ** 2)
    segments_train["elongation"]  = segments_train.geometry.apply(get_elongation)
    mo.md(f"Training segments (315135): {len(segments_train)} polygons with features extracted.")
    segments_train
    return (segments_train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### NickySpatial Layer and Training Spatial Join for tile 35
    """)
    return


@app.cell
def _(BASE_DIR, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    sample_points = gpd.read_file(
        BASE_DIR / "315135_samples" / "315135_sample_points.gpkg"
    )
    _counts = sample_points["class"].value_counts()
    return (sample_points,)


@app.cell
def _(model_selector, segments_train):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    seg_layer_35 = ns.Layer(name="ECognition_Segments", layer_type="segmentation")
    seg_layer_35.objects = segments_train .copy()
    manager_35 = ns.LayerManager()
    manager_35.add_layer(seg_layer_35)
    return (seg_layer_35,)


@app.cell
def _(model_selector, sample_points, seg_layer_35):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    object_samples = gpd.sjoin(
        seg_layer_35.objects,
        sample_points[["class", "geometry"]],
        how="inner",
        predicate="intersects"
    )
    _c = object_samples["class"].value_counts()
    return (object_samples,)


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    selected_features = [
        "band_1_mean", "band_2_mean", "band_3_mean", "band_4_mean",
        "band_1_std",  "band_2_std",  "band_3_std",  "band_4_std",
        "NDWI", "area", "perimeter", "compactness", "elongation"
    ]
    return (selected_features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Downsample for Fast Prototyping (still trainning using tile 35)

    Scale factor 0.25 runs ~16x faster. Set to 1.0 for the final run.
    """)
    return


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    scale_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.25,
        show_value=True, label="Scale factor", debounce=True
    )
    scale_slider
    return (scale_slider,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Visualise Training Sample Distribution
    """)
    return


@app.cell
def _(image_data, model_selector, scale_slider, transform):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    _sf      = scale_slider.value

    ds_image = scipy.ndimage.zoom(image_data, zoom=(1, _sf, _sf), order=0) if _sf < 1.0 else image_data

    ds_transform = transform * Affine.scale(1 / _sf)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Trainning the model of choice
    """)
    return


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    n_trees_slider = mo.ui.slider(
        start=50, stop=500, step=50, value=200,
        show_value=True, label="Number of trees", debounce=True
    )
    n_trees_slider
    return (n_trees_slider,)


@app.cell
def _(model_selector, n_trees_slider, object_samples, selected_features):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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
    return (clf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## eCognition segments
    """)
    return


@app.cell
def _(RGB_PATH, model_selector, segments_raw):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("A method that does not require eCognition segments was selected")
    )

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
    mo.md(r"""
    ## Random Forest Classification
    """)
    return


@app.cell
def _(clf, model_selector, seg_layer, selected_features):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    seg_layer.objects["classification"] = clf.predict(
        seg_layer.objects[selected_features].fillna(0)
    )
    classified_layer = ns.Layer(name="RF_Classification", layer_type="classification")
    classified_layer.objects = seg_layer.objects.copy()
    _c = classified_layer.objects["classification"].value_counts()
    _s = " | ".join([f"{k}: {v}" for k, v in _c.items()])
    mo.md(f"Total Ships in the image: {_s}")
    return (classified_layer,)


@app.cell
def _(classified_layer, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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
def _(gdf, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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
    return (vessels,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Size Classification Map
    """)
    return


@app.cell
def _(gdf, model_selector, vessels):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive Vessel Explorer

    Adjust filters and the map updates reactively.
    """)
    return


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("")
    )

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GeoAI method
    """)
    return


@app.cell
def _(BASE_DIR, model_selector):
    mo.stop(
        model_selector.value != "Geoai",
        mo.md("⚠️ Another method was selected")
    )

    ortho_path = BASE_DIR / "data_store" / "BE_ORTHO_27032011_315140_56865_UTM31N.tif"
    resampled_path = BASE_DIR/"ships_15cm.tif"
    ships_masks = BASE_DIR/"ships_masks.tif"
    ships_geojson = BASE_DIR/"ships.geojson"
    return resampled_path, ships_geojson, ships_masks


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Geoai",
        mo.md("⚠️ Another method was selected")
    )

    rasterio_proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    os.environ["PROJ_LIB"] = rasterio_proj_dir
    os.environ["PROJ_DATA"] = rasterio_proj_dir
    return


@app.cell
def _(model_selector):
    mo.stop(
        model_selector.value != "Geoai",
        mo.md("⚠️ Another method was selected")
    )

    import geoai
    detector = geoai.ShipDetector()
    return (detector,)


@app.cell
def _(detector, model_selector, resampled_path, ships_masks):
    mo.stop(
        model_selector.value != "Geoai",
        mo.md("⚠️ Another method was selected")
    )

    masks_path_resampled = detector.generate_masks(
        resampled_path,
        output_path=ships_masks,
        confidence_threshold=0.5,
        mask_threshold=0.5,       
        overlap=0.5,              
        chip_size=(512, 512),     
        batch_size=4,
    )
    return


@app.cell
def _(detector, model_selector, ships_geojson, ships_masks):
    mo.stop(
        model_selector.value != "Geoai",
        mo.md("⚠️ Another method was selected")
    )

    gdf_masked = detector.vectorize_masks(
        ships_masks,
        output_path=ships_geojson,
        confidence_threshold=0.8,
        min_object_area=100,
        max_object_size=100000,
    )
    return (gdf_masked,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Classification
    """)
    return


@app.cell
def _(
    RGB_PATH,
    filtered,
    gdf_masked,
    model_selector,
    resampled_path,
    tile_selector,
):
    if model_selector.value == "Random Forest":
        VCOL = {"SHIP_WATER": "#e63946", "SHIP_DOCK": "#f4a261", "SHIP": "#0096FF"}
        fig_m, ax_m = plt.subplots(figsize=(10, 10))

        with rasterio.open(RGB_PATH) as s:
            from rasterio.plot import show as rs
            rs(s, ax=ax_m)

        for cls2, col2 in VCOL.items():
            sub = filtered[filtered["classification"] == cls2]
            if len(sub) > 0:
                sub.plot(ax=ax_m, color=col2, alpha=0.7, edgecolor="white", linewidth=0.5, label=cls2)

        ax_m.legend(handles=[mpatches.Patch(facecolor=v, label=k) for k, v in VCOL.items()], loc="upper right")

        ax_m.set_title(f"{len(filtered)} vessel(s) in the image"); ax_m.axis("off")

        plt.tight_layout()
        ax = mo.ui.matplotlib(ax_m, debounce=True)
        ax

    elif model_selector.value ==  "Geoai":
        if tile_selector == "Tile 40":
            # 1. Read RGB raster
            with rasterio.open(resampled_path) as _src:
                img = _src.read([1, 2, 3]).transpose(1, 2, 0)  # (H,W,3)
                extent = [
                    _src.bounds.left, _src.bounds.right,
                    _src.bounds.bottom, _src.bounds.top
                ]
                raster_crs = _src.crs

            # 2. Reproject gdf to raster CRS if needed
            _gdf_plot = gdf_masked.to_crs(raster_crs) if gdf_masked.crs != raster_crs else gdf_masked

            # 3. Plot
            fig_m, ax_m = plt.subplots(figsize=(10, 10))
            ax_m.imshow(img, extent=extent)
            _gdf_plot.plot(ax=ax_m, facecolor="none", edgecolor="red", linewidth=1.5,
                           label="Detected vessels")
            ax_m.legend(loc="upper right")
            ax_m.set_title(f"{len(_gdf_plot)} vessel(s) detected")
            ax_m.axis("off")

            plt.tight_layout()
            ax = mo.ui.matplotlib(ax_m, debounce=True)
            ax

    else:
            print("Analysis on tile 35 is not available")
    return (ax,)


@app.cell
def _(ax):
    ax
    return


@app.cell
def _(ax, filtered, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    #Filtering the dataframe using the interactive aoi
    aoi = ax.value
    BS = mo._plugins.ui._impl.mpl.BoxSelection

    if isinstance(aoi, BS):
        aoi_vessels = filtered[
            (filtered.geometry.centroid.x >= aoi.x_min) &
            (filtered.geometry.centroid.x <= aoi.x_max) &
            (filtered.geometry.centroid.y >= aoi.y_min) &
            (filtered.geometry.centroid.y <= aoi.y_max)
        ]
    else:
        aoi_vessels = filtered
    return (aoi_vessels,)


@app.cell
def _(aoi_vessels, model_selector):
    mo.stop(
        model_selector.value != "Random Forest",
        mo.md("⚠️ Another method was selected")
    )

    fig_h, ax_h = plt.subplots(2, 1, figsize=(15, 4))

    #Plotting vessel count
    vc = aoi_vessels["classification"].value_counts()

    ax_h[0].barh(vc.index, 
                vc.values,
                color=[
                    "#e63946" if "WATER" in c else "#f4a261" 
                    if "DOCK" in c else "#ffb3b3" for c in vc.index])

    ax_h[0].set_title(f"Vessel Count ({len(aoi_vessels)} in selection)")

    #Plotting altittude distribution by class
    sw = aoi_vessels[aoi_vessels["classification"] == "SHIP_WATER"]["ndsm_mean"].dropna()

    sd = aoi_vessels[aoi_vessels["classification"] == "SHIP_DOCK"]["ndsm_mean"].dropna()

    ax_h[1].hist(
        sw, 
        bins=8, 
        color="#e63946", 
        alpha=0.8, 
        label="SHIP_WATER", 
        edgecolor="white",
        orientation="horizontal")

    ax_h[1].hist(
        sd, 
        bins=8, 
        color="#f4a261", 
        alpha=0.8, 
        label="SHIP_DOCK",  
        edgecolor="white",
        orientation="horizontal")

    ax_h[1].set_ylabel("nDSM mean (m)"); ax_h[1].set_xlabel("Count")
    ax_h[1].set_title("Height Distribution by Class (~3m=water, ~8m=dock)")
    ax_h[1].set_ylim(-10, 10); 
    ax_h[1].legend()

    # Source - https://stackoverflow.com/a/28720127
    # Posted by igr, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-25, License - CC BY-SA 3.0
    ax_h[0].spines['top'].set_visible(False)
    ax_h[0].spines['right'].set_visible(False)

    ax_h[1].spines['top'].set_visible(False)
    ax_h[1].spines['right'].set_visible(False)



    plt.tight_layout()
    ay = plt.gcf()
    ay
    return


if __name__ == "__main__":
    app.run()
