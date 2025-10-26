# Taming 3DGS - Codebase Architecture Documentation

> **Comprehensive guide to understanding the Python codebase architecture**
> Last Updated: October 26, 2025

## Table of Contents

1. [Overview](#overview)
2. [Core Entry Points](#core-entry-points)
3. [Scene Representation](#scene-representation)
4. [Gaussian Rendering](#gaussian-rendering)
5. [Utilities](#utilities)
6. [Arguments and Configuration](#arguments-and-configuration)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Key Algorithms](#key-algorithms)

---

## Overview

Taming 3DGS is a research implementation for high-quality 3D scene reconstruction using Gaussian Splatting with budget constraints. The codebase is organized into several modules:

```
taming-3dgs/
├── train.py                    # Main training script
├── render.py                   # Rendering trained models
├── metrics.py                  # Evaluation metrics (PSNR, SSIM, LPIPS)
├── full_eval.py                # Full evaluation pipeline
├── convert.py                  # COLMAP data conversion
├── arguments/                  # Argument parsing and configuration
├── scene/                      # Scene representation and data loading
├── gaussian_renderer/          # Rendering engine
├── utils/                      # Utility functions
└── lpipsPyTorch/              # LPIPS perceptual loss

Key dependencies:
- PyTorch (2.4+ for RTX 5080/5090)
- CUDA custom extensions (diff-gaussian-rasterization, simple-knn, fused-ssim)
```

---

## Core Entry Points

### 1. train.py - Main Training Pipeline

**Purpose**: Trains 3D Gaussian Splatting models with budget-constrained densification.

**Location**: `train.py`

**Key Components**:

#### Main Function: `training()`
```python
def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, websockets,
             score_coefficients, args)
```

**What it does**:
1. **Initialization** (lines 37-48):
   - Creates GaussianModel and Scene objects
   - Loads checkpoint if provided
   - Sets up background color (white or black)

2. **Main Training Loop** (lines 77-237):
   - **Per Iteration**:
     - Select random camera viewpoint (lines 100-105)
     - Render scene from that viewpoint (line 114)
     - Compute loss: L1 + SSIM (lines 118-121)
     - Backpropagate gradients (line 130)
     - Update learning rate (line 93)

3. **Densification** (lines 154-187):
   - Runs between iterations 500-15000 at specified intervals
   - **Importance Scoring**: Computes Gaussian importance using `compute_gaussian_score()` (line 174)
   - **Budget Target**: Gets target count from `counts_array` (line 178)
   - **Densification**: Calls `densify_with_score()` to add/remove Gaussians (lines 180-186)

4. **Opacity Reset** (lines 189-190):
   - Resets opacities every 3000 iterations to prevent "floaters"

5. **Optimizer Step** (lines 196-217):
   - Updates Gaussian parameters using Adam or SparseGaussianAdam
   - Optional: Less frequent SH updates with `--sh_lower` flag

**Key Functions**:

##### `prepare_output_and_logger(args)` (lines 242-262)
- Creates output directory structure
- Saves configuration to `cfg_args` file
- Initializes TensorBoard writer if available

##### `training_report()` (lines 264-305)
- Logs training metrics to TensorBoard
- Runs test set evaluation at specified iterations
- Computes PSNR, SSIM, LPIPS on validation views

**Important Parameters**:
- `--budget`: Final Gaussian count (interpreted by `--mode`)
- `--mode`: Either `multiplier` (budget = multiplier × SfM points) or `final_count` (exact count)
- `--densification_interval`: Frequency of densification operations (default: 100)
- `--cams`: Number of cameras for importance scoring (default: 10)
- `--data_device`: Where to store images (`cuda` or `cpu`)
- `--ho_iteration`: Iteration to switch from sigmoid to absolute opacity (default: 15000)

**Score Coefficients** (line 340):
```python
score_coefficients = {
    'view_importance': 50,
    'edge_importance': 50,
    'mse_importance': 50,
    'grad_importance': 25,
    'dist_importance': 50,
    'opac_importance': 100,
    'dept_importance': 5,
    'loss_importance': 10,
    'radii_importance': 10,
    'scale_importance': 25,
    'count_importance': 0.1,
    'blend_importance': 50
}
```

---

### 2. render.py - Scene Rendering

**Purpose**: Renders trained Gaussian models to generate images.

**Location**: `render.py`

**Key Functions**:

#### `render_set(model_path, name, iteration, views, gaussians, pipeline, background)` (lines 24-35)
**What it does**:
- Renders all views in a camera set (train or test)
- Saves rendered images to `<model_path>/<name>/ours_<iteration>/renders/`
- Saves ground truth images to `<model_path>/<name>/ours_<iteration>/gt/`

**Parameters**:
- `model_path`: Path to trained model
- `name`: "train" or "test"
- `iteration`: Which checkpoint to load (-1 for latest)
- `views`: List of Camera objects
- `gaussians`: GaussianModel instance
- `pipeline`: PipelineParams configuration
- `background`: Background color tensor

**Rendering Mode**:
- Uses `rendering_mode="abs"` for absolute opacity values (not sigmoid)
- This ensures consistent rendering at test time

#### `render_sets()` (lines 37-49)
**What it does**:
- Wrapper function that renders both train and test sets
- Loads trained model from checkpoint
- Calls `render_set()` for each camera set

**Usage**:
```bash
python render.py -m <model_path> [--iteration 30000] [--skip_train] [--skip_test]
```

---

### 3. metrics.py - Evaluation Metrics

**Purpose**: Computes image quality metrics for rendered scenes.

**Location**: `metrics.py`

**Key Functions**:

#### `readImages(renders_dir, gt_dir)` (lines 24-34)
**What it does**:
- Loads rendered and ground truth images from directories
- Converts to PyTorch tensors on GPU
- Returns lists of (renders, ground_truths, image_names)

#### `evaluate(model_paths)` (lines 36-93)
**What it does**:
1. For each scene in `model_paths`:
2. For each method in `<scene>/test/`:
3. Load all rendered and GT images
4. Compute per-image metrics:
   - **SSIM**: Structural Similarity Index
   - **PSNR**: Peak Signal-to-Noise Ratio
   - **LPIPS**: Learned Perceptual Image Patch Similarity (VGG-based)
5. Save results to:
   - `<scene>/results.json`: Average metrics
   - `<scene>/per_view.json`: Per-image metrics

**Metrics Explained**:
- **PSNR** (higher is better): Measures pixel-wise accuracy
- **SSIM** (higher is better): Measures structural similarity
- **LPIPS** (lower is better): Measures perceptual similarity using deep features

**Usage**:
```bash
python metrics.py -m <model_path1> [<model_path2> ...]
```

---

### 4. full_eval.py - Complete Evaluation Pipeline

**Purpose**: Automated training, rendering, and metrics computation for all datasets.

**Location**: `full_eval.py`

**Key Components**:

#### Dataset Organization (lines 15-52)
```python
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

# Budget multipliers for "budget" mode
budget_multipliers = {
    "bicycle": 15, "flowers": 15, "garden": 15, ...  # outdoor: 15x
    "room": 2, "counter": 2, ...                     # indoor: 2x
}

# Exact counts for "big" mode
big_budgets = {
    "bicycle": 5987095, "flowers": 3618411, ...
}
```

#### Execution Flow:
1. **Training Phase** (lines 82-149):
   - Runs `train.py` for each scene with appropriate budgets
   - Uses `--densification_interval 100` for big mode, `500` for budget mode
   - Records timing for each dataset group

2. **Rendering Phase** (lines 155-166):
   - Runs `render.py` for all trained models

3. **Metrics Phase** (lines 168-179):
   - Runs `metrics.py` for all rendered scenes

**Usage**:
```bash
python full_eval.py \
    --mipnerf360 /path/to/mipnerf360/ \
    --tanksandtemples /path/to/tandt/ \
    --deepblending /path/to/db/ \
    --mode budget \          # or --mode big
    [--skip_training] \      # Skip training phase
    [--skip_rendering] \     # Skip rendering phase
    [--skip_metrics] \       # Skip metrics phase
    [--dry_run]             # Print commands without executing
```

---

### 5. convert.py - COLMAP Dataset Conversion

**Purpose**: Converts raw images to COLMAP format with multiple resolutions.

**Location**: `convert.py`

**Workflow**:

1. **Feature Extraction** (lines 34-44):
   - Runs COLMAP feature extractor on input images
   - Detects SIFT keypoints

2. **Feature Matching** (lines 46-53):
   - Matches features between images using exhaustive matcher

3. **Bundle Adjustment** (lines 55-66):
   - Runs COLMAP mapper to estimate camera poses and 3D points

4. **Image Undistortion** (lines 68-78):
   - Undistorts images to ideal pinhole camera model

5. **Multi-Resolution Generation** (lines 90-122):
   - Creates `images_2/` (50% scale)
   - Creates `images_4/` (25% scale)
   - Creates `images_8/` (12.5% scale)

**Usage**:
```bash
python convert.py -s <source_path> [--resize] [--no_gpu]
```

---

## Scene Representation

### 1. scene/__init__.py - Scene Manager

**Purpose**: Loads datasets, manages cameras, and initializes Gaussian models.

**Location**: `scene/__init__.py`

**Class: `Scene`**

#### `__init__(args, gaussians, load_iteration, shuffle, resolution_scales)` (lines 25-83)
**What it does**:
1. **Dataset Detection** (lines 43-49):
   - Checks for `sparse/` directory → COLMAP format
   - Checks for `transforms_train.json` → Blender format
   - Calls appropriate reader from `sceneLoadTypeCallbacks`

2. **Camera Loading** (lines 71-75):
   - Loads train and test cameras at specified resolution scales
   - Converts CameraInfo to Camera objects

3. **Gaussian Initialization** (lines 77-83):
   - If loading checkpoint: Loads from `.ply` file
   - Else: Initializes from point cloud using `create_from_pcd()`

#### Key Methods:

##### `save(iteration)` (lines 85-87)
- Saves Gaussian model to `<model_path>/point_cloud/iteration_<N>/point_cloud.ply`

##### `getTrainCameras(scale)` / `getTestCameras(scale)` (lines 89-92)
- Returns camera lists for train/test splits at given resolution scale

---

### 2. scene/gaussian_model.py - Gaussian Model

**Purpose**: Core representation of 3D Gaussians with all optimization logic.

**Location**: `scene/gaussian_model.py`

**Class: `GaussianModel`**

**Gaussian Properties** (lines 64-69):
```python
self._xyz          # 3D positions
self._features_dc  # DC component of spherical harmonics (base color)
self._features_rest  # Higher-order SH coefficients (view-dependent color)
self._scaling      # 3D scale (log space)
self._rotation     # Rotation quaternion
self._opacity      # Opacity value
```

#### Key Methods:

##### `__init__(sh_degree, optimizer_type, rendering_mode)` (lines 60-78)
**What it does**:
- Initializes empty Gaussian parameters
- Sets up activation functions based on `rendering_mode`:
  - `None`: Sigmoid opacity (training mode)
  - `"abs"`: Absolute opacity (rendering mode)

##### `create_from_pcd(pcd, spatial_lr_scale)` (lines 154-208)
**What it does**:
1. **Color Initialization** (lines 156-160):
   - Converts RGB colors to SH coefficients using `RGB2SH()`
   - Sets DC component, zeros out higher frequencies

2. **Scale Initialization** (lines 164-194):
   - **Memory-Efficient k-NN**: Computes 3-nearest neighbor distances
   - Handles OOM errors with batch processing (50k points at a time)
   - Falls back to fixed scale (0.01) if k-NN fails
   - Uses `distCUDA2()` from simple-knn extension

3. **Parameter Setup** (lines 196-208):
   - Positions: From point cloud
   - Scales: `log(sqrt(dist_to_3nn))`
   - Rotations: Identity quaternion [1,0,0,0]
   - Opacities: 0.1 (in inverse sigmoid space)

##### `training_setup(training_args)` (lines 210-232)
**What it does**:
- Creates optimizer parameter groups with different learning rates:
  - xyz: `0.00016 → 0.0000016` (exponential decay)
  - features_dc: `0.0025`
  - features_rest: `0.00025` (via separate shoptimizer)
  - opacity: `0.025`
  - scaling: `0.005`
  - rotation: `0.001`

##### `densify_with_score(scores, ...)` (lines 536-593)
**What it does** (Taming-3DGS core algorithm):
1. **Qualification** (lines 538-549):
   - Clone qualifiers: Small Gaussians with high gradients
   - Split qualifiers: Large Gaussians with high gradients

2. **Budget Allocation** (lines 551-562):
   - Available budget = target - current count
   - Distributes budget proportionally between cloning and splitting

3. **Importance-Based Sampling** (lines 565-568):
   - Uses `torch.multinomial()` with importance scores
   - Clones most important small Gaussians
   - Splits most important large Gaussians

4. **Pruning** (lines 570-588):
   - Removes low-opacity Gaussians
   - Removes oversized Gaussians (in screen space or world space)
   - Uses inverse importance sampling for controlled removal

##### `densify_and_clone_taming(grads, budget, filter)` (lines 518-534)
**What it does**:
- Samples `budget` Gaussians based on importance scores
- Duplicates their parameters exactly
- Adds them to the model

##### `densify_and_split_taming(grads, budget, filter, N=2)` (lines 490-516)
**What it does**:
- Samples `budget` Gaussians based on importance scores
- Splits each into N=2 child Gaussians:
  - Positions: Parent position + Gaussian noise scaled by parent scale
  - Scales: Parent scale / (0.8 * N)
  - Other params: Copied from parent
- Removes parent Gaussians

##### `save_ply(path)` / `load_ply(path)` (lines 256-319)
- Saves/loads Gaussian parameters to/from PLY format
- PLY attributes: `[x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*]`

---

### 3. scene/dataset_readers.py - Data Loading

**Purpose**: Parses COLMAP and Blender dataset formats.

**Location**: `scene/dataset_readers.py`

**Key Data Structures**:

#### `CameraInfo` (NamedTuple, lines 26-36)
```python
uid: int              # Unique camera ID
R: np.array           # Rotation matrix (3x3)
T: np.array           # Translation vector (3,)
FovY: float           # Vertical field of view
FovX: float           # Horizontal field of view
image: PIL.Image      # Loaded image
image_path: str       # Path to image file
image_name: str       # Image filename (no extension)
width: int            # Image width
height: int           # Image height
```

#### `SceneInfo` (NamedTuple, lines 38-43)
```python
point_cloud: BasicPointCloud      # Initial 3D points
train_cameras: List[CameraInfo]   # Training camera views
test_cameras: List[CameraInfo]    # Test camera views
nerf_normalization: dict          # Scene normalization (center, radius)
ply_path: str                     # Path to point cloud file
```

#### Key Functions:

##### `readColmapSceneInfo(path, images, eval, llffhold=8)` (lines 132-177)
**What it does**:
1. **Load COLMAP Data** (lines 133-142):
   - Reads `sparse/0/images.bin` (or `.txt`) for camera extrinsics
   - Reads `sparse/0/cameras.bin` (or `.txt`) for camera intrinsics

2. **Parse Cameras** (line 145):
   - Calls `readColmapCameras()` to create CameraInfo objects

3. **Train/Test Split** (lines 148-153):
   - If `eval=True`: Every 8th image → test set (llffhold=8)
   - Else: All images → train set

4. **Load Point Cloud** (lines 157-170):
   - Converts `points3D.bin` to `.ply` if needed
   - Loads using `fetchPly()`

5. **Compute Normalization** (line 155):
   - Centers scene at camera centroid
   - Scales to fit within unit sphere (×1.1 margin)

##### `readNerfSyntheticInfo(path, white_background, eval)` (lines 221-255)
**What it does**:
1. **Load Transforms** (lines 223-225):
   - Reads `transforms_train.json`
   - Reads `transforms_test.json`

2. **Parse Blender Cameras** (via `readCamerasFromTransforms`, lines 179-219):
   - Converts Blender's camera-to-world matrices to COLMAP format
   - Handles RGBA images with alpha compositing

3. **Generate Point Cloud** (lines 234-244):
   - Since Blender scenes have no SfM data
   - Generates 100,000 random points in range `[-1.3, 1.3]`

**Scene Normalization** (`getNerfppNorm`, lines 45-66):
```python
# Centers cameras and computes bounding sphere
center = mean(camera_positions)
radius = 1.1 * max(distance(camera_positions, center))
return {"translate": -center, "radius": radius}
```

---

### 4. scene/cameras.py - Camera Representation

**Purpose**: Represents individual camera viewpoints.

**Location**: `scene/cameras.py`

**Class: `Camera(nn.Module)`**

#### `__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, trans, scale, data_device)` (lines 18-57)

**What it does**:
1. **Store Camera Parameters** (lines 24-30):
   - Rotation matrix R, translation vector T
   - Field of view (FoVx, FoVy)
   - Image name and unique ID

2. **Image Loading** (lines 32-46):
   - **Key Feature**: Loads image to specified `data_device` (CPU or CUDA)
   - Applies alpha mask if provided
   - Stores in `self.original_image`

3. **Compute Matrices** (lines 54-56):
   - `world_view_transform`: World → camera transform (4x4, on CUDA)
   - `projection_matrix`: Camera → NDC projection (on CUDA)
   - `full_proj_transform`: Combined world → NDC

**Important**: Images can be on CPU (`--data_device cpu`) to save GPU memory, but transformation matrices are always on CUDA for rendering.

---

## Gaussian Rendering

### 1. gaussian_renderer/__init__.py - Rendering Engine

**Purpose**: Renders 3D Gaussians to 2D images using custom CUDA rasterizer.

**Location**: `gaussian_renderer/__init__.py`

**Main Function: `render()`** (lines 18-128)

#### `render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None, pixel_weights=None)`

**What it does**:

1. **Setup Rasterization Settings** (lines 26-50):
```python
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
    tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
    bg=bg_color,                              # Background color
    scale_modifier=scaling_modifier,          # Global scale multiplier
    viewmatrix=viewpoint_camera.world_view_transform,
    projmatrix=viewpoint_camera.full_proj_transform,
    sh_degree=pc.active_sh_degree,           # Current SH degree
    campos=viewpoint_camera.camera_center,
    prefiltered=False,
    debug=pipe.debug,
    pixel_weights=pixel_weights              # For importance scoring
)
```

2. **Prepare Gaussian Data** (lines 54-68):
   - Positions: `pc.get_xyz`
   - Opacities: `pc.get_opacity` (sigmoid or abs depending on mode)
   - **Covariance**: Either precomputed 3D or scales + rotations
   - **Colors**: Either from SH coefficients or precomputed RGB

3. **Rasterize** (lines 91-110):
   - Calls CUDA rasterizer from `diff-gaussian-rasterization`
   - Returns:
     - `rendered_image`: RGB image (3, H, W)
     - `radii`: Screen-space radii of each Gaussian
     - Additional outputs for importance scoring:
       - `accum_weights`: Per-Gaussian weighted loss accumulation
       - `accum_dist`: Distance accumulation
       - `accum_blend`: Blending weights
       - `accum_count`: Render counts
       - `gaussian_depths`: Depth values
       - `gaussian_radii`: Radii in pixels

**Return Dictionary** (lines 114-128):
```python
{
    "render": rendered_image,              # Final RGB image
    "viewspace_points": screenspace_points,  # 2D projections (for gradients)
    "visibility_filter": (radii > 0),      # Which Gaussians are visible
    "radii": radii,                        # Screen-space radii

    # For importance scoring:
    "accum_weights": accum_weights,        # Loss accumulation per Gaussian
    "accum_dist": accum_dist,              # Distance accumulation
    "accum_blend": accum_blend,            # Blending weights
    "accum_count": accum_count,            # Render count
    "gaussian_depths": depths,             # Depth values
    "gaussian_radii": my_radii            # Pixel radii
}
```

**Rendering Modes**:

- **Training Mode** (sigmoid opacity):
  - Opacity ∈ [0, 1] via sigmoid activation
  - Allows gradient-based optimization
  - Can have "semi-transparent" Gaussians

- **Rendering Mode** (absolute opacity):
  - Opacity = |raw_value|
  - Used after `ho_iteration` (default: 15000)
  - More stable for final rendering

---

### 2. gaussian_renderer/network_gui_ws.py - Web Viewer

**Purpose**: WebSocket-based real-time viewer for monitoring training.

**Location**: `gaussian_renderer/network_gui_ws.py`

**Functionality**:
- Starts WebSocket server on specified port
- Streams rendered images during training
- Allows user to select which training camera to view
- Updates in real-time as training progresses

**Usage in train.py** (lines 79-85):
```python
if websockets:
    if network_gui_ws.curr_id >= 0:
        cam = scene.getTrainCameras()[network_gui_ws.curr_id]
        net_image = render(cam, gaussians, pipe, background, 1.0)["render"]
        network_gui_ws.latest_result = net_image_bytes
```

---

## Utilities

### 1. utils/taming_utils.py - Taming Algorithm Core

**Purpose**: Implements the importance scoring mechanism for budget-constrained densification.

**Location**: `utils/taming_utils.py`

**Key Functions**:

#### `compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg, importance_values, opt)` (lines 40-91)

**What it does** (Core of Taming-3DGS):

1. **Initialize** (lines 43-51):
   - Create importance tensor: `(num_cameras, num_gaussians)`
   - Get global Gaussian properties: opacity, scales, gradients

2. **Per-Camera Scoring** (lines 53-88):
   For each camera view:

   a. **Render Image** (line 55):
      - Get current reconstruction

   b. **Compute Edge-Weighted Loss Map** (lines 58-59):
      - Combines MSE loss with edge detection
      - Weights = `mse_importance * L1_loss + edge_importance * edges`

   c. **Render with Pixel Weights** (lines 61-66):
      - Passes `pixel_weights` to rasterizer
      - Rasterizer accumulates weighted losses per Gaussian
      - Returns per-Gaussian importance metrics

   d. **Gaussian-Level Importance** (lines 73-78):
      ```python
      g_importance = (
          normalize(grad_importance, all_grads) +
          normalize(opac_importance, all_opacity) +
          normalize(dept_importance, all_depths) +
          normalize(radii_importance, all_radii) +
          normalize(scale_importance, all_scales)
      )
      ```

   e. **Pixel-Level Importance** (lines 80-85):
      ```python
      p_importance = (
          normalize(dist_importance, dist_accum) +
          normalize(loss_importance, loss_accum) +
          normalize(count_importance, reverse_counts) +
          normalize(blend_importance, blending_weights)
      )
      ```

   f. **Aggregate** (line 87):
      ```python
      agg_importance = view_importance * photometric_loss * (p_importance + g_importance)
      ```

3. **Sum Across Views** (line 90):
   - Final score = sum of importance across all sampled cameras

**Importance Components**:

| Component | What It Measures |
|-----------|------------------|
| `grad_importance` | How much position gradients this Gaussian receives |
| `opac_importance` | Opacity value (higher = more important) |
| `dept_importance` | Depth (closer = more important) |
| `radii_importance` | Screen-space size |
| `scale_importance` | World-space size (volume) |
| `dist_importance` | Distance accumulation (rendering proximity) |
| `loss_importance` | Per-Gaussian contribution to reconstruction loss |
| `count_importance` | How many pixels this Gaussian affects |
| `blend_importance` | Blending weights in alpha composition |
| `edge_importance` | Contribution to edge reconstruction |
| `mse_importance` | Contribution to photometric error |

#### `get_count_array(start_count, multiplier, opt, mode)` (lines 100-117)

**What it does**:
- Generates densification schedule following Equation (2) from the paper
- Uses quadratic growth: `count(x) = a*x² + b*x + c`
  - `c = start_count` (initial SfM points)
  - `budget = final target count`
  - `num_steps = (15000 - 500) / densification_interval`
- Returns array of target counts for each densification iteration

**Example**:
```python
# Budget mode with multiplier=15
start_count = 100000  # SfM points
multiplier = 15
budget = 1500000      # 15x initial

# Generates: [100000, 120000, 145000, ..., 1500000]
# with quadratic acceleration
```

#### `get_edges(image)` (lines 9-15)

**What it does**:
- Converts image to grayscale
- Applies PIL edge detection filter
- Returns edge map as tensor
- Used for edge-weighted importance scoring

#### `normalize(config_value, value_tensor)` (lines 28-38)

**What it does**:
- Normalizes importance values by median
- Formula: `normalized = config_value * (value / median(value))`
- Handles NaN and zero values
- Ensures numerical stability

---

### 2. utils/loss_utils.py - Loss Functions

**Purpose**: Photometric loss functions.

**Location**: `utils/loss_utils.py`

**Key Functions**:

#### `l1_loss(network_output, gt)`
- Computes L1 (absolute) distance between rendered and ground truth
- `loss = |rendered - gt|`

#### `ssim(img1, img2)`
- Structural Similarity Index Measure
- Considers luminance, contrast, and structure
- Returns value in [0, 1] (1 = identical)

**Note**: The codebase uses `fused_ssim` from the CUDA extension for faster computation during training.

---

### 3. utils/camera_utils.py - Camera Utilities

**Purpose**: Camera data conversion and management.

**Location**: `utils/camera_utils.py`

**Key Functions**:

#### `cameraList_from_camInfos(cam_infos, resolution_scale, args)` (lines 30-53)
**What it does**:
- Converts list of `CameraInfo` objects to `Camera` objects
- Applies resolution scaling if requested
- Passes `data_device` parameter for CPU/GPU image loading

#### `camera_to_JSON(id, camera)`
- Serializes camera parameters to JSON format
- Saves intrinsics and extrinsics for later use

---

### 4. utils/graphics_utils.py - Graphics Math

**Purpose**: 3D graphics transformations and projections.

**Location**: `utils/graphics_utils.py`

**Key Functions**:

#### `getWorld2View2(R, T, translate, scale)`
- Constructs world-to-camera transformation matrix
- Applies scene normalization (translation and scaling)

#### `getProjectionMatrix(znear, zfar, fovX, fovY)`
- Builds perspective projection matrix
- Maps camera space to NDC (Normalized Device Coordinates)

#### `focal2fov(focal, pixels)` / `fov2focal(fov, pixels)`
- Converts between focal length and field of view

#### `BasicPointCloud` (NamedTuple)
```python
points: np.array   # (N, 3) positions
colors: np.array   # (N, 3) RGB colors
normals: np.array  # (N, 3) normal vectors
```

---

### 5. utils/sh_utils.py - Spherical Harmonics

**Purpose**: Spherical Harmonics for view-dependent appearance.

**Location**: `utils/sh_utils.py`

**Key Functions**:

#### `RGB2SH(rgb)`
- Converts RGB color to SH coefficient (DC component only)
- Formula: `C0 = 0.28209479177387814`
- `SH_coeff = (rgb - 0.5) / C0`

#### `SH2RGB(sh)`
- Converts SH coefficient back to RGB
- Inverse of `RGB2SH()`

#### `eval_sh(degree, sh, dirs)`
- Evaluates spherical harmonics at given view directions
- Computes view-dependent color from SH coefficients
- Used when `convert_SHs_python=True`

**SH Degrees**:
- Degree 0: 1 coefficient (constant color)
- Degree 1: 4 coefficients (linear variation)
- Degree 2: 9 coefficients (quadratic)
- Degree 3: 16 coefficients (cubic) ← Default

---

### 6. utils/general_utils.py - General Utilities

**Purpose**: Miscellaneous utility functions.

**Location**: `utils/general_utils.py`

**Key Functions**:

#### `inverse_sigmoid(x)`
- Computes logit: `log(x / (1 - x))`
- Used to initialize opacity in sigmoid space

#### `get_expon_lr_func(lr_init, lr_final, lr_delay_mult, max_steps)`
- Returns learning rate scheduler function
- Exponential decay: `lr = lr_init * (lr_final/lr_init)^(t/max_steps)`
- Includes delay period with slower decay

#### `build_rotation(r)`
- Converts quaternion to rotation matrix
- Quaternion format: `[w, x, y, z]`

#### `build_scaling_rotation(s, r)`
- Builds combined scaling + rotation matrix
- Used to construct covariance matrices for Gaussians

#### `strip_symmetric(matrix)`
- Extracts upper triangle of symmetric matrix
- Reduces storage: 9 values → 6 values for 3x3 symmetric

#### `safe_state(silent)`
- Seeds random number generators for reproducibility
- Sets seed for: Python random, NumPy, PyTorch

---

### 7. utils/system_utils.py - System Utilities

**Purpose**: File system and checkpoint management.

**Location**: `utils/system_utils.py`

**Key Functions**:

#### `searchForMaxIteration(folder)`
- Finds latest checkpoint in `<folder>/iteration_*`
- Returns highest iteration number

#### `mkdir_p(folder_path)`
- Creates directory and all parent directories
- Equivalent to `mkdir -p` in Unix

---

### 8. utils/image_utils.py - Image Utilities

**Purpose**: Image processing and metrics.

**Location**: `utils/image_utils.py`

**Key Functions**:

#### `psnr(img1, img2)`
- Computes Peak Signal-to-Noise Ratio
- Formula: `PSNR = 20 * log10(1 / RMSE)`
- Higher values = better quality

---

## Arguments and Configuration

### arguments/__init__.py - Argument Parsing

**Purpose**: Defines all command-line arguments and configuration parameters.

**Location**: `arguments/__init__.py`

**Class Hierarchy**:

```
ParamGroup (base class)
├── ModelParams      # Dataset and scene configuration
├── PipelineParams   # Rendering configuration
└── OptimizationParams  # Training hyperparameters
```

#### **ModelParams** (lines 47-62)

**Dataset Parameters**:
```python
sh_degree: int = 3              # Spherical harmonics degree
source_path: str                # Path to dataset (-s)
model_path: str                 # Output path (-m)
images: str = "images"          # Image subdirectory (-i)
resolution: int = -1            # Downscale factor (-1 = auto)
white_background: bool = False  # Use white background
data_device: str = "cuda"       # Image storage device
eval: bool = False              # Create train/test split
```

**Usage**:
```bash
python train.py \
    -s data/scene \      # source_path
    -m output/scene \    # model_path
    -i images_4 \        # images (4x downsampled)
    --data_device cpu    # Load images to CPU
    --eval               # Create test split
```

#### **PipelineParams** (lines 64-70)

**Rendering Configuration**:
```python
separate_sh: bool = True           # Separate DC and rest SH in rasterizer
convert_SHs_python: bool = False   # Compute SH→RGB in Python (slower)
compute_cov3D_python: bool = False  # Compute covariance in Python (slower)
debug: bool = False                # Enable debug mode
```

**Note**: Keep defaults for best performance. Python implementations are for debugging only.

#### **OptimizationParams** (lines 72-93)

**Training Hyperparameters**:
```python
# Iteration counts
iterations: int = 30_000
densify_from_iter: int = 500          # Start densification
densify_until_iter: int = 15_000      # Stop densification
densification_interval: int = 100     # Densification frequency
opacity_reset_interval: int = 3000    # Opacity reset frequency

# Learning rates
position_lr_init: float = 0.00016     # Initial position LR
position_lr_final: float = 0.0000016  # Final position LR (exponential decay)
position_lr_delay_mult: float = 0.01  # Delay multiplier for LR warmup
position_lr_max_steps: int = 30_000   # Steps for LR decay

feature_lr: float = 0.0025            # Feature (DC) learning rate
shfeature_lr: float = 0.005           # SH feature learning rate
opacity_lr: float = 0.025             # Opacity learning rate
scaling_lr: float = 0.005             # Scaling learning rate
rotation_lr: float = 0.001            # Rotation learning rate

# Densification
percent_dense: float = 0.01           # Threshold for clone vs split
densify_grad_threshold: float = 0.0002  # Gradient threshold for densification

# Loss
lambda_dssim: float = 0.2             # SSIM loss weight (L1 weight = 0.8)
random_background: bool = False        # Random background colors

# Optimizer
optimizer_type: str = "default"       # "default" or "sparse_adam"
```

**Key Parameter Explanations**:

- **densification_interval**:
  - Budget mode: 500 (less frequent, more stable)
  - Big mode: 100 (frequent, aggressive growth)

- **percent_dense**:
  - If `max(scale) > 0.01 * scene_extent`: Split
  - Else: Clone

- **lambda_dssim**:
  - Total loss = `0.8 * L1 + 0.2 * (1 - SSIM)`

#### **get_combined_args(parser)** (lines 95-115)

**What it does**:
- Merges command-line arguments with saved config from `cfg_args`
- Used by render.py and metrics.py to inherit training config
- Command-line args override saved config

---

## Data Flow Diagrams

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                            │
└─────────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ┌──────────────┐
   │ train.py     │
   └──────┬───────┘
          │
          ├─→ ModelParams.extract(args)
          ├─→ Scene(dataset, gaussians)
          │   ├─→ dataset_readers.readColmapSceneInfo()
          │   │   ├─→ Load COLMAP data (cameras, points3D)
          │   │   └─→ Create CameraInfo objects
          │   │
          │   └─→ gaussians.create_from_pcd(point_cloud)
          │       ├─→ distCUDA2(): Compute k-NN distances
          │       ├─→ Initialize positions, colors, scales
          │       └─→ Create nn.Parameters
          │
          └─→ gaussians.training_setup(opt)
              └─→ Create Adam optimizers

2. MAIN LOOP (iter=1 to 30000)
   ┌────────────────────────────────────────────┐
   │ for iteration in range(1, iterations+1):   │
   │                                             │
   │   ┌──────────────────────────────────┐    │
   │   │ A. SELECT CAMERA                  │    │
   │   │    viewpoint_cam = random(cams)   │    │
   │   └──────────────────────────────────┘    │
   │                                             │
   │   ┌──────────────────────────────────┐    │
   │   │ B. RENDER                         │    │
   │   │    render_pkg = render(cam, ...) │    │
   │   │    ├─→ GaussianRasterizer          │    │
   │   │    │   ├─→ Project Gaussians       │    │
   │   │    │   ├─→ Tile-based rendering    │    │
   │   │    │   └─→ Alpha blending          │    │
   │   │    └─→ Returns: image, radii, ...  │    │
   │   └──────────────────────────────────┘    │
   │                                             │
   │   ┌──────────────────────────────────┐    │
   │   │ C. COMPUTE LOSS                   │    │
   │   │    L1 = l1_loss(image, gt)        │    │
   │   │    SSIM = ssim(image, gt)         │    │
   │   │    loss = 0.8*L1 + 0.2*(1-SSIM)  │    │
   │   └──────────────────────────────────┘    │
   │                                             │
   │   ┌──────────────────────────────────┐    │
   │   │ D. BACKWARD & OPTIMIZER STEP      │    │
   │   │    loss.backward()                │    │
   │   │    optimizer.step()               │    │
   │   └──────────────────────────────────┘    │
   │                                             │
   │   ┌──────────────────────────────────┐    │
   │   │ E. DENSIFICATION (every N iters)  │    │
   │   │    if iter % densify_interval:    │    │
   │   │      ├─→ compute_gaussian_score() │    │
   │   │      │   ├─→ Render with weights  │    │
   │   │      │   ├─→ Compute importance   │    │
   │   │      │   └─→ Return scores        │    │
   │   │      │                             │    │
   │   │      └─→ densify_with_score()     │    │
   │   │          ├─→ Allocate budget      │    │
   │   │          ├─→ Clone small Gaussians│    │
   │   │          ├─→ Split large Gaussians│    │
   │   │          └─→ Prune low-importance │    │
   │   └──────────────────────────────────┘    │
   └────────────────────────────────────────────┘

3. SAVE MODEL
   ┌──────────────────┐
   │ scene.save(iter) │
   └──────┬───────────┘
          │
          └─→ gaussians.save_ply()
              └─→ Saves to point_cloud/iteration_N/point_cloud.ply
```

### Rendering Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        RENDERING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

1. LOAD TRAINED MODEL
   ┌──────────────┐
   │ render.py    │
   └──────┬───────┘
          │
          ├─→ Scene(dataset, gaussians, load_iteration=N)
          │   └─→ gaussians.load_ply(path)
          │       └─→ Loads: _xyz, _features_dc, _opacity, ...
          │
          └─→ Set rendering_mode="abs"
              └─→ Use absolute opacity (not sigmoid)

2. RENDER LOOP
   for each camera in train_cameras + test_cameras:

   ┌──────────────────────────────────────┐
   │ A. RENDER IMAGE                       │
   │    image = render(cam, gaussians)    │
   │    └─→ gaussian_renderer.render()    │
   │        ├─→ Setup rasterizer          │
   │        ├─→ Project 3D → 2D           │
   │        ├─→ Tile-based splatting      │
   │        └─→ Return RGB image          │
   └──────────────────────────────────────┘
          │
          ├─→ Save render to renders/XXXXX.png
          └─→ Save GT to gt/XXXXX.png

3. OUTPUT STRUCTURE
   <model_path>/
   ├─ train/ours_30000/
   │  ├─ renders/
   │  │  ├─ 00000.png
   │  │  ├─ 00001.png
   │  │  └─ ...
   │  └─ gt/
   │     ├─ 00000.png
   │     └─ ...
   └─ test/ours_30000/
      ├─ renders/
      └─ gt/
```

### Metrics Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       METRICS COMPUTATION                        │
└─────────────────────────────────────────────────────────────────┘

1. LOAD IMAGES
   ┌──────────────┐
   │ metrics.py   │
   └──────┬───────┘
          │
          └─→ readImages(renders_dir, gt_dir)
              ├─→ Load all PNG files
              ├─→ Convert to tensors
              └─→ Move to GPU

2. COMPUTE PER-IMAGE METRICS
   for each (render, gt) pair:

   ┌───────────────────────────────────┐
   │ SSIM = ssim(render, gt)           │
   │   └─→ Structural similarity [0,1] │
   │                                    │
   │ PSNR = psnr(render, gt)           │
   │   └─→ Peak SNR in dB              │
   │                                    │
   │ LPIPS = lpips(render, gt)         │
   │   └─→ VGG perceptual distance     │
   └───────────────────────────────────┘

3. AGGREGATE AND SAVE
   ├─→ Compute mean across all images
   └─→ Save to JSON:
       ├─ results.json: {method: {SSIM, PSNR, LPIPS}}
       └─ per_view.json: {method: {metric: {img_name: value}}}
```

---

## Key Algorithms

### 1. Importance-Based Densification

**Algorithm**: `densify_with_score()` in `scene/gaussian_model.py:536-593`

**Input**:
- `scores`: Importance score for each Gaussian (from `compute_gaussian_score()`)
- `budget`: Target number of Gaussians
- `extent`: Scene bounding sphere radius

**Steps**:

1. **Compute Gradients and Qualifiers**:
```python
grads = xyz_gradient_accum / denom  # Average position gradients
grad_qualifiers = (norm(grads) >= 0.0002)  # High-gradient Gaussians

clone_qualifiers = (max(scale) <= 0.01 * extent)  # Small Gaussians
split_qualifiers = (max(scale) > 0.01 * extent)   # Large Gaussians

all_clones = grad_qualifiers AND clone_qualifiers
all_splits = grad_qualifiers AND split_qualifiers
```

2. **Allocate Budget**:
```python
curr_points = len(gaussians)
available_budget = max(0, budget - curr_points)

clone_budget = (available_budget * num_clones) / (num_clones + num_splits)
split_budget = (available_budget * num_splits) / (num_clones + num_splits)
```

3. **Sample Based on Importance**:
```python
# Clone: Duplicate most important small Gaussians
if clone_budget > 0:
    indices = torch.multinomial(scores * all_clones, clone_budget)
    duplicate(gaussians[indices])

# Split: Divide most important large Gaussians
if split_budget > 0:
    indices = torch.multinomial(scores * all_splits, split_budget)
    split(gaussians[indices], N=2)
```

4. **Prune Low-Importance Gaussians**:
```python
prune_mask = (opacity < min_opacity) OR
             (radii2D > max_screen_size) OR
             (scale > 0.1 * extent)

# Inverse importance sampling
remove_budget = 0.5 * sum(prune_mask)
inverse_scores = 1 / (1e-6 + scores)
to_remove = torch.multinomial(inverse_scores, remove_budget)
prune_points(to_remove AND prune_mask)
```

**Key Insight**: Unlike vanilla 3DGS which densifies ALL qualified Gaussians, Taming-3DGS uses importance sampling to stay within budget constraints.

---

### 2. Gaussian Importance Scoring

**Algorithm**: `compute_gaussian_score()` in `utils/taming_utils.py:40-91`

**Input**:
- `camlist`: Sampled training cameras (default: 10 cameras)
- `edge_losses`: Precomputed edge maps
- `importance_values`: Configuration dictionary with weights

**Steps**:

1. **For Each Camera View**:

   a. **Render Current State**:
   ```python
   render_image = render(camera, gaussians)
   ```

   b. **Compute Edge-Weighted Loss Map**:
   ```python
   l1_map = |render_image - gt_image|  # Per-pixel L1
   edge_map = edge_detector(gt_image)

   pixel_weights = (mse_importance * l1_map +
                   edge_importance * edge_map)
   ```

   c. **Render with Weighted Pixels**:
   ```python
   render_pkg = render(camera, gaussians, pixel_weights=pixel_weights)

   # Rasterizer accumulates weighted losses per Gaussian:
   loss_accum[g] = sum_{pixels p where g contributes}(
       weight[p] * alpha[g,p]
   )
   ```

   d. **Extract Per-Gaussian Metrics**:
   ```python
   depths = render_pkg["gaussian_depths"]
   radii = render_pkg["gaussian_radii"]
   dist_accum = render_pkg["accum_dist"]
   blend_weights = render_pkg["accum_blend"]
   render_counts = render_pkg["accum_count"]
   ```

   e. **Combine Gaussian-Level Importance**:
   ```python
   g_importance = (
       normalize(grad_importance, gradients) +
       normalize(opac_importance, opacities) +
       normalize(dept_importance, depths) +
       normalize(radii_importance, radii) +
       normalize(scale_importance, scales)
   )
   ```

   f. **Combine Pixel-Level Importance**:
   ```python
   p_importance = (
       normalize(dist_importance, dist_accum) +
       normalize(loss_importance, loss_accum) +
       normalize(count_importance, render_counts) +
       normalize(blend_importance, blend_weights)
   )
   ```

   g. **Aggregate**:
   ```python
   photometric_loss = compute_photometric_loss(camera, render_image)

   importance[camera][gaussian] = (
       view_importance * photometric_loss * (g_importance + p_importance)
   )
   ```

2. **Sum Across Views**:
```python
final_scores = sum(importance, axis=cameras)
```

**Output**: Per-Gaussian importance scores used for densification sampling.

---

### 3. Quadratic Budget Schedule

**Algorithm**: `get_count_array()` in `utils/taming_utils.py:100-117`

**Input**:
- `start_count`: Initial number of Gaussians (SfM points)
- `multiplier`: Budget multiplier or exact final count
- `mode`: "multiplier" or "final_count"

**Steps**:

1. **Determine Final Budget**:
```python
if mode == "multiplier":
    budget = start_count * multiplier
else:
    budget = multiplier  # Already exact count
```

2. **Compute Number of Densification Steps**:
```python
num_steps = (densify_until_iter - densify_from_iter) / densification_interval
# Example: (15000 - 500) / 500 = 29 steps
```

3. **Quadratic Growth Formula** (Equation 2 from paper):
```python
# Goal: Grow from start_count to budget in num_steps iterations
# Using parabola: count(x) = a*x² + b*x + c

# Constraints:
# - count(0) = start_count
# - count(num_steps) = budget
# - Minimum slope >= (budget - start_count) / num_steps

slope_lower_bound = (budget - start_count) / num_steps
k = 2 * slope_lower_bound  # Slope at x=0

# Solve for coefficients
a = (budget - start_count - k*num_steps) / (num_steps²)
b = k
c = start_count

# Generate schedule
counts = [a*x² + b*x + c for x in range(num_steps)]
```

**Rationale**: Quadratic growth allows faster densification early (when Gaussians are few) and slower growth later (when approaching budget).

**Example Output**:
```python
start_count = 100000
budget = 1500000
num_steps = 29

counts = [100000, 145172, 193448, 244828, 299310, ..., 1500000]
```

---

### 4. Memory-Efficient k-NN Initialization

**Algorithm**: `create_from_pcd()` in `scene/gaussian_model.py:154-208`

**Input**:
- `pcd`: Point cloud with positions and colors

**Steps**:

1. **Try Full k-NN Computation**:
```python
try:
    dist2 = distCUDA2(points)  # Compute 3-NN distances
except (RuntimeError, MemoryError):
    # Continue to fallback...
```

2. **Fallback: Batch Processing**:
```python
batch_size = 50000  # Process 50k points at a time
dist2_list = []

for i in range(0, num_points, batch_size):
    batch = points[i:i+batch_size]
    try:
        batch_dist2 = distCUDA2(batch)
        dist2_list.append(batch_dist2.cpu())
        torch.cuda.empty_cache()
    except (RuntimeError, MemoryError):
        # Ultimate fallback: fixed scale
        dist2 = torch.full((num_points,), 0.01)
        break

if dist2_list:
    dist2 = torch.cat(dist2_list).cuda()
```

3. **Initialize Scales**:
```python
# Scale = log(sqrt(distance_to_3rd_nearest_neighbor))
scales = log(sqrt(dist2)).repeat(1, 3)  # (N, 3)
```

**Rationale**: RTX 5080 has 16GB VRAM. Large scenes (>100k points) may OOM during k-NN. Batch processing keeps memory usage under control.

---

## File Reference Quick Guide

### Core Scripts

| File | Purpose | Key Functions | Used By |
|------|---------|---------------|---------|
| `train.py` | Training pipeline | `training()`, `training_report()` | User (CLI) |
| `render.py` | Render trained models | `render_sets()`, `render_set()` | User, full_eval.py |
| `metrics.py` | Compute metrics | `evaluate()`, `readImages()` | User, full_eval.py |
| `full_eval.py` | Automated eval | `run_cmd()` | User (CLI) |
| `convert.py` | COLMAP conversion | N/A (script) | User (CLI) |

### Scene Module

| File | Purpose | Key Classes | Key Functions |
|------|---------|-------------|---------------|
| `scene/__init__.py` | Scene management | `Scene` | `__init__()`, `save()`, `getTrainCameras()` |
| `scene/gaussian_model.py` | Gaussian representation | `GaussianModel` | `create_from_pcd()`, `densify_with_score()`, `training_setup()` |
| `scene/dataset_readers.py` | Data loading | `CameraInfo`, `SceneInfo` | `readColmapSceneInfo()`, `readNerfSyntheticInfo()` |
| `scene/cameras.py` | Camera class | `Camera` | `__init__()` |
| `scene/colmap_loader.py` | COLMAP parsing | N/A | `read_extrinsics_binary()`, `read_intrinsics_binary()` |

### Rendering Module

| File | Purpose | Key Functions | Returns |
|------|---------|---------------|---------|
| `gaussian_renderer/__init__.py` | Rendering engine | `render()` | RGB image + metadata |
| `gaussian_renderer/network_gui_ws.py` | Web viewer | `init()`, `handle_connection()` | N/A |

### Utilities

| File | Purpose | Key Functions | Used By |
|------|---------|---------------|---------|
| `utils/taming_utils.py` | Importance scoring | `compute_gaussian_score()`, `get_count_array()` | train.py |
| `utils/loss_utils.py` | Loss functions | `l1_loss()`, `ssim()` | train.py, render.py, metrics.py |
| `utils/camera_utils.py` | Camera conversion | `cameraList_from_camInfos()` | scene/__init__.py |
| `utils/graphics_utils.py` | Graphics math | `getWorld2View2()`, `getProjectionMatrix()` | scene/cameras.py |
| `utils/sh_utils.py` | Spherical harmonics | `RGB2SH()`, `eval_sh()` | scene/gaussian_model.py |
| `utils/general_utils.py` | General utilities | `inverse_sigmoid()`, `build_rotation()` | scene/gaussian_model.py |
| `utils/system_utils.py` | System utilities | `searchForMaxIteration()`, `mkdir_p()` | scene/__init__.py |
| `utils/image_utils.py` | Image utilities | `psnr()` | metrics.py |

### Arguments

| File | Purpose | Key Classes | Parameters |
|------|---------|-------------|------------|
| `arguments/__init__.py` | Configuration | `ModelParams`, `PipelineParams`, `OptimizationParams` | All training/rendering params |

### LPIPS Module

| File | Purpose | Key Functions | Used By |
|------|---------|---------------|---------|
| `lpipsPyTorch/modules/lpips.py` | LPIPS metric | `lpips()` | train.py, metrics.py |
| `lpipsPyTorch/modules/networks.py` | VGG network | VGG feature extraction | lpips.py |

---

## Glossary

**3DGS**: 3D Gaussian Splatting - Represents scenes as 3D Gaussian primitives

**Densification**: Process of adding new Gaussians (via cloning or splitting) to improve reconstruction

**Clone**: Duplicate a Gaussian at the same position (for small, high-gradient Gaussians)

**Split**: Divide a Gaussian into N child Gaussians (for large, high-gradient Gaussians)

**Pruning**: Remove Gaussians with low opacity or excessive size

**SfM**: Structure from Motion - COLMAP's 3D reconstruction used for initialization

**SH**: Spherical Harmonics - Encodes view-dependent color in a compact form

**Taming**: Budget-constrained densification using importance sampling

**Budget Mode**: Train with multiplier (e.g., 15× SfM points)

**Big Mode**: Train with exact final count (e.g., 5,987,095 Gaussians)

**data_device**: Where training images are stored (CPU or CUDA) - critical for memory management

**Importance Scoring**: Computes per-Gaussian scores for prioritized densification

**SSIM**: Structural Similarity Index Measure - perceptual image quality metric

**PSNR**: Peak Signal-to-Noise Ratio - pixel-wise accuracy metric

**LPIPS**: Learned Perceptual Image Patch Similarity - deep learning-based perceptual metric

---

## Recommended Reading Order

For understanding the codebase, read in this order:

1. **Start Here**:
   - `arguments/__init__.py` - Understand all parameters
   - `scene/gaussian_model.py` - Core data structure

2. **Data Loading**:
   - `scene/dataset_readers.py` - How data is loaded
   - `scene/__init__.py` - Scene initialization

3. **Rendering**:
   - `gaussian_renderer/__init__.py` - How rendering works

4. **Training**:
   - `train.py` - Main training loop
   - `utils/taming_utils.py` - Importance scoring algorithm

5. **Evaluation**:
   - `render.py` - Generate test images
   - `metrics.py` - Compute quality metrics

6. **Utilities** (as needed):
   - `utils/graphics_utils.py` - Math functions
   - `utils/sh_utils.py` - Spherical harmonics

---

## Quick Reference: Common Operations

### Training a Scene
```bash
python train.py \
    -s data/bicycle \          # Source dataset
    -i images_4 \              # Use 4x downsampled images
    -m output/bicycle_budget \ # Output directory
    --budget 15 \              # 15x multiplier
    --mode multiplier \        # Budget mode
    --densification_interval 500 \
    --data_device cpu \        # Load images to CPU (saves VRAM)
    --eval                     # Create train/test split
```

### Rendering a Trained Model
```bash
python render.py \
    -m output/bicycle_budget \ # Model directory
    --iteration 30000          # Which checkpoint (or -1 for latest)
```

### Computing Metrics
```bash
python metrics.py \
    -m output/bicycle_budget   # Model directory
```

### Full Evaluation Pipeline
```bash
python full_eval.py \
    --mipnerf360 data/ \
    --tanksandtemples data/ \
    --deepblending data/ \
    --mode budget              # or --mode big
```

---

## Troubleshooting Guide

### Common Issues

**1. CUDA Out of Memory (OOM)**
- **Cause**: GPU ran out of memory during training
- **Solution**: Use `--data_device cpu` to store images in RAM
- **Alternative**: Use lower resolution images (e.g., `images_8` instead of `images_4`)

**2. "cannot sample n_sample <= 0 samples"**
- **Cause**: Budget target is less than current Gaussian count
- **Fix**: Already patched in `scene/gaussian_model.py:554-562`
- **Prevention**: Ensure budget > initial SfM point count

**3. "TORCH_CUDA_ARCH_LIST not set" on RTX 5080/5090**
- **Cause**: Blackwell GPUs (sm_120) not supported by older PyTorch
- **Solution**: See CLAUDE.md RTX 5080/5090 setup instructions

**4. Black images in renders**
- **Cause**: Model not trained or loaded incorrectly
- **Check**: Verify `point_cloud.ply` exists in model directory
- **Debug**: Check if Gaussians have valid opacities (not all zero)

**5. Very low PSNR (<20 dB)**
- **Cause**: Insufficient training or budget too low
- **Solution**: Increase `--budget` or train for more iterations

---

**End of Documentation**
For questions or issues, refer to the [main README](../../README.md) or open an issue on GitHub.