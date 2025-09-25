# SplaTAM via RealSense Depth Camera (D435i)(Offline Capture)  
**This repo adds offline RGB-D capture from an Intel RealSense D435i and runs mapping with a fork of SplaTAM using RealSense-friendly configs.**  
## Record once (RGB, Depth, Intrinsics) → run SplaTAM later.    

- Dataset is saved in a TUM-style layout (color/, depth/, intrinsic/, timestamps.txt).    
- RealSense configs live in my SplaTAM fork, not only here.  
   └─(So a recursive clone will be required to obtain the submodule)  

# Repo layout

## *If you cloned this repo (wrapper) and also cloned my SplaTAM fork inside it:*

**SplaTAM_VIA_Depth-Camera/**    
**├─  data/**                               *your recorded sequences (gitignored)*  
 │      
 │...*Note: While developing this pipeline modification I was using remote view to access my GPU at home while on my laptop*    
 │...*I decided to leave the data in as it serves as a good example of a sample data collection from the script and*  
 │...*shows that you may want to fine turn the config settings such as depth scale as not every camera will be flawless*  
 │...*in mathematical conversion there will be some float error to address*  
 │  
**├─ tools/**  
 │..**└─ realsense/**  
 │.....**└─** capture_realsense_d435i.py    # recorder (RGB + aligned depth + intrinsics)  
 │  
 │..... **Note:** *Feel free to add or modify the script for a different camera model most work very similar and record the same data*  
 │..... **However** *you will need to install the SDK of said camera + viewer*  
 │..... *if you haven't aready for this instance the realsense SDK is required*  
 │  
**├─ third_party/**  
 │.....**└─ SplaTAM/**     # my FORK of SplaTAM (contains RealSense configs) + yaml  
 │...........**├─ configs/**  
 │........... **│...├─ data/RealSense/** realsense_generic.yaml  
 │........... **│...└─ realsense/** splatam_realsense.py  
 │........... **└─ submodules/**      
 │................**└─ diff-gaussian-rasterization-w-depth/** *- rasterizer (CUDA/C++ extension) Original SplaTAM repository forks/links*  
 │  
 │...**Note:** *Currently working on getting AMD support up. There are modifications of the origional repo using ROCM*  
 │...*some tuning for working here required though*  
 │  
 └─ **README.md**  


### *If you’re working directly inside your SplaTAM fork (no wrapper), keep tools/realsense/ in the fork and the dataset under <fork root>/data/…. Paths below still apply—just drop the third_party/SplaTAM prefix.*

## 0) Prereqs (Windows)  
- Python 3.10/3.11/3.12    
- (For GPU)  
   - NVIDIA GPU + recent driver  
   - CUDA Toolkit 12.1 (for NVCC)  
   - VS Build Tools (Desktop C++)  
- Intel RealSense SDK (librealsense) and firmware (use RealSense Viewer)  

## 1) Create a venv  
python -m venv .venv  
& .\.venv\Scripts\Activate.ps1  
python -m pip install --upgrade pip  

## 2) Install packages  
**A) RealSense capture (only needed on a machine that records)**  
pip install numpy==1.26.4 opencv-python==4.11.0.86 imageio==2.37.0 pillow pyrealsense2==2.56.5.9235  
  
**B) SplaTAM (GPU recommended)**  
Pick one Torch option:  

**CUDA (recommended):**  

- pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 `
  --index-url https://download.pytorch.org/whl/cu121
  
**CPU (will NOT support the rasterizer; mapping is very slow and limited):**  
  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
  
**Common deps used by the fork:**
  
pip install "matplotlib<3.9" pyyaml tqdm plyfile open3d==0.18.0 `  
           natsort==8.4.0 kornia==0.7.2 transforms3d scikit-image einops omegaconf yacs easydict
  
**optional if you keep wandb enabled anywhere**
pip install wandb

## 3) Build the rasterizer (GPU only)  
**from repo root:**  
cd third_party\SplaTAM\submodules\diff-gaussian-rasterization-w-depth  
pip install -v .  
cd ..\..\..  
  
### Sanity checks:
  
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization as dgr; print('dgr OK')"  
  
  
**If CUDA/Build Tools aren’t set up, the build will fail. Install CUDA 12.1 + VS C++ tools, reopen a new terminal, and retry.**  
**If you get an SM/arch error, set your GPU arch:**  
**setx TORCH_CUDA_ARCH_LIST "8.6" (example for RTX 30xx) → reopen terminal → rebuild.**  
  
## 4) Record a RealSense sequence (offline)  
  
Plug the D435i via USB 3.x. Then:  
  
**from repo root**  
python tools\realsense\capture_realsense_d435i.py --out .\data\my_sequence --fps 15 --duration 15 --width 640 --height 480 --visualize  

*You’ll get:*  
  
data/my_sequence/  
  color/0.jpg ...  
  depth/0.png ...        # 16-bit, millimeters  
  intrinsic/camera.txt   # "fx fy cx cy width height"  
  timestamps.txt  
  
## 5) Point SplaTAM to your dataset (configs live in my SplaTAM fork)  
  
**Edit these two files inside the fork as required for your camera:** 
  
third_party/SplaTAM/configs/data/RealSense/realsense_generic.yaml  
  
dataset_name: TUM_RGBD_like  
rgb_dirname: color  
depth_dirname: depth  
rgb_ext: .jpg  
depth_ext: .png  
associate_by: index  
depth_scale: 1000.0  
intrinsics:  
  from_file: true  
  path: intrinsic/camera.txt  
  
  
third_party/SplaTAM/configs/realsense/splatam_realsense.py  
  
primary_device = "cuda:0"      # set "cpu" if you really must  
use_wandb = False              # avoid wandb dependency  
  
scene_name = "my_sequence"     # folder name under .../data  
config = dict(  
    workdir="./experiments/RealSense",  
    run_name=f"{scene_name}_seed0",  
    primary_device=primary_device,  
    data=dict(  
        basedir=r"<ABSOLUTE_PATH_TO_REPO>\data",  
        gradslam_data_cfg=r"./configs/data/RealSense/realsense_generic.yaml",  
        sequence=scene_name,  
        desired_image_height=480,  
        desired_image_width=640,  
        start=0, end=-1, stride=1, num_frames=-1,  
    ),  
    # …tracking/mapping/viz blocks unchanged (or use my lighter CPU settings)  
)  
  
**Tip: Use an absolute Windows path for basedir (r"C:\path\to\repo\data"), and keep width/height matching your recording.**  
  
## 6) Run SplaTAM, visualize, export  
cd third_party\SplaTAM  
  
**Mapping**  
python scripts\splatam.py configs\realsense\splatam_realsense.py  
  
**Visualize**  
python viz_scripts\final_recon.py configs\realsense\splatam_realsense.py  
  
**Eexport ply**  
python scripts\export_ply.py configs\realsense\splatam_realsense.py  
  
**Outputs land under:**  
  
third_party/SplaTAM/experiments/RealSense/<run_name>/  
  
## Troubleshooting (real-world fixes)  
  
**No device connected during capture**    
Close RealSense Viewer/Zoom/Camera app; use USB 3.0 port; check rs-enumerate-devices.  
   
**ModuleNotFoundError: wandb when running scripts**  
The scripts/splatam.py driver sometimes imports wandb unconditionally. Either:  
  
pip install wandb, or  
  
edit scripts/splatam.py → wrap import in try/except and guard wandb.init(...), or  
  
make sure you run your realsense config with use_wandb=False.  
  
**Missing deps (kornia, natsort, …)**  
pip install kornia==0.7.2 natsort==8.4.0 (others listed above).  
  
**diff_gaussian_rasterization import/build errors**  
You need GPU torch, CUDA toolkit, and VS C++ build tools. Reopen a fresh terminal after installing.  
Check: nvcc --version, cl, and python -c "import torch; print(torch.version.cuda)".  
  
**torch.cuda.is_available() == False**  
Install/update NVIDIA driver + CUDA 12.1, reinstall torch cu121 wheels, reboot or open a new shell.  
  
**Wrong interpreter / mixed envs**  
Make sure only .venv is active. VS Code → Python: Select Interpreter → choose .venv\Scripts\python.exe.  
  
**Reproduce files to commit (recommended)**  
  
requirements-recording.txt  
  
numpy==1.26.4  
opencv-python==4.11.0.86  
imageio==2.37.0  
pillow  
pyrealsense2==2.56.5.9235  
  
requirements-splatam-gpu.txt  
  
**torch suite installed from CUDA wheel index:**  
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121  
  
numpy==1.26.4  
matplotlib<3.9  
opencv-python==4.11.0.86  
imageio  
pillow  
pyyaml  
tqdm  
plyfile  
open3d==0.18.0  
natsort==8.4.0  
kornia==0.7.2  
transforms3d  
scikit-image  
einops  
omegaconf  
yacs  
easydict  
  
**.gitignore**  
  
.venv/  
__pycache__/  
data/  
*.bag  
experiments/  
*.egg-info/  
dist/  
build/  
  
# Credit  
  
**Original SplaTAM authors** (see fork for full credits)  
**3DGS Diff Rasterizer** - https://github.com/graphdeco-inria/gaussian-splatting & its authors.  
**RealSense capture + RealSense configs:** Nolan Bryson (me)(this repo)  
