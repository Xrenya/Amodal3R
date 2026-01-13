<h1 align="center">Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images</h1>
  <p align="center">
    <a href="https://sm0kywu.github.io/CV/CV.html">Tianhao Wu</a>
    ·
    <a href="https://chuanxiaz.com/">Chuanxia Zheng</a>
    ·
    <a href="https://www.singaporetech.edu.sg/directory/faculty/frank-guan">Frank Guan</a>
    .
    <a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a>
    .
    <a href="https://personal.ntu.edu.sg/astjcham/index.html">Tat-Jen Cham</a>

  </p>
  <h3 align="center">ICCV 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2503.13439">Paper</a> | <a href="https://sm0kywu.github.io/Amodal3R/">Project Page</a> | <a href="https://huggingface.co/Sm0kyWu/Amodal3R">Pretrain Weight</a> | <a href="https://huggingface.co/spaces/Sm0kyWu/Amodal3R">Demo</a></h3>
  <div align="center"></div>
</p>

### Demo Video
<div align="center">

![Demo Video](asset/teaser.gif)

</div>

### Setup
This code has been tested on Ubuntu 22.02 with torch 2.4.0 & CUDA 11.8. We sincerely thank [TRELLIS](https://github.com/Microsoft/TRELLIS) for providing the environment setup and follow exactly as their instruction in this work.

Create a new conda environment named `amodal3r` and install the dependencies:
```sh
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```
The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
```sh
Usage: setup.sh [OPTIONS]
Options:
    -h, --help              Display this help message
    --new-env               Create a new conda environment
    --basic                 Install basic dependencies
    --train                 Install training dependencies
    --xformers              Install xformers
    --flash-attn            Install flash-attn
    --diffoctreerast        Install diffoctreerast
    --vox2seq               Install vox2seq
    --spconv                Install spconv
    --mipgaussian           Install mip-splatting
    --kaolin                Install kaolin
    --nvdiffrast            Install nvdiffrast
    --demo                  Install all dependencies for demo
```

### Pretrained models
We have provided our pretrained weights of both sparse structure module and SLAT module on [HuggingFace](https://huggingface.co/Sm0kyWu/Amodal3R).

### Data Preprocessing

#### Training Data
We use three datasets for training: [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), and [HSSD](https://huggingface.co/datasets/hssd/hssd-models). To obtain the training data, please also refer to [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). **Thanks to them for the amazing work!!!**. 

When the data is ready, combine them and put under `./dataset/abo_3dfuture_hssd`. If you want to train on a single dataset, feel free to modify the dataloader. For training, rendering images, Sparse Structure and SLAT are required.

### Training

To train you own model, you can start either on [our weights](https://huggingface.co/Sm0kyWu/Amodal3R) or [TRELLIS original weights](https://huggingface.co/microsoft/TRELLIS-image-large/tree/main). Please download the weights and put them under `./ckpts`.

To train the sparse structure module with our designed mask-weighted cross-attention and occlusion-aware attention, please run:
```sh
. ./train_ss.sh
```
To train the sparse structure module with our designed mask-weighted cross-attention and occlusion-aware attention, please run:
```sh
. ./train_slat.sh
```
The output folder where the model will be saved can be changed by modifying `--vis` parameter in the script.


### Inference
We have prepared examples under ./example folder. It supports both single and multiple image as input. For inference, please run:
```sh
python ./inference.py
```

If you want to try on you own data. You should prepare: 1) original image and 2) mask image (background is white (255,255,255), visible area is gray (188,188,188), occluded area is black (0,0,0)).

You can use [Segment Anything](https://github.com/facebookresearch/segment-anything) to obtain the corresponding mask, which is used for our in-the-wild examples in the paper and also in our demo.


### Evalutation
We render Toys4K and GSO exactly the same as training data. To obtain the evaluation dataset, please modify the directory in `3d_mask_render.py` and run:
```sh
python ./3d_mask_render.py
```
It will create a `renders_mask` folder with the 3D consistent mask in it.


### Usage Example (scaling the asset)

The code is extansion for usage 3DGS in scenes which was implementated in the paper: [R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation](https://research.zenseact.com/publications/R3D2/).

This code allow you to scale the initial 3DGS to the size of the real object in the 3DGS scenes so it would fit correctly. 
```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model with your default parameters
gaussian_model = Gaussian(
    aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
    sh_degree=3,                
    scaling_activation="softplus",   
    device=device
)

# Load the model
gaussian_model.load_ply("input_model.ply", transform=None)

# Scale to target size
target_size = [5.4310, 2.4373, 2.0099]
gaussian_model.scale_to_target_size(target_size, mode='fit', center_at_origin=True)

# ============ GET RODRIGUES VECTORS AFTER SCALING ============

# Method 1: Direct property access
rodrigues = gaussian_model.get_rodrigues  # (N, 3)
print(f"Rodrigues shape: {rodrigues.shape}")
print(f"Rodrigues range: [{rodrigues.min().item():.4f}, {rodrigues.max().item():.4f}]")

# Method 2: Get all parameters at once
params = gaussian_model.get_all_parameters(rotation_format='rodrigues')
xyz = params['xyz']           # (N, 3)
scales = params['scales']     # (N, 3)
rodrigues = params['rotation'] # (N, 3)
opacity = params['opacity']   # (N, 1)

print(f"XYZ shape: {xyz.shape}")
print(f"Scales shape: {scales.shape}")
print(f"Rodrigues shape: {rodrigues.shape}")
print(f"Opacity shape: {opacity.shape}")

# Method 3: Get different rotation formats
params_quat_wxyz = gaussian_model.get_all_parameters(rotation_format='quaternion_wxyz')
params_quat_xyzw = gaussian_model.get_all_parameters(rotation_format='quaternion_xyzw')
params_matrix = gaussian_model.get_all_parameters(rotation_format='matrix')

print(f"Quaternion (w,x,y,z) shape: {params_quat_wxyz['rotation'].shape}")  # (N, 4)
print(f"Quaternion (x,y,z,w) shape: {params_quat_xyzw['rotation'].shape}")  # (N, 4)
print(f"Rotation matrix shape: {params_matrix['rotation'].shape}")          # (N, 3, 3)

# Save scaled model
gaussian_model.save_ply("scaled_model.ply", transform=None)
```

#### Asset inseration and rotation (rotate.py)

Sometimes objects' positions in 4DGS scenes were generate not correctly and we need to adjust the angle of the object. This code will help to adjust angle in the scene (rotate 3DGS in the scene).  

##### Usage
```python
import torch

# Your saved data
data = {
    "params": {
        "means": torch.randn(1000, 3).cuda(),
        "scales": torch.randn(1000, 3).cuda(),
        "rotations": torch.randn(1000, 3).cuda() * 0.1,
        "rgbs": torch.rand(1000, 3).cuda(),
        "opacities": torch.zeros(1000, 1).cuda(),
    }
}

# ============ ROTATE USING EULER ANGLES ============
# Rotate 45 degrees around Y axis (yaw)
rotated_data = rotate_gaussians_euler(
    data, 
    yaw=45,      # rotation around Z (vertical)
    pitch=0,     # rotation around Y
    roll=0,      # rotation around X
    degrees=True,
    center=None  # rotates around centroid
)

# ============ ROTATE AROUND SPECIFIC AXIS ============
# Rotate 90 degrees around Y axis
rotated_data = rotate_gaussians_y(data, angle=90, degrees=True)

# Rotate 30 degrees around custom axis
rotated_data = rotate_gaussians_axis_angle(
    data, 
    axis=[1, 1, 0],  # diagonal axis
    angle=30,
    degrees=True
)

# ============ FULL TRANSFORMATION ============
# Scale, rotate, and translate
R = euler_to_rotation_matrix(yaw=45, pitch=0, roll=0)
transformed_data = transform_gaussians(
    data,
    rotation_matrix=R,
    translation=[1.0, 0.0, 0.0],  # move 1 unit in X
    scale=2.0,  # double the size
    center=None
)

# ============ CHAIN MULTIPLE ROTATIONS ============
# First rotate around X, then around Y
data_step1 = rotate_gaussians_x(data, angle=30)
data_step2 = rotate_gaussians_y(data_step1, angle=45)
```

##### Save and load
```python
gaussian_model = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=3, 
               scaling_activation="softplus", device='cuda')
gaussian_model.load_ply("original.ply", transform=None)

# Get original values
xyz_before = gs1.get_xyz.clone()
rot_before = gs1.get_rotation.clone()
scale_before = gs1.get_scaling.clone()

# Scaling to the target size
target_size = [5.4310, 2.4373, 2.0099]
# sometimes `mode='exact'` is more suitable if the object replacing the original but it might cause some distortion 
gaussian_model.scale_to_target_size(target_size, mode='fit', center_at_origin=True)

# Save and reload
gaussian_model.save_ply("test.ply", transform=None)
```

##### Full usage
```python
from general_utils import inverse_sigmoid

# Load your Gaussian model
gaussian_model = Gaussian(
    aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
    sh_degree=3,                
    scaling_activation="softplus",   
    device='cuda'
)
gaussian_model.load_ply("input_model.ply", transform=None)

# Scale to target size first
target_size = [5.4310, 2.4373, 2.0099]
# sometimes `mode='exact'` is more suitable if the object replacing the original but it might cause some distortion 
gaussian_model.scale_to_target_size(target_size, mode='fit', center_at_origin=True)

# Create data dict
gs = gaussian_model
data = {
    "params": {
        "means": gs.get_xyz.clone().to(torch.float32),
        "scales": torch.log(gs.get_scaling).clone().to(torch.float32),
        "rotations": gs.get_rodrigues.clone().to(torch.float32),
        "rgbs": gs._features_dc.squeeze(1).clone().to(torch.float32),
        "opacities": inverse_sigmoid(gs.get_opacity).clone().to(torch.float32),
    }
}

# Rotate the data (e.g., 90 degrees around Y axis)
rotated_data = rotate_gaussians_y(data, angle=90, degrees=True)

# Or use Euler angles for complex rotations
rotated_data = rotate_gaussians_euler(
    data,
    yaw=45,    # around Z
    pitch=30,  # around Y  
    roll=0,    # around X
    degrees=True
)

# Access rotated parameters
rotated_means = rotated_data['params']['means']
rotated_scales = rotated_data['params']['scales']
rotated_rotations = rotated_data['params']['rotations']

print(f"Means: {rotated_means.shape}")
print(f"Scales: {rotated_scales.shape}")
print(f"Rotations: {rotated_rotations.shape}")
```

