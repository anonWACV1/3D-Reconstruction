






https://github.com/user-attachments/assets/daf7c11b-7cef-43bb-a7f4-6b000e38a9e1



[README.md](https://github.com/user-attachments/files/22426640/README.md)
# 3D Reconstruction



### ✨ Functionality

<details>
<summary>Click to expand functionality details</summary>

We have implemented interesting and useful functionalities:

1. **Flexible multi-camera training:** Choose any combination of cameras for training - single, multiple, or all. You can set these up by **SIMPLY** configuring your selection in the config file.

2. **Powered by gsplat** Integrated [gsplat](https://github.com/nerfstudio-project/gsplat) rasterization kernel with its advanced functions, e.g. absolute gradients, anti-aliasing, etc.

3. **Camera Pose Refinement:** Recognizing that camera poses may not always be sufficiently accurate, we provide a method to refine and optimize these poses.

4. **Objects' GT Bounding Box Refinement:** To address noise in ground truth boxes, we've added this feature to further improve accuracy and robustness.

5. **Affine Transformation:** This feature handles camera exposure and other related issues, enhancing the quality of scene reconstruction. 

6. **Scene Editing:** Capabilities for object deletion, replacement, position editing, and motion trajectory modification.

7. **Novel View Synthesis:** Generate views from arbitrary camera positions with configurable trajectories.

8. **SMPL Model Replacement:** Replace human models with different meshes and motion sequences.

9. **Cross-Scene Object Transfer:** Copy objects between different scenes, including rigid objects and SMPL models.

These functionalities are designed to enhance the overall performance and flexibility of our system, allowing for more accurate and adaptable scene reconstruction across various datasets and conditions.
</details>




## 📊 Prepare Data
We support most popular public driving datasets. Detailed instructions for downloading and processing each dataset are available in the following documents:

- Waymo: [Data Process Instruction](docs/Waymo.md)
- NuScenes: [Data Process Instruction](docs/NuScenes.md)
- NuPlan: [Data Process Instruction](docs/Nuplan.md)
- ArgoVerse: [Data Process Instruction](docs/ArgoVerse.md)
- PandaSet: [Data Process Instruction](docs/Pandaset.md)
- KITTI: [Data Process Instruction](docs/KITTI.md)



##Time and resource usage:

All experiments (3DGS training and rendering) were run on a workstation with a single NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM), Intel® Core™ i9-10920X CPU @ 3.50 GHz (24 cores), and 64 GB RAM, running CUDA 12.2 with NVIDIA driver 535.247.01.

Data were collected in *CARLA* at 15 FPS with five cameras at 375 × 1242 resolution. We use four 8-second scenarios (each with 120 frames per camera, 600 frames total). Peak GPU memory during training was 40 GB of 48 GB for all scenarios. Per-scenario training times were 00:46:47, 00:48:17, 00:47:32, and 00:41:53 (hh:mm:ss), averaging 46 min 07 s per 8-second scenario at the above settings. For rendering (image to video at 15 FPS), on average, a representative scenario took 6 s with <20 GB GPU VRAM and 7.4 GB RAM.

The measurements above illustrate two practical advantages of 3D Gaussian Splatting for AD/ADAS pipelines. First, training cost is tractable on a single machine: sub-hour per scenario with one high-VRAM GPU (no cluster or multi-GPU orchestration), which makes scaling across many short scenarios a straightforward batch process and enables rapid re-training after edits. Second, inference and rendering are fast: 21 FPS at the stated resolution with modest GPU utilization and <20 GB VRAM, leaving significant headroom for concurrent processes (e.g., perception-in-the-loop evaluation, multi-view rendering, or on-the-fly scenario edits). Together, these properties translate to high throughput and low operational overhead, aligning with industrial needs to generate and re-render large volumes of camera trajectories for development, V&V, and closed-loop testing.




## 🔧 Preprocessing

### Standard Dataset Preprocessing
```shell
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/custom_kitti/raw \
    --dataset custom_kitti \
    --split 2025_02_20 \
    --split_file data/custom_kitti_example_scenes.txt \
    --target_dir data/custom_kitti/processed \
    --workers 8    \
    --process_keys images lidar pose calib dynamic_masks objects
```

### Multi-Camera Setup Processing
```shell
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/custom_kitti/raw \
    --dataset kitti_5cams \
    --split 2025_02_20 \
    --split_file data/custom_kitti_example_scenes.txt \
    --target_dir data/custom_kitti/processed \
    --workers 8    \
    --process_keys images lidar pose calib dynamic_masks objects
```

### SMPL Model Setup

If you encounter missing SMPL model issues, manually download and create symlinks:
```shell
mkdir -p ~/.cache/phalp/3D/models/smpl/
mkdir -p ~/.cache/4DHumans/data/smpl/
# Assuming your SMPL_NEUTRAL.pkl file is in ~/Downloads/
ln -s ~/Downloads/SMPL_NEUTRAL.pkl ~/.cache/phalp/3D/models/smpl/SMPL_NEUTRAL.pkl
ln -s ~/Downloads/SMPL_NEUTRAL.pkl ~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl
```

### Human Pose Processing
```shell
conda activate 4D-humans
export PYTHONPATH=$(pwd)
python datasets/tools/humanpose_process.py \
    --dataset kitti \
    --data_root data/kitti/processed \
    --split_file data/kitti_example_scenes.txt 
    # Optional: [--save_temp] [--verbose]
```

## 🚀 Training

### Standard Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0
end_timestep=-1
output_root=$PYTHONPATH/output
project=waymo_3cams
scene_idx=23

python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```


### Memory-Limited Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0 
end_timestep=99 # Use smaller value if memory constrained
output_root=$PYTHONPATH/output
project=waymo_3cams
scene_idx=788

python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams_788 \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

### KITTI Dataset Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0
end_timestep=30 # Use -1 for all frames
output_root=$PYTHONPATH/output
project=Kitti
scene_idx=2011_09_26_drive_0059_sync

python tools/train.py \
    --config_file configs/omnire-KITTI.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=Kitti/1cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

### Custom Dataset Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0
end_timestep=-1
output_root=$PYTHONPATH/output
project=Kitti
scene_idx=2011_09_26_drive_0099_sync

python tools/train.py \
    --config_file configs/omnire-KITTI.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=Kitti/5cams_99 \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

### Street Gaussians Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0
end_timestep=-1
output_root=$PYTHONPATH/output
project=streetgs
scene_idx=2011_09_26_drive_0096_sync

python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=Kitti/5cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

### Extended Camera Setup
```shell
export PYTHONPATH=$(pwd)
start_timestep=0
end_timestep=-1
output_root=$PYTHONPATH/output
project=waymo_5cams
scene_idx=788

python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/5cams_788 \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

## 📊 Evaluation

### Full Scene Evaluation
=======
- To run other methods, change `--config_file`. See `configs/` for more options.
- Specify dataset and number of cameras by setting `dataset`. Examples: `waymo/1cams`, `waymo/5cams`, `pandaset/6cams`, `argoverse/7cams`, etc.
  You can set up arbitrary camera combinations for each dataset. See `configs/datasets/` for custom configuration details.
- For over 3 cameras or 450+ images, we recommend using `omnire_extended_cam.yaml`. It works better in practice.
### Evaluation
```shell
python tools/eval.py --resume_from $ckpt_path
```

### Single Frame Evaluation
```shell
export PYTHONPATH=$(pwd)
python tools/eval_single.py --resume_from ./output/waymo_lite/dataset\=waymo/1cams/checkpoint_final.pth --image_idx 0
```

### View Point Cloud
```shell
python tools/view_pc.py \
    --scene "./data/custom_kitti/processed/2025_02_20_drive_0003_sync" \
    --frame 0
```

## 📦 Export Assets

### Export Point Cloud to PLY
```shell
export PYTHONPATH=$(pwd)
python tools/export_ply.py \
    --resume_from ./output/waymo_lite/dataset\=waymo/1cams/checkpoint_final.pth \
    --save_dir output/ply \
    --alpha_thresh 0.01
```

### Export Specific Model Instance
```shell
export PYTHONPATH=$(pwd)
python tools/export_ply.py \
    --resume_from ./output/waymo_lite/dataset\=waymo/1cams/checkpoint_final.pth \
    --save_dir output/ply \
    --model_name RigidNodes \
    --instance_id 0
```

## 🎬 Scene Editing

### Generate Novel Views
```shell
python tools/novel_view_generate.py \
    --resume_from output/Kitti/dataset=Kitti/1cams_edit/checkpoint_final.pth \
    --start_frame 0 \
    --end_frame 100 \
    --stride 5
```

### Edit SMPL Models
```shell
python tools/edit_smpl.py \
    --resume_from output/Kitti_Changed_Lidar/dataset=Kitti/5cams/checkpoint_final.pth \
    --instance_id 0
```

### Edit SMPL Single Frame
```shell
export PYTHONPATH=$(pwd)
python tools/edit_smpl_single.py \
    --resume_from output/Kitti_Changed_Lidar/dataset=Kitti/5cams/checkpoint_final.pth \
    --image_idx 0 --source_frame 10 --target_frame 0
```

### Insert/Transfer Objects
```shell
python tools/insert_object.py \
    --source_checkpoint /path/to/source/checkpoint \
    --target_checkpoint /path/to/target/checkpoint \
    --rigid_ids 1 2 3 \
    --smpl_ids 0 \
    --output_dir output_videos \
    --video_name inserted_objects \
    --fps 30
```

### Replace SMPL Human Models
```shell
export PYTHONPATH=$(pwd)
python scene_editing/replace_smpl.py \
    --resume_from output/Kitti/dataset=Kitti/1cams_edit/checkpoint_final.pth \
    --instance_id 1 \
    --new_npz_path ./ply/sample1.npz \
    --new_ply_path ./ply/sample1.ply  \
    --smplx_path ./smpl_models \
    --keep_translation \
    --keep_global_rot
```

### Insert SMPL Copy
```shell
python scene_editing/insert_smpl_copy.py \
    --resume_from output/Kitti/dataset=Kitti/1cams_edit/checkpoint_final.pth \
    --instance_id 1 \
    --new_npz_path ./ply/sample1.npz \
    --new_ply_path ./ply/sample1.ply 
```

### Replace with GART Model
```shell
python scene_editing/replace_gart.py \
    --resume_from output/Kitti/dataset=Kitti/1cams_edit/checkpoint_final.pth \
    --gart_model_dir /home/yzhan536/thesis/src/3D-Reconstruction/GART_DATA/frog/ \
    --instance_id 1
```

### Replace SMPL Pose
```shell
export PYTHONPATH=$(pwd)                      
python scene_editing/replace_smpl_pose.py \
    --resume_from output/Kitti/dataset=Kitti/1cams/checkpoint_final.pth \
    --instance_id 1 \
    --new_pose_path $(pwd)/motion/11_out.npy \
    --keep_translation \
    --keep_global_rot
```

### Combined GART and Pose Replacement
```shell
export PYTHONPATH=$(pwd)
python scene_editing/replace_combined.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --gart_model_dir $(pwd)/GART_DATA/skywalker/ \
    --new_pose_path $(pwd)/motion/47_out.npy \
    --instance_id 0 \   
    --keep_translation \
    --keep_global_rot
```

### Multi-Instance Replacement
```shell
OUTPUT_DIR="./multi_instance_output"
CHECKPOINT_PATH="output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth"
python scene_editing/multi_instance_replace.py \
    --resume_from $CHECKPOINT_PATH \
    --config_file ./configs/multi_instance_replace.yaml \
    --output_dir $OUTPUT_DIR
```
