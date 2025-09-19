
```bash
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/custom_kitti/raw \
    --dataset kitti_5cams \
    --split 2025_02_20 \
    --split_file data/custom_kitti_example_scenes.txt \
    --target_dir data/kitti/processed \
    --workers 8    \
    --process_keys images lidar pose calib dynamic_masks objects
```




export PYTHONPATH=$(pwd):/home/yzhan536/anaconda3/envs/4D-humans/lib/python3.10/site-packages
python datasets/tools/humanpose_process.py --dataset kitti --data_root data/kitti/processed --split_file data/kitti_example_scenes.txt

export PYTHONPATH=$(pwd):/home/yzhan536/anaconda3/envs/segformer/lib/python3.8/site-packages

```bash
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root=$PYTHONPATH/output
project=NuScenes
scene_idx=000 

python tools/train.py \
    --config_file configs/omnire_nuscenes.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=NuScenes/000   \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

```bash
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root=$PYTHONPATH/output
project=pvg
scene_idx=training_20250628_171319_FollowLeadingVehicle_1 

python tools/train.py \
    --config_file configs/pvg.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1   \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

```bash
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root=$PYTHONPATH/output
project=deformablegs
scene_idx=training_20250628_171319_FollowLeadingVehicle_1 

python tools/train.py \
    --config_file configs/deformablegs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1   \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

```bash
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root=$PYTHONPATH/output
project=streetgs
scene_idx=training_20250628_171319_FollowLeadingVehicle_1 

python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1   \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```
```bash
export PYTHONPATH=$(pwd)
start_timestep=0
end_timestep=-1
output_root=$PYTHONPATH/output
project=waymo_new
scene_idx=542

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
```bash
python tools/novel_view_generate.py \
    --resume_from /home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250416_143028_FollowLeadingVehicleWithObstacle \
    --output_dir /home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250416_143028_FollowLeadingVehicleWithObstacle \
    --start_frame 0 \
    --end_frame 100 \
    --stride 5
```

```
python tools/novel_view_generate.py \
    --resume_from output/Kitti/dataset=Kitti/training_20250416_143028_FollowLeadingVehicleWithObstacle/checkpoint_final.pth \
    --start_frame 0 \
    --end_frame 100 \
    --stride 5
```

python tools/novel_view_generate.py \

  --resume_from output/Kitti/dataset=Kitti/training_20250416_143028_FollowLeadingVehicleWithObstacle/checkpoint_final.pth \

  --start_frame 0 \

  --end_frame 100 \

  --stride 5 \

  --save_video \

  --save_image False



```
python tools/eval.py --resume_from output/Kitti/dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1/checkpoint_final.pth
```

```
python tools/eval.py --resume_from output/waymo_new/dataset=waymo/3cams/checkpoint_final.pth
```


OUTPUT_DIR="./multi_instance_output"
CHECKPOINT_PATH="output/NuScenes/dataset=NuScenes/002/checkpoint_final.pth"
python scene_editing/multi_instance_replace.py \
    --resume_from $CHECKPOINT_PATH \
    --config_file ./configs/multi_instance_replace.yaml \
    --output_dir $OUTPUT_DIR


python tools/quick_pose_visualization.py output/Kitti/dataset=Kitti/training_20250630_162211_FollowLeadingVehicleWithObstacle_1/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz --output result.png

python tools/quick_pose_visualization.py output/Kitti/dataset=Kitti/training_20250630_162211_FollowLeadingVehicleWithObstacle_1/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz --list 20 --save-poses ./pose_output

python tools/quick_pose_visualization.py output/Kitti/dataset=Kitti/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/camera_poses_eval/full_poses_2025-07-02_21-57-59.npz --list 20 --save-poses ./pose_output2

python tools/pose_visualization.py --npz_file output/Kitti/dataset=Kitti/training_20250630_162211_FollowLeadingVehicleWithObstacle_1/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz --camera_filter CAM_TOP --output_dir ./analysis


蓝车id：0
白车id: 11
python scene_editing/edit_rigid.py \
    --resume_from output/waymo_new/dataset=waymo/3cams/checkpoint_final.pth \
    --operation remove \
    --instance_ids 11 \
    --output_dir output/edited_rigid_output

python scene_editing/edit_rigid.py --resume_from output/waymo_new/dataset=waymo/3cams/checkpoint_final.pth --operation remove_and_offset --remove_ids 0 --offset_id 11 --offset -5.0 2.0 0.0

python scene_editing/replace_gart_with_clustering.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --gart_model_dir ./GART_DATA/people_2m_data \
    --instance_id 0 \
    --use_dense \
    --keep_translation \
    --keep_global_rot


python scene_editing/insert_instance.py \
    --resume_from output/NuScenes/dataset=NuScenes/000/checkpoint_final.pth \
    --instance_type smpl \
    --operation insert_and_transform \
    --instance_files ./saved_instances/smpl_instance_0.pkl \
    --rotation_sequence "0,0,1,-90;1,0,0,90" \
    --output_dir ./insert_output


1. 插入并变换实例:
   python scene_editing/insert_instance.py \
       --resume_from output/waymo_3cams/dataset=waymo/5cams_788/checkpoint_final.pth \
       --operation insert_and_transform \
       --instance_files  ./saved_instances/scene1_smpl_instance_0.pkl \
       --new_instance_ids 5 \
       --translation 10.0 0.5 0.0 \
       --output_dir ./insert_transform_output

python scene_editing/insert_instance.py \
    --resume_from output/waymo_new/dataset=waymo/3cams/checkpoint_final.pth \
    --operation insert_and_transform \
    --instance_files  ./saved_instances/smpl_instance_0.pkl \
    --translation 5.0 0.0 0.0 \
    --output_dir ./insert_transform_output


export PYTHONPATH=$(pwd)
python scene_editing/insert_instance.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams//checkpoint_final.pth \
    --operation save \
    --instance_type smpl \
    --instance_ids 0 1 3  \
    --save_dir ./saved_instances

python scene_editing/insert_instance.py \
    --resume_from output/Kitti/dataset=Kitti/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/checkpoint_final.pth \
    --operation insert \
    --instance_files  ./saved_instances/rigid_instance_0.pkl \
    --output_dir ./insert_output

python scene_editing/insert_instance.py \
    --resume_from output/Kitti/dataset=Kitti/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/checkpoint_final.pth \
    --operation insert \
    --instance_files  ./saved_instances/smpl_instance_0.pkl \
    --output_dir ./insert_output


export PYTHONPATH=$(pwd)
python scene_editing/insert_instance_with_trajectory.py \
    --resume_from output/Kitti/dataset=Kitti/background/checkpoint_final.pth \
    --instance_files ./saved_instances/smpl_instance_3.pkl \
    --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \
    --trajectory_instance_id 1 \
    --output_dir ./trajectory_output

export PYTHONPATH=$(pwd)
python scene_editing/insert_instance_with_trajectory.py \
    --resume_from output/NuScenes/dataset=NuScenes/000/checkpoint_final.pth \
    --instance_files ./saved_instances/smpl_instance_3.pkl \
    --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \
    --trajectory_instance_id 1 \
    --output_dir ./trajectory_output

export PYTHONPATH=$(pwd)
python scene_editing/insert_replace_instance.py \
    --resume_from output/NuScenes/dataset=NuScenes/002/checkpoint_final.pth \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --target_instance_id 0 \
    --start_frame 10     --output_dir ./replace_output

export PYTHONPATH=$(pwd)
python scene_editing/insert_replace_instance.py \
    --resume_from output/Kitti/dataset=Kitti/change_line_gt/checkpoint_final.pth \
    --instance_file ./saved_instances/smpl_instance_3.pkl \
    --target_instance_id 0 \
    --start_frame 10     --output_dir ./replace_output


export PYTHONPATH=$(pwd)
python scene_editing/insert_replace_instance.py \
    --resume_from output/Kitti/dataset=Kitti/original/checkpoint_final.pth \
    --instance_file ./saved_instances/rigid_instance_0.pkl \
    --target_instance_id 0 \
    --output_dir ./replace_output

 python scene_editing/insert_instance.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --operation insert_and_transform \
    --instance_files  ./saved_instances/smpl_instance_0.pkl \
    --translation 5.0 0.0 0.0 \
    --output_dir ./insert_transform_output


python trajectory_comparison_tool.py \
    --instance_file ./saved_instances/rigid_instance_0.pkl \
    --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \
    --trajectory_instance_id 0 \
    --output_dir ./trajectory_comparison



    python scene_editing/replace_gart.py \
    --resume_from output/NuScenes/dataset=NuScenes/002/checkpoint_final.pth\
    --gart_model_dir GART_DATA/male-3-casual_prof \
    --instance_id 0


python tools/combined_trajectory_viewer.py \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --camera_traj output/NuScenes/dataset=NuScenes/000/camera_poses/full_poses_2025-07-14_14-34-28.npz \
    --camera_name CAM_FRONT \
    --output_dir ./combined_visualization \
    --output_name instance_vs_camera \
    --animate

python tools/quick_pose_visualization.py output/NuScenes/dataset=NuScenes/000/camera_poses/full_poses_2025-07-14_14-34-28.npz  --output result.png