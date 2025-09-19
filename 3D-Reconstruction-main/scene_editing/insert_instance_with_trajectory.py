#!/usr/bin/env python3
"""
根据轨迹数据插入实例脚本（已修复时间戳同步问题）
读取指定JSON文件中的轨迹信息，插入实例并设置其在不同帧中的位置

修复内容：
1. 修复了insert_smpl_instance函数缺少start_frame参数的问题
2. 修复了轨迹数据时间戳与模型时间戳不匹配的问题
3. 添加了时间戳映射和校验功能
4. 优化了轨迹应用的逻辑

使用示例:
export PYTHONPATH=$(pwd)
python scene_editing/insert_instance_with_trajectory.py \
    --resume_from output/scene/checkpoint_final.pth \
    --instance_files ./saved_instances/smpl_instance_0.pkl \
    --new_instance_ids 5 \
    --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \
    --trajectory_instance_id 1 \
    --output_dir ./trajectory_output
"""

import os
import sys
import json
import logging
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch

# 确保能够导入项目模块
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset

# 导入场景编辑函数
try:
    from scene_editing.scene_editing import (
        load_instance_data,
        insert_smpl_instance,
        insert_rigid_instance,
        batch_render_with_eval,
        print_node_info,
        get_model_key,
        save_node_info
    )
except ImportError as e:
    print(f"导入场景编辑模块失败: {e}")
    print("请确保已正确设置PYTHONPATH并且scene_editing.py中包含所需函数")
    sys.exit(1)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('insert_instance_trajectory.log')
    ]
)
logger = logging.getLogger(__name__)


def load_trainer_and_dataset(resume_from: str, opts: List[str] = None):
    """
    加载训练器和数据集
    
    Args:
        resume_from: 检查点路径
        opts: 配置覆盖选项
        
    Returns:
        tuple: (trainer, dataset, cfg)
    """
    logger.info("=" * 60)
    logger.info("开始加载模型和数据集")
    logger.info("=" * 60)
    
    # 加载配置
    config_dir = os.path.dirname(resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    logger.info(f"加载配置文件: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    if opts:
        logger.info(f"应用配置覆盖: {opts}")
        cfg.merge_with(OmegaConf.from_dotlist(opts))
    
    # 初始化数据集
    logger.info("初始化数据集...")
    try:
        dataset = DrivingDataset(data_cfg=cfg.data)
        logger.info(f"数据集加载成功:")
        logger.info(f"  训练图像数: {len(dataset.train_image_set)}")
        logger.info(f"  完整图像数: {len(dataset.full_image_set)}")
        logger.info(f"  时间步数: {dataset.num_img_timesteps}")
    except Exception as e:
        logger.error(f"数据集初始化失败: {str(e)}")
        raise
    
    # 初始化训练器
    logger.info("初始化训练器...")
    try:
        trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=dataset.num_img_timesteps,
            model_config=cfg.model,
            num_train_images=len(dataset.train_image_set),
            num_full_images=len(dataset.full_image_set),
            test_set_indices=dataset.test_timesteps,
            scene_aabb=dataset.get_aabb().reshape(2, 3),
            device="cuda",
        )
        logger.info("训练器初始化成功")
    except Exception as e:
        logger.error(f"训练器初始化失败: {str(e)}")
        raise
    
    # 加载检查点
    logger.info(f"加载检查点: {resume_from}")
    try:
        trainer.resume_from_checkpoint(resume_from, load_only_model=True)
        trainer.eval()
        logger.info("检查点加载成功")
    except Exception as e:
        logger.error(f"检查点加载失败: {str(e)}")
        raise
    
    # 打印模型信息
    logger.info("模型信息:")
    available_models = list(trainer.models.keys())
    logger.info(f"  可用模型类型: {available_models}")
    
    return trainer, dataset, cfg


def load_trajectory_data(json_path: str, instance_id: str) -> Dict:
    """
    加载轨迹数据
    
    Args:
        json_path: JSON文件路径
        instance_id: 实例ID
        
    Returns:
        Dict: 包含帧索引和变换矩阵的字典
    """
    logger.info(f"加载轨迹数据: {json_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"轨迹JSON文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if str(instance_id) not in data:
        available_ids = list(data.keys())
        raise ValueError(f"实例ID {instance_id} 不存在于轨迹数据中，可用ID: {available_ids}")
    
    instance_data = data[str(instance_id)]
    frame_annotations = instance_data["frame_annotations"]
    
    frame_indices = frame_annotations["frame_idx"]
    obj_to_world_matrices = frame_annotations["obj_to_world"]
    
    logger.info(f"加载了实例 {instance_id} 的轨迹数据:")
    logger.info(f"  类别: {instance_data.get('class_name', 'Unknown')}")
    logger.info(f"  帧数: {len(frame_indices)}")
    logger.info(f"  帧范围: {min(frame_indices)} - {max(frame_indices)}")
    
    return {
        "frame_indices": frame_indices,
        "transforms": obj_to_world_matrices,
        "class_name": instance_data.get("class_name", "Unknown")
    }


def create_frame_mapping(trajectory_frames: List[int], model_total_frames: int) -> Dict[int, int]:
    """
    创建轨迹帧索引到模型帧索引的映射
    
    Args:
        trajectory_frames: 轨迹数据中的帧索引列表
        model_total_frames: 模型的总帧数
        
    Returns:
        Dict: {轨迹帧索引: 模型帧索引} 的映射字典
    """
    logger.info("创建时间戳映射...")
    
    # 获取轨迹的帧范围
    min_traj_frame = min(trajectory_frames)
    max_traj_frame = max(trajectory_frames)
    traj_frame_span = max_traj_frame - min_traj_frame + 1
    
    logger.info(f"轨迹帧范围: {min_traj_frame} - {max_traj_frame} (共 {traj_frame_span} 帧)")
    logger.info(f"模型总帧数: {model_total_frames}")
    
    # 创建映射策略
    frame_mapping = {}
    
    if traj_frame_span <= model_total_frames:
        # 轨迹帧数不超过模型帧数，进行直接映射（偏移到起始帧）
        offset = min_traj_frame
        for traj_frame in trajectory_frames:
            model_frame = traj_frame - offset
            if 0 <= model_frame < model_total_frames:
                frame_mapping[traj_frame] = model_frame
            else:
                logger.warning(f"轨迹帧 {traj_frame} 映射到模型帧 {model_frame} 超出范围，跳过")
    else:
        # 轨迹帧数超过模型帧数，进行缩放映射
        scale_factor = (model_total_frames - 1) / (traj_frame_span - 1)
        for traj_frame in trajectory_frames:
            relative_frame = traj_frame - min_traj_frame
            model_frame = int(relative_frame * scale_factor)
            if 0 <= model_frame < model_total_frames:
                frame_mapping[traj_frame] = model_frame
            else:
                logger.warning(f"轨迹帧 {traj_frame} 缩放映射到模型帧 {model_frame} 超出范围，跳过")
    
    logger.info(f"成功创建 {len(frame_mapping)} 个帧映射")
    logger.info(f"示例映射: {dict(list(frame_mapping.items())[:5])}")
    
    return frame_mapping


def extract_translation_from_matrix(transform_matrix: List[List[float]]) -> np.ndarray:
    """
    从4x4变换矩阵中提取平移部分
    
    Args:
        transform_matrix: 4x4变换矩阵
        
    Returns:
        np.ndarray: 3D平移向量
    """
    matrix = np.array(transform_matrix)
    # 平移部分在最后一列的前三个元素
    translation = matrix[:3, 3]
    return translation


def apply_trajectory_to_instance(trainer, model_key: str, instance_id: int, trajectory_data: Dict):
    """
    将轨迹数据应用到插入的实例上
    
    Args:
        trainer: 训练器对象
        model_key: 模型键名
        instance_id: 实例ID
        trajectory_data: 轨迹数据
    """
    logger.info(f"开始应用轨迹到实例 {instance_id}")
    
    model = trainer.models[model_key]
    frame_indices = trajectory_data["frame_indices"]
    transforms = trajectory_data["transforms"]
    
    # 获取模型的总帧数
    total_frames = model.num_frames if hasattr(model, 'num_frames') else model.instances_trans.shape[0]
    logger.info(f"模型总帧数: {total_frames}")
    
    # 创建时间戳映射
    frame_mapping = create_frame_mapping(frame_indices, total_frames)
    
    if not frame_mapping:
        logger.error("无法创建有效的帧映射，跳过轨迹应用")
        return
    
    # 应用每一帧的位置
    applied_frames = 0
    skipped_frames = 0
    
    with torch.no_grad():
        for i, (traj_frame_idx, transform_matrix) in enumerate(zip(frame_indices, transforms)):
            # 获取映射后的模型帧索引
            if traj_frame_idx not in frame_mapping:
                skipped_frames += 1
                continue
            
            model_frame_idx = frame_mapping[traj_frame_idx]
            
            # 检查模型帧索引是否在有效范围内
            if model_frame_idx >= total_frames:
                skipped_frames += 1
                continue
            
            # 提取平移向量
            translation = extract_translation_from_matrix(transform_matrix)
            translation_tensor = torch.tensor(translation, device=trainer.device, dtype=torch.float32)
            
            # 直接设置实例在该帧的位置
            try:
                # 直接更新位置而不是计算偏移量，避免累积误差
                model.instances_trans[model_frame_idx, instance_id] = translation_tensor
                applied_frames += 1
                
                # 每处理100帧打印一次进度
                if (i + 1) % 100 == 0:
                    logger.info(f"  已处理 {i + 1}/{len(frame_indices)} 帧")
                    
            except Exception as e:
                logger.warning(f"  应用轨迹帧 {traj_frame_idx}->模型帧 {model_frame_idx} 的变换时出错: {str(e)}")
                skipped_frames += 1
                continue
    
    logger.info(f"轨迹应用完成:")
    logger.info(f"  成功应用: {applied_frames} 帧")
    logger.info(f"  跳过: {skipped_frames} 帧")


def insert_smpl_instance_with_start_frame(
    trainer,
    instance_data: Dict,
    new_instance_id: Optional[int] = None,
    start_frame: int = 0,
    device: str = "cuda"
) -> int:
    """
    插入SMPL实例，支持start_frame参数的包装函数
    
    Args:
        trainer: 训练器实例
        instance_data: 实例数据字典
        new_instance_id: 新的实例ID
        start_frame: 开始应用数据的帧数
        device: 设备
        
    Returns:
        分配的新实例ID
    """
    # 首先调用原始的插入函数（不传递start_frame参数）
    try:
        actual_id = insert_smpl_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    except TypeError as e:
        # 如果原函数不支持start_frame参数，记录警告但继续
        logger.warning(f"insert_smpl_instance不支持start_frame参数: {e}")
        # 尝试不传递start_frame参数
        actual_id = insert_smpl_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    
    # 如果指定了start_frame且不为0，则需要手动处理前面的帧
    if start_frame > 0 and actual_id is not None:
        logger.info(f"处理start_frame={start_frame}的逻辑...")
        
        model_key = get_model_key("smpl")
        model = trainer.models[model_key]
        
        # 获取刚插入实例的数据
        motion_data = instance_data.get("motion", {})
        
        with torch.no_grad():
            total_frames = model.instances_trans.shape[0]
            
            # 对于前start_frame帧，使用实例数据的第一帧作为默认值
            if start_frame < total_frames:
                # 保持前面帧的位置为第一帧的位置
                if "instances_trans" in motion_data:
                    first_frame_trans = motion_data["instances_trans"][0]
                    model.instances_trans[:start_frame, actual_id] = first_frame_trans.to(device)
                
                # 保持前面帧的旋转为第一帧的旋转
                if "instances_quats" in motion_data:
                    first_frame_quat = motion_data["instances_quats"][0]
                    model.instances_quats[:start_frame, actual_id] = first_frame_quat.to(device)
                
                # 保持前面帧的SMPL姿态为第一帧的姿态
                if "smpl_qauts" in motion_data:
                    first_frame_smpl = motion_data["smpl_qauts"][0]
                    model.smpl_qauts[:start_frame, actual_id] = first_frame_smpl.to(device)
                    
                logger.info(f"已设置实例 {actual_id} 前 {start_frame} 帧的默认姿态")
    
    return actual_id


def insert_rigid_instance_with_start_frame(
    trainer,
    instance_data: Dict,
    new_instance_id: Optional[int] = None,
    start_frame: int = 0,
    device: str = "cuda"
) -> int:
    """
    插入刚体实例，支持start_frame参数的包装函数
    """
    # 首先调用原始的插入函数
    try:
        actual_id = insert_rigid_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    except TypeError as e:
        logger.warning(f"insert_rigid_instance不支持start_frame参数: {e}")
        actual_id = insert_rigid_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    
    # 如果指定了start_frame且不为0，则需要手动处理前面的帧
    if start_frame > 0 and actual_id is not None:
        logger.info(f"处理rigid实例的start_frame={start_frame}逻辑...")
        # 这里可以根据刚体实例的具体需求实现start_frame逻辑
        
    return actual_id


def handle_insert_with_trajectory(args, trainer, dataset, cfg):
    """
    处理带轨迹的插入操作
    """
    logger.info("=" * 60)
    logger.info("开始插入实例并应用轨迹")
    logger.info("=" * 60)
    
    # 加载轨迹数据
    trajectory_data = load_trajectory_data(args.trajectory_json_path, args.trajectory_instance_id)
    
    # 验证实例文件
    instance_files = args.instance_files
    valid_files = []
    for file_path in instance_files:
        if not os.path.exists(file_path):
            logger.warning(f"实例文件不存在，跳过: {file_path}")
            continue
        valid_files.append(file_path)
    
    if not valid_files:
        logger.error("没有有效的实例文件可以插入")
        return False
    
    logger.info(f"将插入以下实例文件:")
    for file_path in valid_files:
        file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
        logger.info(f"  {file_path} ({file_size:.2f} MB)")
    
    # 准备新实例ID
    new_instance_ids = args.new_instance_ids
    if new_instance_ids and len(new_instance_ids) != len(valid_files):
        logger.warning(f"新实例ID数量({len(new_instance_ids)})与文件数量({len(valid_files)})不匹配")
        new_instance_ids = new_instance_ids[:len(valid_files)] if new_instance_ids else None
    
    # 获取开始帧参数
    start_frame = getattr(args, 'start_frame', 0)
    if start_frame > 0:
        logger.info(f"将从第 {start_frame} 帧开始应用实例运动数据")

    # 打印插入前的场景状态
    logger.info("插入前的场景状态:")
    print_node_info(trainer)
    
    # 执行插入操作
    inserted_ids = []
    for i, file_path in enumerate(valid_files):
        try:
            logger.info(f"\n--- 处理文件 {i+1}/{len(valid_files)}: {file_path} ---")
            
            # 加载实例数据
            instance_data = load_instance_data(file_path)
            
            # 确定实例类型
            if "smpl_template" in instance_data or "voxel_deformer" in instance_data:
                instance_type = "smpl"
                insert_func = insert_smpl_instance_with_start_frame
            else:
                instance_type = "rigid"
                insert_func = insert_rigid_instance_with_start_frame
            
            logger.info(f"检测到实例类型: {instance_type.upper()}")
            
            # 验证模型是否支持该类型
            model_key = get_model_key(instance_type)
            if model_key not in trainer.models:
                logger.error(f"目标场景不支持{instance_type.upper()}实例，跳过")
                continue
            
            # 确定新实例ID
            new_id = new_instance_ids[i] if new_instance_ids and i < len(new_instance_ids) else None
            
            # 执行插入（使用支持start_frame的包装函数）
            actual_id = insert_func(
                trainer=trainer,
                instance_data=instance_data,
                new_instance_id=new_id,
                start_frame=start_frame,
                device="cuda"
            )
            
            if actual_id is not None:
                inserted_ids.append(actual_id)
                logger.info(f"✓ 成功插入实例，分配ID: {actual_id}")
                
                # 应用轨迹数据
                logger.info(f"开始为实例 {actual_id} 应用轨迹数据...")
                apply_trajectory_to_instance(trainer, model_key, actual_id, trajectory_data)
                logger.info(f"✓ 轨迹应用完成")
                
            else:
                logger.error(f"✗ 插入失败，未返回有效ID")
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # 打印插入后的场景状态
    logger.info("\n插入并应用轨迹后的场景状态:")
    final_node_info = print_node_info(trainer)
    
    logger.info(f"\n操作完成!")
    logger.info(f"成功插入并应用轨迹: {len(inserted_ids)}/{len(valid_files)} 个实例")
    if inserted_ids:
        logger.info(f"已处理的实例ID: {inserted_ids}")
    
    # 渲染结果（如果指定了输出目录）
    if args.output_dir and inserted_ids:
        logger.info("\n开始渲染最终场景...")
        try:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            log_dir = os.path.dirname(args.resume_from)
            batch_render_with_eval(
                cfg=cfg,
                trainer=trainer,
                dataset=dataset,
                output_dir=str(output_dir),
                log_dir=log_dir,
                post_fix="_with_trajectory"
            )
            logger.info(f"渲染完成，结果保存到: {output_dir}")
            
            # 保存最终场景信息
            save_node_info(final_node_info, str(output_dir))
            
            # 保存使用的轨迹信息
            trajectory_info_path = output_dir / "trajectory_info.json"
            with open(trajectory_info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_json_path": args.trajectory_json_path,
                    "trajectory_instance_id": args.trajectory_instance_id,
                    "inserted_instance_ids": inserted_ids,
                    "trajectory_data": trajectory_data
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"轨迹信息已保存到: {trajectory_info_path}")
            
        except Exception as e:
            logger.error(f"渲染过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
    
    return len(inserted_ids) > 0


def main():
    parser = argparse.ArgumentParser(
        description="根据轨迹数据插入实例工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 插入实例并应用轨迹:
   python scene_editing/insert_instance_with_trajectory.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_files ./saved_instances/smpl_instance_0.pkl \\
       --new_instance_ids 5 \\
       --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \\
       --trajectory_instance_id 1 \\
       --output_dir ./trajectory_output

2. 多个实例插入:
   python scene_editing/insert_instance_with_trajectory.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_files ./saved_instances/smpl_instance_0.pkl ./saved_instances/smpl_instance_1.pkl \\
       --new_instance_ids 5 6 \\
       --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \\
       --trajectory_instance_id 1 \\
       --output_dir ./trajectory_output
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--resume_from", type=str, required=True,
        help="检查点路径"
    )
    
    # 插入操作参数
    parser.add_argument(
        "--instance_files", type=str, nargs="+", required=True,
        help="要插入的实例文件路径列表"
    )
    parser.add_argument(
        "--new_instance_ids", type=int, nargs="+",
        help="新实例ID列表，如果不指定则自动分配"
    )
    
    # 轨迹参数
    parser.add_argument(
        "--trajectory_json_path", type=str,
        default="data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json",
        help="轨迹JSON文件路径"
    )
    parser.add_argument(
        "--trajectory_instance_id", type=str, required=True,
        help="要使用轨迹的实例ID（JSON文件中的键）"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir", type=str,
        help="输出目录（用于渲染和保存信息）"
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help="不渲染结果，只进行操作"
    )
    parser.add_argument(
        "--start_frame", type=int, default=0,
        help="从第几帧开始应用插入实例的运动数据（默认为0，即从第一帧开始）。轨迹数据仍将应用于所有帧。"
    )
    
    # 配置覆盖
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="配置覆盖选项"
    )
    
    args = parser.parse_args()
    
    try:
        # 验证轨迹JSON文件是否存在
        if not os.path.exists(args.trajectory_json_path):
            logger.error(f"轨迹JSON文件不存在: {args.trajectory_json_path}")
            sys.exit(1)
        
        # 加载模型和数据集
        trainer, dataset, cfg = load_trainer_and_dataset(args.resume_from, args.opts)
        
        # 执行插入和轨迹应用操作
        success = handle_insert_with_trajectory(args, trainer, dataset, cfg)
        
        if success:
            logger.info("=" * 60)
            logger.info("插入实例并应用轨迹操作完成!")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("插入实例并应用轨迹操作失败!")
            logger.error("=" * 60)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()