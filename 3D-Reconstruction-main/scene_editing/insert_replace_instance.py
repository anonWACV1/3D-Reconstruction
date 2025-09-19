#!/usr/bin/env python3
"""
插入并替换实例脚本
插入新实例并替换场景中现有的实例，保留原实例的位置信息，使用新实例的旋转和动作

使用示例:
export PYTHONPATH=$(pwd)
python scene_editing/insert_replace_instance.py \
    --resume_from output/scene/checkpoint_final.pth \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --target_instance_id 2 \
    --output_dir ./replace_output

# 可选：指定新实例ID
python scene_editing/insert_replace_instance.py \
    --resume_from output/scene/checkpoint_final.pth \
    --instance_file ./saved_instances/smpl_instance_0.pkl \
    --target_instance_id 2 \
    --new_instance_id 5 \
    --output_dir ./replace_output
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from omegaconf import OmegaConf
import torch
import numpy as np

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
        batch_render_with_eval,
        print_node_info,
        save_node_info
    )

    # 尝试导入其他函数，如果不存在则定义替代函数
    try:
        from scene_editing.scene_editing import load_instance_data
    except ImportError:
        def load_instance_data(file_path):
            """加载实例数据的替代实现"""
            import pickle
            logger.info(f"使用内置函数加载实例数据: {file_path}")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    try:
        from scene_editing.scene_editing import get_model_key
    except ImportError:
        def get_model_key(instance_type):
            """获取模型键的替代实现"""
            if instance_type == "smpl":
                return "SMPLNodes"
            elif instance_type == "rigid":
                return "RigidNodes"
            else:
                return f"{instance_type.capitalize()}Nodes"
    
    try:
        from scene_editing.scene_editing import insert_smpl_instance, insert_rigid_instance
    except ImportError:
        def insert_smpl_instance(trainer, instance_data, new_instance_id, device):
            logger.error("insert_smpl_instance函数不可用")
            return None
        
        def insert_rigid_instance(trainer, instance_data, new_instance_id, device):
            logger.error("insert_rigid_instance函数不可用")
            return None

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('insert_replace_instance.log')
        ]
    )
    logger = logging.getLogger(__name__) 
    # 尝试导入replace_smpl_instance_improved
    try:
        from scene_editing.scene_editing import replace_smpl_instance_improved
        HAS_REPLACE_FUNCTION = True
        logger.info("成功导入replace_smpl_instance_improved函数")
    except ImportError:
        replace_smpl_instance_improved = None
        HAS_REPLACE_FUNCTION = False
        logger.warning("replace_smpl_instance_improved函数不存在，将使用替代方案")
        
except ImportError as e:
    print(f"导入场景编辑模块失败: {e}")
    print("请确保已正确设置PYTHONPATH并且scene_editing.py中包含所需函数")
    sys.exit(1)


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
    
    return trainer, dataset, cfg


def get_instance_ids_from_model(model):
    """从模型中获取所有实例ID"""
    if hasattr(model, 'point_ids'):
        return model.point_ids[..., 0].unique().cpu().numpy()
    else:
        return []


def print_instance_info(model, model_name):
    """打印实例详细信息"""
    all_ids = get_instance_ids_from_model(model)
    logger.info(f"{model_name} 实例信息:")
    logger.info(f"  总实例数: {len(all_ids)}")
    
    # 统计每个实例的点数
    if hasattr(model, 'point_ids'):
        for instance_id in all_ids:
            point_count = (model.point_ids[..., 0] == instance_id).sum().item()
            logger.info(f"    实例 {instance_id}: {point_count} 个点")


def backup_instance_transforms(model, instance_id: int) -> Dict[str, torch.Tensor]:
    """
    备份指定实例的变换信息（位置和旋转）
    
    Args:
        model: 模型对象
        instance_id: 实例ID
        
    Returns:
        Dict: 包含位置和旋转信息的字典
    """
    logger.info(f"备份实例 {instance_id} 的变换信息...")
    
    backup_data = {}
    
    # 备份位置信息
    if hasattr(model, 'instances_trans'):
        backup_data['translation'] = model.instances_trans[:, instance_id].clone()
        logger.info(f"  已备份位置信息，形状: {backup_data['translation'].shape}")
    
    # 备份旋转信息 - 检查不同的旋转属性名
    rotation_attrs = ['instances_rot', 'instances_quats', 'instances_quat']
    for attr in rotation_attrs:
        if hasattr(model, attr):
            backup_data['rotation'] = getattr(model, attr)[:, instance_id].clone()
            logger.info(f"  已备份旋转信息({attr})，形状: {backup_data['rotation'].shape}")
            break
    
    # 备份SMPL特有的旋转信息
    if hasattr(model, 'smpl_qauts'):
        backup_data['smpl_rotation'] = model.smpl_qauts[:, instance_id].clone()
        logger.info(f"  已备份SMPL旋转信息，形状: {backup_data['smpl_rotation'].shape}")
    
    # 备份缩放信息（如果存在）
    if hasattr(model, 'instances_scale'):
        backup_data['scale'] = model.instances_scale[:, instance_id].clone()
        logger.info(f"  已备份缩放信息，形状: {backup_data['scale'].shape}")
    
    logger.info(f"✓ 实例 {instance_id} 变换信息备份完成")
    return backup_data


def restore_instance_transforms(model, instance_id: int, backup_data: Dict[str, torch.Tensor], 
                              restore_translation: bool = True, restore_rotation: bool = False, 
                              restore_scale: bool = False):
    """
    恢复指定实例的变换信息
    
    Args:
        model: 模型对象
        instance_id: 实例ID
        backup_data: 备份的变换数据
        restore_translation: 是否恢复位置信息
        restore_rotation: 是否恢复旋转信息
        restore_scale: 是否恢复缩放信息
    """
    logger.info(f"恢复实例 {instance_id} 的变换信息...")
    
    with torch.no_grad():
        # 恢复位置信息
        if restore_translation and 'translation' in backup_data and hasattr(model, 'instances_trans'):
            model.instances_trans[:, instance_id] = backup_data['translation']
            logger.info(f"  ✓ 已恢复位置信息")
        
        # 恢复旋转信息 - 检查不同的旋转属性名
        if restore_rotation and 'rotation' in backup_data:
            rotation_attrs = ['instances_rot', 'instances_quats', 'instances_quat']
            for attr in rotation_attrs:
                if hasattr(model, attr):
                    getattr(model, attr)[:, instance_id] = backup_data['rotation']
                    logger.info(f"  ✓ 已恢复旋转信息({attr})")
                    break
        
        # 恢复SMPL旋转信息
        if restore_rotation and 'smpl_rotation' in backup_data and hasattr(model, 'smpl_qauts'):
            model.smpl_qauts[:, instance_id] = backup_data['smpl_rotation']
            logger.info(f"  ✓ 已恢复SMPL旋转信息")
        
        # 恢复缩放信息
        if restore_scale and 'scale' in backup_data and hasattr(model, 'instances_scale'):
            model.instances_scale[:, instance_id] = backup_data['scale']
            logger.info(f"  ✓ 已恢复缩放信息")
    
    logger.info(f"✓ 实例 {instance_id} 变换信息恢复完成")


def remove_instance_from_model(model, instance_id: int) -> bool:
    """
    从模型中移除指定实例
    
    Args:
        model: 模型对象
        instance_id: 要移除的实例ID
        
    Returns:
        bool: 是否成功移除
    """
    logger.info(f"从模型中移除实例 {instance_id}...")
    
    try:
        with torch.no_grad():
            if hasattr(model, 'point_ids'):
                # 获取该实例的点云掩码
                instance_mask = model.point_ids[..., 0] == instance_id
                num_points = instance_mask.sum().item()
                
                if num_points > 0:
                    # 方法1: 将实例ID设置为无效值
                    invalid_id = -1
                    model.point_ids[instance_mask, 0] = invalid_id
                    logger.info(f"✓ 已将实例 {instance_id} 的 {num_points} 个点设置为无效ID")
                    
                    # 方法2: 可选择性地移除点云数据（如果模型支持）
                    if hasattr(model, '_remove_points_by_mask'):
                        model._remove_points_by_mask(~instance_mask)
                        logger.info(f"✓ 已从模型中物理移除实例 {instance_id} 的点云")
                    
                    return True
                else:
                    logger.warning(f"实例 {instance_id} 没有关联的点云数据")
                    return False
            else:
                logger.warning("模型没有point_ids属性，无法移除实例")
                return False
                
    except Exception as e:
        logger.error(f"移除实例 {instance_id} 时出错: {str(e)}")
        return False


def safe_compare_instance_info(smpl_nodes, instance_id, new_instance):
    """
    安全版本的实例信息比较函数，避免空列表错误
    """
    try:
        logger.info(f"=== 实例 {instance_id} 替换前后对比 ===")
        
        # 获取原始实例信息
        points_per_instance = smpl_nodes.smpl_points_num
        start_idx = instance_id * points_per_instance
        end_idx = (instance_id + 1) * points_per_instance
        
        # 基本信息输出
        logger.info(f"实例ID: {instance_id}")
        logger.info(f"点云范围: {start_idx} - {end_idx}")
        logger.info(f"每个实例点数: {points_per_instance}")
        
        # 检查新实例数据的关键属性
        if "pts" in new_instance:
            logger.info(f"新实例点云形状: {new_instance['pts'].shape}")
        if "global_orient" in new_instance:
            logger.info(f"新实例全局旋转形状: {new_instance['global_orient'].shape}")
        if "body_pose" in new_instance:
            logger.info(f"新实例关节旋转形状: {new_instance['body_pose'].shape}")
        if "transl" in new_instance:
            logger.info(f"新实例位移形状: {new_instance['transl'].shape}")
            
        return True
        
    except Exception as e:
        logger.warning(f"实例信息比较时出错: {str(e)}，跳过比较")
        return False


def apply_instance_data_with_frame_offset(model, instance_id, instance_data, start_frame=0):
    """
    应用实例数据，但从指定帧开始
    
    Args:
        model: 模型对象
        instance_id: 实例ID
        instance_data: 新实例数据
        start_frame: 开始应用的帧数（默认为0，即从第一帧开始）
    """
    logger.info(f"从第 {start_frame} 帧开始应用新实例数据到实例 {instance_id}")
    
    with torch.no_grad():
        total_frames = model.instances_trans.shape[0]
        
        if start_frame >= total_frames:
            logger.warning(f"起始帧 {start_frame} 超出总帧数 {total_frames}，将从最后一帧开始")
            start_frame = total_frames - 1
        
        # 应用全局旋转（如果存在且从指定帧开始）
        if "global_orient" in instance_data and hasattr(model, 'instances_quats'):
            global_orient = instance_data["global_orient"]
            if len(global_orient.shape) > 2:  # 如果是多帧数据
                # 计算可用的帧数
                available_frames = min(global_orient.shape[0] - start_frame, total_frames - start_frame)
                if available_frames > 0:
                    source_start = start_frame
                    source_end = source_start + available_frames
                    target_start = start_frame
                    target_end = target_start + available_frames
                    
                    model.instances_quats[target_start:target_end, instance_id] = global_orient[source_start:source_end, 0]
                    logger.info(f"已应用全局旋转：帧 {target_start}-{target_end-1}（来源帧 {source_start}-{source_end-1}）")
        
        # 应用关节旋转（如果存在且从指定帧开始）
        if "body_pose" in instance_data and hasattr(model, 'smpl_qauts'):
            body_pose = instance_data["body_pose"]
            if len(body_pose.shape) > 2:  # 如果是多帧数据
                # 计算可用的帧数
                available_frames = min(body_pose.shape[0] - start_frame, total_frames - start_frame)
                if available_frames > 0:
                    source_start = start_frame
                    source_end = source_start + available_frames
                    target_start = start_frame
                    target_end = target_start + available_frames
                    
                    model.smpl_qauts[target_start:target_end, instance_id] = body_pose[source_start:source_end]
                    logger.info(f"已应用关节旋转：帧 {target_start}-{target_end-1}（来源帧 {source_start}-{source_end-1}）")
        
        # 应用位移（如果存在且从指定帧开始，但通常我们保留原位置）
        if "transl" in instance_data and hasattr(model, 'instances_trans'):
            transl = instance_data["transl"]
            if len(transl.shape) > 1:  # 如果是多帧数据
                # 计算可用的帧数
                available_frames = min(transl.shape[0] - start_frame, total_frames - start_frame)
                if available_frames > 0:
                    source_start = start_frame
                    source_end = source_start + available_frames
                    target_start = start_frame
                    target_end = target_start + available_frames
                    
                    # 注意：这里我们通常不替换位移，而是保留原位置
                    # model.instances_trans[target_start:target_end, instance_id] = transl[source_start:source_end]
                    logger.info(f"跳过位移应用（保留原位置）：帧 {target_start}-{target_end-1}")


def handle_insert_replace_operation(args, trainer, dataset, cfg):
    """
    处理插入并替换操作
    """
    logger.info("=" * 60)
    logger.info("开始插入并替换实例操作")
    logger.info("=" * 60)
    
    # 验证实例文件
    instance_file = args.instance_file
    if not os.path.exists(instance_file):
        logger.error(f"实例文件不存在: {instance_file}")
        return False
    
    file_size = Path(instance_file).stat().st_size / 1024 / 1024  # MB
    logger.info(f"将使用实例文件: {instance_file} ({file_size:.2f} MB)")
    
    # 获取帧偏移参数
    start_frame = getattr(args, 'start_frame', 0)
    if start_frame > 0:
        logger.info(f"将从第 {start_frame} 帧开始应用新实例数据")
    
    # 加载实例数据
    logger.info("加载新实例数据...")
    try:
        instance_data = load_instance_data(instance_file)
        logger.info("✓ 新实例数据加载成功")
        
        # 打印实例数据的关键信息
        logger.info("新实例数据内容:")
        for key, value in instance_data.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, dict):
                logger.info(f"  {key}: dict with keys {list(value.keys())}")
            else:
                logger.info(f"  {key}: {type(value)}")
                
    except Exception as e:
        logger.error(f"加载实例数据失败: {str(e)}")
        return False
    
    # 确定实例类型
    if "smpl_template" in instance_data or "voxel_deformer" in instance_data:
        instance_type = "smpl"
    else:
        instance_type = "rigid"
    
    logger.info(f"检测到实例类型: {instance_type.upper()}")
    
    # 验证模型是否支持该类型
    model_key = get_model_key(instance_type)
    if model_key not in trainer.models:
        available_models = list(trainer.models.keys())
        logger.error(f"目标场景不支持{instance_type.upper()}实例，可用模型: {available_models}")
        return False
    
    model = trainer.models[model_key]
    
    # 验证目标实例ID是否存在
    all_ids = get_instance_ids_from_model(model)
    target_id = args.target_instance_id
    if target_id not in all_ids:
        logger.error(f"目标实例ID {target_id} 不存在，可用ID: {all_ids.tolist()}")
        return False
    
    logger.info(f"目标替换实例ID: {target_id}")
    
    # 打印操作前的场景状态
    logger.info("操作前的场景状态:")
    print_node_info(trainer)
    print_instance_info(model, model_key)
    
    # 备份目标实例的变换信息
    backup_data = backup_instance_transforms(model, target_id)
    
    success = False
    new_instance_id = None
    use_replace_function = HAS_REPLACE_FUNCTION  # 创建局部副本避免UnboundLocalError
    
    try:
        # 使用可用的方法进行替换
        if use_replace_function and instance_type == "smpl" and replace_smpl_instance_improved is not None:
            # 使用replace_smpl_instance_improved函数进行替换
            logger.info("使用SMPL改进替换函数进行替换...")
            
            # 安全的实例信息比较
            safe_compare_instance_info(model, target_id, instance_data)
            
            try:
                # 修正函数调用参数
                replace_smpl_instance_improved(
                    trainer=trainer,
                    instance_id=target_id,  # 修正参数名
                    new_instance=instance_data,  # 修正参数名
                    keep_translation=True,  # 保留位置信息
                    keep_global_rot=False   # 使用新的旋转
                )
                
                # 如果指定了起始帧，则重新应用部分数据
                if start_frame > 0:
                    logger.info(f"重新应用从第 {start_frame} 帧开始的数据...")
                    # 先恢复前面的帧数据
                    if 'rotation' in backup_data and hasattr(model, 'instances_quats'):
                        model.instances_quats[:start_frame, target_id] = backup_data['rotation'][:start_frame]
                    if 'smpl_rotation' in backup_data and hasattr(model, 'smpl_qauts'):
                        model.smpl_qauts[:start_frame, target_id] = backup_data['smpl_rotation'][:start_frame]
                    
                    # 然后应用新数据（从指定帧开始）
                    apply_instance_data_with_frame_offset(model, target_id, instance_data, start_frame)
                
                new_instance_id = target_id  # 替换操作保持相同ID
                logger.info(f"✓ 成功使用replace_smpl_instance_improved函数替换实例 {target_id}")
                success = True
                
            except Exception as replace_error:
                logger.error(f"replace_smpl_instance_improved函数执行失败: {str(replace_error)}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                logger.info("将尝试使用备用方法...")
                # 设置标志以使用备用方法
                use_replace_function = False
                
        if not success:  # 如果主方法失败，尝试备用方法
            # 使用插入+移除的方式进行替换
            logger.info("使用插入+移除方式进行替换...")
            
            # 1. 先插入新实例
            if instance_type == "smpl":
                insert_func = insert_smpl_instance
            else:
                insert_func = insert_rigid_instance
            
            # 确定新实例ID - 如果指定了就用指定的，否则使用原ID
            new_id = args.new_instance_id if args.new_instance_id is not None else target_id
            
            # 如果新ID和目标ID相同，需要先临时使用另一个ID
            use_temp_id = (new_id == target_id)
            if use_temp_id:
                # 找一个未使用的临时ID
                temp_id = max(all_ids) + 1
                logger.info(f"使用临时ID {temp_id} 进行插入，之后会调整为 {target_id}")
                actual_insert_id = temp_id
            else:
                actual_insert_id = new_id
            
            # 执行插入
            inserted_id = insert_func(
                trainer=trainer,
                instance_data=instance_data,
                new_instance_id=actual_insert_id,
                device="cuda"
            )
            
            if inserted_id is not None:
                logger.info(f"✓ 成功插入新实例，ID: {inserted_id}")
                
                # 2. 移除原实例
                remove_success = remove_instance_from_model(model, target_id)
                if remove_success:
                    logger.info(f"✓ 成功移除原实例 {target_id}")
                else:
                    logger.warning(f"移除原实例 {target_id} 失败，但继续进行")
                
                # 3. 如果使用了临时ID，现在调整为目标ID
                if use_temp_id and inserted_id != target_id:
                    with torch.no_grad():
                        if hasattr(model, 'point_ids'):
                            temp_mask = model.point_ids[..., 0] == inserted_id
                            model.point_ids[temp_mask, 0] = target_id
                            logger.info(f"✓ 已将实例ID从 {inserted_id} 调整为 {target_id}")
                            new_instance_id = target_id
                        else:
                            new_instance_id = inserted_id
                else:
                    new_instance_id = inserted_id
                
                # 4. 恢复位置信息和应用帧偏移逻辑
                if start_frame > 0:
                    logger.info(f"应用帧偏移逻辑：从第 {start_frame} 帧开始应用新数据...")
                    # 先完全恢复所有变换信息
                    restore_instance_transforms(
                        model=model,
                        instance_id=new_instance_id,
                        backup_data=backup_data,
                        restore_translation=True,
                        restore_rotation=True,
                        restore_scale=True
                    )
                    # 然后从指定帧开始应用新数据
                    apply_instance_data_with_frame_offset(model, new_instance_id, instance_data, start_frame)
                else:
                    # 正常的恢复逻辑
                    restore_instance_transforms(
                        model=model,
                        instance_id=new_instance_id,
                        backup_data=backup_data,
                        restore_translation=True,  # 使用原位置
                        restore_rotation=False,    # 保留新旋转
                        restore_scale=False        # 保留新缩放
                    )
                
                success = True
                
            else:
                logger.error("✗ 新实例插入失败")
    
    except Exception as e:
        logger.error(f"替换操作过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
    if success:
        # 打印操作后的场景状态
        logger.info("操作后的场景状态:")
        print_node_info(trainer)
        print_instance_info(model, model_key)
        
        logger.info(f"✓ 插入并替换操作成功完成!")
        logger.info(f"  原实例ID: {target_id}")
        logger.info(f"  新实例ID: {new_instance_id}")
        logger.info(f"  保留了原实例的位置信息")
        logger.info(f"  使用了新实例的旋转和动作")
        
        # 渲染结果（如果指定了输出目录）
        if args.output_dir and not args.no_render:
            logger.info("开始渲染替换后的场景...")
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
                    post_fix=f"_replace_{target_id}_with_{new_instance_id}"
                )
                logger.info(f"渲染完成，结果保存到: {output_dir}")
                
                # 保存操作信息
                final_node_info = print_node_info(trainer)
                save_node_info(final_node_info, str(output_dir))
                
                # 保存替换操作的详细信息
                import json
                replace_info = {
                    "operation": "insert_replace",
                    "source_instance_file": str(Path(instance_file).absolute()),
                    "target_instance_id": target_id,
                    "new_instance_id": new_instance_id,
                    "instance_type": instance_type,
                    "start_frame": start_frame,
                    "used_replace_function": use_replace_function and instance_type == "smpl",
                    "replace_function_name": "replace_smpl_instance_improved" if (use_replace_function and instance_type == "smpl") else None,
                    "preserved_transforms": {
                        "translation": True,
                        "rotation": False if start_frame == 0 else "partial",
                        "scale": False
                    },
                    "frame_application": {
                        "original_frames_preserved": f"0-{start_frame-1}" if start_frame > 0 else "none",
                        "new_frames_applied": f"{start_frame}-end" if start_frame > 0 else "all"
                    }
                }
                
                replace_info_path = output_dir / "replace_operation_info.json"
                with open(replace_info_path, 'w', encoding='utf-8') as f:
                    json.dump(replace_info, f, indent=2, ensure_ascii=False)
                logger.info(f"替换操作信息已保存到: {replace_info_path}")
                
            except Exception as e:
                logger.error(f"渲染过程中出错: {str(e)}")
                logger.error(traceback.format_exc())
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="插入并替换实例工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 基本替换操作:
   python scene_editing/insert_replace_instance.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_file ./saved_instances/smpl_instance_0.pkl \\
       --target_instance_id 2 \\
       --output_dir ./replace_output

2. 指定新实例ID的替换:
   python scene_editing/insert_replace_instance.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_file ./saved_instances/smpl_instance_0.pkl \\
       --target_instance_id 2 \\
       --new_instance_id 5 \\
       --output_dir ./replace_output

3. 从第10帧开始应用新实例数据:
   python scene_editing/insert_replace_instance.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_file ./saved_instances/smpl_instance_0.pkl \\
       --target_instance_id 2 \\
       --start_frame 10 \\
       --output_dir ./replace_output

4. 只进行替换不渲染:
   python scene_editing/insert_replace_instance.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_file ./saved_instances/smpl_instance_0.pkl \\
       --target_instance_id 2 \\
       --no_render

说明:
- 该脚本会保留目标实例的位置信息
- 使用新实例的旋转、动作和其他属性
- --start_frame 可以指定从新实例的第几帧开始应用数据
- 前面的帧会保持原实例的数据不变
- 对于SMPL实例，优先使用replace_smpl_instance函数
- 对于其他类型，使用插入+手动替换的方式
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--resume_from", type=str, required=True,
        help="检查点路径"
    )
    
    # 替换操作参数
    parser.add_argument(
        "--instance_file", type=str, required=True,
        help="要插入的新实例文件路径"
    )
    parser.add_argument(
        "--target_instance_id", type=int, required=True,
        help="要被替换的目标实例ID"
    )
    parser.add_argument(
        "--new_instance_id", type=int,
        help="新实例ID（可选，如果不指定则自动分配或保持原ID）"
    )
    parser.add_argument(
        "--start_frame", type=int, default=0,
        help="从新实例的第几帧开始应用数据（默认为0，即从第一帧开始）"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir", type=str,
        help="输出目录（用于渲染和保存信息）"
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help="不渲染结果，只进行替换操作"
    )
    
    # 配置覆盖
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="配置覆盖选项"
    )
    
    args = parser.parse_args()
    
    try:
        # 验证实例文件是否存在
        if not os.path.exists(args.instance_file):
            logger.error(f"实例文件不存在: {args.instance_file}")
            sys.exit(1)
        
        # 加载模型和数据集
        trainer, dataset, cfg = load_trainer_and_dataset(args.resume_from, args.opts)
        
        # 执行插入并替换操作
        success = handle_insert_replace_operation(args, trainer, dataset, cfg)
        
        if success:
            logger.info("=" * 60)
            logger.info("插入并替换操作完成!")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("插入并替换操作失败!")
            logger.error("=" * 60)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()