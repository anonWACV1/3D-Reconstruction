#!/usr/bin/env python3
"""
实例保存、插入和变换脚本
支持保存和插入SMPL和Rigid实例，以及对实例进行平移、旋转、缩放操作

使用示例:
# 保存实例
export PYTHONPATH=$(pwd)
python scene_editing/insert_instance.py \
    --resume_from output/Kitti/dataset=Kitti/line_change_gt/checkpoint_final.pth \
    --operation save \
    --instance_type smpl \
    --instance_ids 0 1 2 \
    --save_dir ./saved_instances_kitti

# 插入实例
python scene_editing/insert_instance.py \
    --resume_from output/new_scene/dataset=waymo/1cams/checkpoint_final.pth \
    --operation insert \
    --instance_files ./saved_instances/smpl_instance_0.pkl ./saved_instances/smpl_instance_1.pkl \
    --new_instance_ids 5 6 \
    --output_dir ./insert_output

# 平移实例
python scene_editing/insert_instance.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --operation translate \
    --instance_type smpl \
    --instance_id 0 \
    --translation 1.0 0.5 0.0 \
    --frame_range 10-50 \
    --output_dir ./transform_output

# 旋转实例
python scene_editing/insert_instance.py \
    --resume_from output/waymo_1cam_edit/dataset=waymo/1cams/checkpoint_final.pth \
    --operation rotate \
    --instance_type rigid \
    --instance_id 1 \
    --rotation_axis 0 0 1 \
    --rotation_angle 45 \
    --output_dir ./transform_output
"""

import os
import sys
import logging
import argparse
import traceback
import math
from pathlib import Path
from typing import List, Optional
from omegaconf import OmegaConf
import torch

# 确保能够导入项目模块
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset
from utils.geometry import quaternion_multiply

# 导入场景编辑函数
try:
    from scene_editing.scene_editing import (
        save_smpl_instance,
        save_rigid_instance,
        load_instance_data,
        insert_smpl_instance,
        insert_rigid_instance,
        batch_save_instances,
        handle_insert_and_transform_operation,  # 新增导入
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
        logging.FileHandler('insert_instance.log')
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
    
    for model_name in available_models:
        model = trainer.models[model_name]
        if hasattr(model, 'num_instances'):
            try:
                num_instances = model.num_instances if callable(model.num_instances) else model.num_instances
                logger.info(f"  {model_name}: {num_instances} 个实例")
            except:
                logger.info(f"  {model_name}: 无法获取实例数量")
        if hasattr(model, 'num_points'):
            try:
                num_points = model.num_points if callable(model.num_points) else model.num_points
                logger.info(f"  {model_name}: {num_points} 个点")
            except:
                logger.info(f"  {model_name}: 无法获取点数")
    
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
    # logger.info(f"  实例ID列表: {all_ids.tolist()}")
    
    # 统计每个实例的点数
    if hasattr(model, 'point_ids'):
        for instance_id in all_ids:
            point_count = (model.point_ids[..., 0] == instance_id).sum().item()
            logger.info(f"    实例 {instance_id}: {point_count} 个点")


def handle_save_operation(args, trainer):
    """处理保存操作"""
    logger.info("=" * 60)
    logger.info(f"开始保存{args.instance_type.upper()}实例")
    logger.info("=" * 60)
    
    # 验证模型类型
    model_key = get_model_key(args.instance_type)
    if model_key not in trainer.models:
        available_models = list(trainer.models.keys())
        raise ValueError(f"模型中没有找到{model_key}，可用模型: {available_models}")
    
    model = trainer.models[model_key]
    
    # 打印当前实例信息
    print_instance_info(model, model_key)
    
    # 验证要保存的实例ID
    all_ids = get_instance_ids_from_model(model)
    instance_ids = args.instance_ids
    invalid_ids = [id for id in instance_ids if id not in all_ids]
    if invalid_ids:
        logger.warning(f"以下实例ID不存在，将被跳过: {invalid_ids}")
        instance_ids = [id for id in instance_ids if id in all_ids]
    
    if not instance_ids:
        logger.error("没有有效的实例ID可以保存")
        return
    
    logger.info(f"将保存以下实例ID: {instance_ids}")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"保存目录: {save_dir.absolute()}")
    
    # 执行批量保存
    try:
        prefix = args.prefix + "_" if args.prefix else ""
        saved_files = batch_save_instances(
            trainer=trainer,
            instance_type=args.instance_type,
            instance_ids=instance_ids,
            save_dir=str(save_dir),
            prefix=prefix
        )
        
        logger.info("保存操作完成!")
        logger.info(f"成功保存的文件:")
        for file_path in saved_files:
            file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
            logger.info(f"  {file_path} ({file_size:.2f} MB)")
            
        # 保存场景节点信息
        if args.save_scene_info:
            logger.info("保存场景节点信息...")
            node_info = print_node_info(trainer)
            save_node_info(node_info, str(save_dir))
            
    except Exception as e:
        logger.error(f"保存过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def handle_insert_operation(args, trainer, dataset, cfg):
    """处理插入操作"""
    logger.info("=" * 60)
    logger.info("开始插入实例")
    logger.info("=" * 60)
    
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
        return
    
    logger.info(f"将插入以下实例文件:")
    for file_path in valid_files:
        file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
        logger.info(f"  {file_path} ({file_size:.2f} MB)")
    
    # 准备新实例ID
    new_instance_ids = args.new_instance_ids
    if new_instance_ids and len(new_instance_ids) != len(valid_files):
        logger.warning(f"新实例ID数量({len(new_instance_ids)})与文件数量({len(valid_files)})不匹配")
        new_instance_ids = new_instance_ids[:len(valid_files)] if new_instance_ids else None
    
    # 打印插入前的场景状态
    logger.info("插入前的场景状态:")
    node_info_before = print_node_info(trainer)
    
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
                insert_func = insert_smpl_instance
            else:
                instance_type = "rigid"
                insert_func = insert_rigid_instance
            
            logger.info(f"检测到实例类型: {instance_type.upper()}")
            
            # 验证模型是否支持该类型
            model_key = get_model_key(instance_type)
            if model_key not in trainer.models:
                logger.error(f"目标场景不支持{instance_type.upper()}实例，跳过")
                continue
            
            # 获取插入前的实例ID列表
            model = trainer.models[model_key]
            ids_before = set(get_instance_ids_from_model(model))
            
            # 确定新实例ID
            new_id = new_instance_ids[i] if new_instance_ids and i < len(new_instance_ids) else None
            
            # 执行插入
            actual_id = insert_func(
                trainer=trainer,
                instance_data=instance_data,
                new_instance_id=new_id,
                device="cuda"
            )
            
            # 获取插入后的实例ID列表，确认新增的ID
            ids_after = set(get_instance_ids_from_model(model))
            new_ids = ids_after - ids_before
            
            if actual_id is not None:
                inserted_ids.append(actual_id)
                logger.info(f"✓ 成功插入实例，分配ID: {actual_id}")
                if len(new_ids) > 1:
                    logger.info(f"  新增实例ID列表: {sorted(list(new_ids))}")
            else:
                logger.error(f"✗ 插入失败，未返回有效ID")
            
        except Exception as e:
            logger.error(f"插入文件 {file_path} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # 打印插入后的场景状态
    logger.info("\n插入后的场景状态:")
    node_info_after = print_node_info(trainer)
    
    logger.info(f"\n插入操作完成!")
    logger.info(f"成功插入 {len(inserted_ids)}/{len(valid_files)} 个实例")
    if inserted_ids:
        logger.info(f"已插入的实例ID: {inserted_ids}")
        
        # 详细打印每个模型的实例变化
        for model_name in trainer.models.keys():
            if "Nodes" in model_name:
                model = trainer.models[model_name]
                print_instance_info(model, model_name)
    
    # 渲染结果（如果指定了输出目录）
    if args.output_dir and inserted_ids:
        logger.info("\n开始渲染插入后的场景...")
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
                post_fix="_inserted"
            )
            logger.info(f"渲染完成，结果保存到: {output_dir}")
            
            # 保存插入后的场景信息
            save_node_info(node_info_after, str(output_dir))
            
        except Exception as e:
            logger.error(f"渲染过程中出错: {str(e)}")
            logger.error(traceback.format_exc())


def handle_translate_operation(args, trainer, dataset, cfg):
    """处理平移操作 - 使用现有的add_transform_offset方法"""
    logger.info("=" * 60)
    logger.info("开始平移操作")
    logger.info("=" * 60)
    
    # 验证实例类型
    model_key = get_model_key(args.instance_type)
    if model_key not in trainer.models:
        logger.error(f"模型中没有找到{model_key}")
        return False
    
    model = trainer.models[model_key]
    
    # 验证实例ID是否存在
    all_ids = get_instance_ids_from_model(model)
    if args.instance_id not in all_ids:
        logger.error(f"实例ID {args.instance_id} 不存在，可用ID: {all_ids}")
        return False
    
    logger.info(f"对{args.instance_type.upper()}实例 {args.instance_id} 进行平移")
    logger.info(f"平移偏移量: {args.translation}")
    
    # 打印操作前的场景信息
    logger.info("平移前的场景状态:")
    print_node_info(trainer)
    
    # 将偏移量转换为张量
    translation_offset = torch.tensor(args.translation, device=trainer.device, dtype=torch.float32)
    
    # 确定要操作的帧范围
    total_frames = model.num_frames if hasattr(model, 'num_frames') else model.instances_trans.shape[0]
    
    if args.frame_range:
        try:
            if "-" in args.frame_range:
                start_frame, end_frame = map(int, args.frame_range.split("-"))
                frame_indices = list(range(max(0, start_frame), min(total_frames, end_frame + 1)))
            else:
                frame_idx = int(args.frame_range)
                if 0 <= frame_idx < total_frames:
                    frame_indices = [frame_idx]
                else:
                    logger.error(f"帧索引 {frame_idx} 超出范围 [0, {total_frames-1}]")
                    return False
        except ValueError:
            logger.error(f"无效的帧范围格式: {args.frame_range}")
            return False
    else:
        frame_indices = list(range(total_frames))
    
    logger.info(f"将对 {len(frame_indices)} 帧进行平移: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")
    
    # 执行平移操作 - 使用模型自带的add_transform_offset方法
    try:
        with torch.no_grad():
            for frame_idx in frame_indices:
                model.add_transform_offset(
                    instance_id=args.instance_id,
                    frame_idx=frame_idx,
                    translation_offset=translation_offset
                )
            
            logger.info(f"✓ 成功对实例 {args.instance_id} 进行了平移变换")
            
            # 打印操作后的场景信息
            logger.info("平移后的场景状态:")
            print_node_info(trainer)
            
            # 渲染结果
            if args.output_dir:
                logger.info("开始渲染平移后的场景...")
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
                        post_fix=f"_translate_{args.instance_type}_{args.instance_id}"
                    )
                    logger.info(f"渲染完成，结果保存到: {output_dir}")
                    
                except Exception as e:
                    logger.error(f"渲染过程中出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            return True
            
    except Exception as e:
        logger.error(f"平移操作失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def handle_rotate_operation(args, trainer, dataset, cfg):
    """处理旋转操作 - 使用现有的add_transform_offset方法"""
    logger.info("=" * 60)
    logger.info("开始旋转操作")
    logger.info("=" * 60)
    
    # 验证实例类型
    model_key = get_model_key(args.instance_type)
    if model_key not in trainer.models:
        logger.error(f"模型中没有找到{model_key}")
        return False
    
    model = trainer.models[model_key]
    
    # 验证实例ID是否存在
    all_ids = get_instance_ids_from_model(model)
    if args.instance_id not in all_ids:
        logger.error(f"实例ID {args.instance_id} 不存在，可用ID: {all_ids}")
        return False
    
    logger.info(f"对{args.instance_type.upper()}实例 {args.instance_id} 进行旋转")
    logger.info(f"旋转轴: {args.rotation_axis}, 角度: {args.rotation_angle} 度")
    
    # 打印操作前的场景信息
    logger.info("旋转前的场景状态:")
    print_node_info(trainer)
    
    # 处理角度转换
    angle_radians = math.radians(args.rotation_angle)
    
    # 归一化旋转轴
    rotation_axis = torch.tensor(args.rotation_axis, device=trainer.device, dtype=torch.float32)
    rotation_axis = rotation_axis / torch.norm(rotation_axis)
    
    # 创建轴角表示
    axis_angle = rotation_axis * angle_radians
    
    # 转换为四元数
    from pytorch3d.transforms import axis_angle_to_quaternion
    rotation_quaternion = axis_angle_to_quaternion(axis_angle)
    
    # 确定要操作的帧范围
    total_frames = model.num_frames if hasattr(model, 'num_frames') else model.instances_trans.shape[0]
    
    if args.frame_range:
        try:
            if "-" in args.frame_range:
                start_frame, end_frame = map(int, args.frame_range.split("-"))
                frame_indices = list(range(max(0, start_frame), min(total_frames, end_frame + 1)))
            else:
                frame_idx = int(args.frame_range)
                if 0 <= frame_idx < total_frames:
                    frame_indices = [frame_idx]
                else:
                    logger.error(f"帧索引 {frame_idx} 超出范围 [0, {total_frames-1}]")
                    return False
        except ValueError:
            logger.error(f"无效的帧范围格式: {args.frame_range}")
            return False
    else:
        frame_indices = list(range(total_frames))
    
    logger.info(f"将对 {len(frame_indices)} 帧进行旋转: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")
    
    # 执行旋转操作 - 使用模型自带的add_transform_offset方法
    try:
        with torch.no_grad():
            for frame_idx in frame_indices:
                model.add_transform_offset(
                    instance_id=args.instance_id,
                    frame_idx=frame_idx,
                    rotation_offset=rotation_quaternion
                )
            
            logger.info(f"✓ 成功对实例 {args.instance_id} 进行了旋转变换")
            
            # 打印操作后的场景信息
            logger.info("旋转后的场景状态:")
            print_node_info(trainer)
            
            # 渲染结果
            if args.output_dir:
                logger.info("开始渲染旋转后的场景...")
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
                        post_fix=f"_rotate_{args.instance_type}_{args.instance_id}"
                    )
                    logger.info(f"渲染完成，结果保存到: {output_dir}")
                    
                except Exception as e:
                    logger.error(f"渲染过程中出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            return True
            
    except Exception as e:
        logger.error(f"旋转操作失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False
def handle_insert_and_transform_wrapper(args, trainer, dataset, cfg):
    """处理插入并变换操作的包装函数"""
    logger.info("=" * 60)
    logger.info("开始插入并变换操作")
    logger.info("=" * 60)
    
    # 打印操作前的场景信息
    logger.info("操作前的场景状态:")
    print_node_info(trainer)
    
    # 调用scene_editing.py中的组合操作函数
    inserted_ids, success = handle_insert_and_transform_operation(
        trainer=trainer,
        instance_files=args.instance_files,
        new_instance_ids=args.new_instance_ids,
        translation=args.translation,
        rotation_axis=args.rotation_axis,
        rotation_angle=args.rotation_angle,
        frame_range=args.frame_range,
        device="cuda"
    )
    
    if success:
        logger.info(f"✓ 插入并变换操作成功完成")
        logger.info(f"插入并变换的实例ID: {inserted_ids}")
        
        # 打印操作后的场景信息
        logger.info("操作后的场景状态:")
        print_node_info(trainer)
        
        # 渲染结果（如果指定了输出目录）
        if args.output_dir:
            logger.info("开始渲染最终场景...")
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
                    post_fix="_insert_and_transform"
                )
                logger.info(f"渲染完成，结果保存到: {output_dir}")
                
                # 保存最终场景信息
                final_node_info = print_node_info(trainer)
                save_node_info(final_node_info, str(output_dir))
                
            except Exception as e:
                logger.error(f"渲染过程中出错: {str(e)}")
                logger.error(traceback.format_exc())
    else:
        logger.error("✗ 插入并变换操作失败")
    
    return success

def handle_info_operation(args, trainer):
    """处理信息查看操作"""
    logger.info("=" * 60)
    logger.info("场景信息查看")
    logger.info("=" * 60)
    
    # 打印详细的场景信息
    node_info = print_node_info(trainer)
    
    # 详细打印每个模型的实例信息
    for model_name in trainer.models.keys():
        if "Nodes" in model_name:
            model = trainer.models[model_name]
            print_instance_info(model, model_name)
    
    # 如果指定了输出目录，保存信息到文件
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_node_info(node_info, str(output_dir))
        logger.info(f"场景信息已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="实例保存、插入和变换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 保存实例:
   python scene_editing/insert_instance.py \\
       --resume_from output/scene1/checkpoint_final.pth \\
       --operation save \\
       --instance_type smpl \\
       --instance_ids 0 1 2 \\
       --save_dir ./saved_instances

2. 插入实例:
   python scene_editing/insert_instance.py \\
       --resume_from output/scene2/checkpoint_final.pth \\
       --operation insert \\
       --instance_files ./saved_instances/smpl_instance_0.pkl \\
       --new_instance_ids 5 \\
       --output_dir ./insert_output

3. 平移实例:
   python scene_editing/insert_instance.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --operation translate \\
       --instance_type smpl \\
       --instance_id 0 \\
       --translation 1.0 0.5 0.0 \\
       --frame_range 10-50 \\
       --output_dir ./transform_output

4. 旋转实例:
   python scene_editing/insert_instance.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --operation rotate \\
       --instance_type rigid \\
       --instance_id 1 \\
       --rotation_axis 0 0 1 \\
       --rotation_angle 45 \\
       --output_dir ./transform_output

5. 查看场景信息:
   python scene_editing/insert_instance.py \\
       --resume_from output/scene1/checkpoint_final.pth \\
       --operation info \\
       --output_dir ./scene_info
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--resume_from", type=str, required=True,
        help="检查点路径"
    )
    parser.add_argument(
        "--operation", type=str, required=True,
        choices=["save", "insert", "translate", "rotate", "insert_and_transform", "info"],
        help="操作类型: save(保存实例), insert(插入实例), translate(平移实例), rotate(旋转实例), insert_and_transform(插入并变换), info(查看信息)"
    )
    
    # 保存操作参数
    parser.add_argument(
        "--instance_type", type=str,
        choices=["smpl", "rigid"],
        help="实例类型 (保存、平移、旋转操作需要)"
    )
    parser.add_argument(
        "--instance_ids", type=int, nargs="+",
        help="要保存的实例ID列表 (保存操作必需)"
    )
    parser.add_argument(
        "--save_dir", type=str,
        help="保存目录 (保存操作必需)"
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="保存文件的前缀"
    )
    parser.add_argument(
        "--save_scene_info", action="store_true",
        help="是否保存场景节点信息"
    )
    
    # 插入操作参数
    parser.add_argument(
        "--instance_files", type=str, nargs="+",
        help="要插入的实例文件路径列表 (插入操作必需)"
    )
    parser.add_argument(
        "--new_instance_ids", type=int, nargs="+",
        help="新实例ID列表，如果不指定则自动分配"
    )
    
    # 变换操作参数
    parser.add_argument(
        "--instance_id", type=int,
        help="要变换的实例ID (平移、旋转操作必需)"
    )
    parser.add_argument(
        "--translation", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
        help="平移偏移量 [x, y, z] (平移操作必需)"
    )
    parser.add_argument(
        "--rotation_axis", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
        help="旋转轴 [x, y, z] (旋转操作必需)"
    )
    parser.add_argument(
        "--rotation_angle", type=float,
        help="旋转角度(度) (旋转操作必需)"
    )
    parser.add_argument(
        "--frame_range", type=str,
        help="帧范围，格式为 'start-end' 或单个数字，不指定则应用于所有帧"
    )
    
    # 通用参数
    parser.add_argument(
        "--output_dir", type=str,
        help="输出目录（用于渲染或保存信息）"
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help="不渲染结果，只进行操作"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="配置覆盖选项"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.operation == "save":
        if not args.instance_type or not args.instance_ids or not args.save_dir:
            parser.error("保存操作需要指定 --instance_type, --instance_ids 和 --save_dir")
    elif args.operation == "insert":
        if not args.instance_files:
            parser.error("插入操作需要指定 --instance_files")
    elif args.operation == "translate":
        if not args.instance_type or args.instance_id is None or not args.translation:
            parser.error("平移操作需要指定 --instance_type, --instance_id 和 --translation")
    elif args.operation == "rotate":
        if not args.instance_type or args.instance_id is None or not args.rotation_axis or args.rotation_angle is None:
            parser.error("旋转操作需要指定 --instance_type, --instance_id, --rotation_axis 和 --rotation_angle")
    
    try:
        # 加载模型和数据集
        trainer, dataset, cfg = load_trainer_and_dataset(args.resume_from, args.opts)
        
        # 根据操作类型执行相应功能
        success = True
        
        if args.operation == "save":
            handle_save_operation(args, trainer)
        elif args.operation == "insert":
            handle_insert_operation(args, trainer, dataset, cfg)
        elif args.operation == "translate":
            success = handle_translate_operation(args, trainer, dataset, cfg)
        elif args.operation == "rotate":
            success = handle_rotate_operation(args, trainer, dataset, cfg)
        elif args.operation == "insert_and_transform":
            success = handle_insert_and_transform_wrapper(args, trainer, dataset, cfg)
        elif args.operation == "info":
            handle_info_operation(args, trainer)
        
        if success:
            logger.info("=" * 60)
            logger.info(f"{args.operation.upper()}操作完成!")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error(f"{args.operation.upper()}操作失败!")
            logger.error("=" * 60)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()