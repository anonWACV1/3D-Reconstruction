from typing import List, Optional, Dict
from omegaconf import OmegaConf
import os
import time
import logging
import torch
import numpy as np

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_novel_views
from torch.nn import Parameter

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def load_config(config_path: str) -> OmegaConf:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # 验证必要参数
    if not cfg.source_checkpoint:
        raise ValueError("source_checkpoint must be specified in config file")
    if not cfg.target_checkpoint:
        raise ValueError("target_checkpoint must be specified in config file")
    
    logger.info(f"Loaded config from {config_path}")
    return cfg

def load_checkpoint(checkpoint_path: str) -> tuple:
    """加载checkpoint并返回训练器和数据集"""
    log_dir = os.path.dirname(checkpoint_path)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建数据集
    dataset = DrivingDataset(data_cfg=cfg.data)
    
    # 设置训练器
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    
    # 从检查点恢复
    trainer.resume_from_checkpoint(ckpt_path=checkpoint_path, load_only_model=True)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return trainer, dataset

def extract_nodes(
    trainer: BasicTrainer,
    rigid_ids: Optional[List[int]] = None,
    smpl_ids: Optional[List[int]] = None,
    deformable_ids: Optional[List[int]] = None
) -> Dict:
    """从训练器中提取指定节点"""
    extracted = {"rigid": {}, "smpl": {}, "deformable": {}}
    
    # 提取Rigid节点
    if "RigidNodes" in trainer.models and rigid_ids:
        logger.info(f"Extracting rigid nodes with IDs: {rigid_ids}")
        rigid_nodes = trainer.models["RigidNodes"]
        for id in rigid_ids:
            extracted["rigid"][id] = {
                "points": rigid_nodes._means[rigid_nodes.point_ids[..., 0] == id].clone(),
                "point_ids": rigid_nodes.point_ids[rigid_nodes.point_ids[..., 0] == id].clone(),
                "features": rigid_nodes._features_dc[rigid_nodes.point_ids[..., 0] == id].clone(),
                "scales": rigid_nodes._scaling[rigid_nodes.point_ids[..., 0] == id].clone(),
                "rotations": rigid_nodes._rotation[rigid_nodes.point_ids[..., 0] == id].clone(),
                "opacities": rigid_nodes._opacity[rigid_nodes.point_ids[..., 0] == id].clone(),
            }
        logger.info(f"Successfully extracted {len(rigid_ids)} rigid nodes")
    else:
        logger.info("No rigid nodes specified or RigidNodes model not found - skipping rigid nodes extraction")
    
    # 提取SMPL节点
    if "SMPLNodes" in trainer.models and smpl_ids:
        logger.info(f"Extracting SMPL nodes with IDs: {smpl_ids}")
        smpl_nodes = trainer.models["SMPLNodes"]
        for id in smpl_ids:
            extracted["smpl"][id] = {
                "points": smpl_nodes._means[smpl_nodes.point_ids[..., 0] == id].clone(),
                "point_ids": smpl_nodes.point_ids[smpl_nodes.point_ids[..., 0] == id].clone(),
                "features": smpl_nodes._features_dc[smpl_nodes.point_ids[..., 0] == id].clone(),
                "scales": smpl_nodes._scaling[smpl_nodes.point_ids[..., 0] == id].clone(),
                "rotations": smpl_nodes._rotation[smpl_nodes.point_ids[..., 0] == id].clone(),
                "opacities": smpl_nodes._opacity[smpl_nodes.point_ids[..., 0] == id].clone(),
            }
        logger.info(f"Successfully extracted {len(smpl_ids)} SMPL nodes")
    else:
        logger.info("No SMPL nodes specified or SMPLNodes model not found - skipping SMPL nodes extraction")
    
    # 提取Deformable节点
    if "DeformableNodes" in trainer.models and deformable_ids:
        logger.info(f"Extracting deformable nodes with IDs: {deformable_ids}")
        deformable_nodes = trainer.models["DeformableNodes"]
        for id in deformable_ids:
            extracted["deformable"][id] = {
                "points": deformable_nodes._means[deformable_nodes.point_ids[..., 0] == id].clone(),
                "point_ids": deformable_nodes.point_ids[deformable_nodes.point_ids[..., 0] == id].clone(),
                "features": deformable_nodes._features_dc[deformable_nodes.point_ids[..., 0] == id].clone(),
                "scales": deformable_nodes._scaling[deformable_nodes.point_ids[..., 0] == id].clone(),
                "rotations": deformable_nodes._rotation[deformable_nodes.point_ids[..., 0] == id].clone(),
                "opacities": deformable_nodes._opacity[deformable_nodes.point_ids[..., 0] == id].clone(),
                "embedding": deformable_nodes.instances_embedding[id].clone()
            }
        logger.info(f"Successfully extracted {len(deformable_ids)} deformable nodes")
    else:
        logger.info("No deformable nodes specified or DeformableNodes model not found - skipping deformable nodes extraction")
    
    return extracted

def insert_nodes(
    target_trainer: BasicTrainer,
    extracted_nodes: Dict,
    rigid_offset: Optional[int] = None,
    smpl_offset: Optional[int] = None,
    deformable_offset: Optional[int] = None
):
    """将提取的节点插入到目标训练器中"""
    # 插入Rigid节点
    if "RigidNodes" in target_trainer.models and extracted_nodes["rigid"]:
        logger.info("Inserting rigid nodes into target model")
        rigid_nodes = target_trainer.models["RigidNodes"]
        for src_id, node_data in extracted_nodes["rigid"].items():
            new_id = src_id + rigid_offset if rigid_offset is not None else src_id
            rigid_nodes._means = torch.cat([rigid_nodes._means, node_data["points"]])
            rigid_nodes._features_dc = torch.cat([rigid_nodes._features_dc, node_data["features"]])
            rigid_nodes._scaling = torch.cat([rigid_nodes._scaling, node_data["scales"]])
            rigid_nodes._rotation = torch.cat([rigid_nodes._rotation, node_data["rotations"]])
            rigid_nodes._opacity = torch.cat([rigid_nodes._opacity, node_data["opacities"]])
            
            # 更新point_ids
            new_point_ids = node_data["point_ids"].clone()
            new_point_ids[..., 0] = new_id
            rigid_nodes.point_ids = torch.cat([rigid_nodes.point_ids, new_point_ids])
        logger.info(f"Successfully inserted {len(extracted_nodes['rigid'])} rigid nodes")
    else:
        logger.info("No rigid nodes to insert or RigidNodes model not found - skipping rigid nodes insertion")
    
    # 插入SMPL节点
    if "SMPLNodes" in target_trainer.models and extracted_nodes["smpl"]:
        logger.info("Inserting SMPL nodes into target model")
        smpl_nodes = target_trainer.models["SMPLNodes"]
        for src_id, node_data in extracted_nodes["smpl"].items():
            new_id = src_id + smpl_offset if smpl_offset is not None else src_id
            smpl_nodes._means = torch.cat([smpl_nodes._means, node_data["points"]])
            smpl_nodes._features_dc = torch.cat([smpl_nodes._features_dc, node_data["features"]])
            smpl_nodes._scaling = torch.cat([smpl_nodes._scaling, node_data["scales"]])
            smpl_nodes._rotation = torch.cat([smpl_nodes._rotation, node_data["rotations"]])
            smpl_nodes._opacity = torch.cat([smpl_nodes._opacity, node_data["opacities"]])
            
            # 更新point_ids
            new_point_ids = node_data["point_ids"].clone()
            new_point_ids[..., 0] = new_id
            smpl_nodes.point_ids = torch.cat([smpl_nodes.point_ids, new_point_ids])
        logger.info(f"Successfully inserted {len(extracted_nodes['smpl'])} SMPL nodes")
    else:
        logger.info("No SMPL nodes to insert or SMPLNodes model not found - skipping SMPL nodes insertion")
    
    # 插入Deformable节点
    if "DeformableNodes" in target_trainer.models and extracted_nodes["deformable"]:
        logger.info("Inserting deformable nodes into target model")
        deformable_nodes = target_trainer.models["DeformableNodes"]
        for src_id, node_data in extracted_nodes["deformable"].items():
            new_id = src_id + deformable_offset if deformable_offset is not None else src_id
            deformable_nodes._means = torch.cat([deformable_nodes._means, node_data["points"]])
            deformable_nodes._features_dc = torch.cat([deformable_nodes._features_dc, node_data["features"]])
            deformable_nodes._scaling = torch.cat([deformable_nodes._scaling, node_data["scales"]])
            deformable_nodes._rotation = torch.cat([deformable_nodes._rotation, node_data["rotations"]])
            deformable_nodes._opacity = torch.cat([deformable_nodes._opacity, node_data["opacities"]])
            
            # 更新point_ids
            new_point_ids = node_data["point_ids"].clone()
            new_point_ids[..., 0] = new_id
            deformable_nodes.point_ids = torch.cat([deformable_nodes.point_ids, new_point_ids])
            
            # 更新embedding
            if deformable_nodes.instances_embedding.shape[0] <= new_id:
                # 如果需要扩展embedding矩阵
                new_embedding = torch.randn(new_id + 1, deformable_nodes.networks_cfg.embed_dim, 
                                          device=deformable_nodes.device)
                new_embedding[:deformable_nodes.instances_embedding.shape[0]] = deformable_nodes.instances_embedding
                deformable_nodes.instances_embedding = Parameter(new_embedding)
            deformable_nodes.instances_embedding.data[new_id] = node_data["embedding"]
        logger.info(f"Successfully inserted {len(extracted_nodes['deformable'])} deformable nodes")
    else:
        logger.info("No deformable nodes to insert or DeformableNodes model not found - skipping deformable nodes insertion")

def render_result_video(
    cfg: OmegaConf,
    trainer: BasicTrainer,
    dataset: DrivingDataset,
    output_dir: str,
    video_name: str,
    fps: int = 30
):
    """渲染结果视频"""
    # 获取前视图渲染轨迹
    render_traj = dataset.get_novel_render_traj(
        traj_types=["front"],
        target_frames=dataset.frame_num
    )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 渲染并保存视频
    for traj_type, traj in render_traj.items():
        render_data = dataset.prepare_novel_view_render_data(traj)
        save_path = os.path.join(output_dir, f"{video_name}.mp4")
        
        render_novel_views(
            trainer,
            render_data,
            save_path,
            fps=fps
        )
        logger.info(f"Result video saved to {save_path}")

def main(cfg: OmegaConf):
    """主函数"""
    # 加载源checkpoint(从中提取节点)
    src_trainer, src_dataset = load_checkpoint(cfg.source_checkpoint)
    
    # 加载目标checkpoint(将节点插入到这里)
    target_trainer, target_dataset = load_checkpoint(cfg.target_checkpoint)
    
    # 提取指定节点
    extracted_nodes = extract_nodes(
        src_trainer,
        rigid_ids=cfg.rigid_ids,
        smpl_ids=cfg.smpl_ids,
        deformable_ids=cfg.deformable_ids
    )
    
    # 将节点插入目标训练器
    insert_nodes(
        target_trainer,
        extracted_nodes,
        rigid_offset=cfg.rigid_offset,
        smpl_offset=cfg.smpl_offset,
        deformable_offset=cfg.deformable_offset
    )
    
    # 渲染结果视频
    render_result_video(
        cfg=OmegaConf.load(os.path.join(os.path.dirname(cfg.target_checkpoint), "config.yaml")),
        trainer=target_trainer,
        dataset=target_dataset,
        output_dir=cfg.output_dir,
        video_name=cfg.video_name,
        fps=cfg.fps
    )

if __name__ == "__main__":
    import argparse
    
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="从YAML配置文件插入对象")
    parser.add_argument(
        "--config",
        type=str,
        default="config/insert_object.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    # 加载配置文件
    cfg = load_config(args.config)
    
    # 运行主函数
    main(cfg)