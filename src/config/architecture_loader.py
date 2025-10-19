# src/config/architecture_loader.py
from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import yaml

from src.config.models import (
    CognitiveArchitectureMaster,
    ClusterDataModel,
    NWayDefinitionModel,
)


def load_master_yaml(master_path: str | Path) -> CognitiveArchitectureMaster:
    path = Path(master_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CognitiveArchitectureMaster(**data)


def load_cluster_yaml(cluster_path: str | Path, cluster_name: str) -> ClusterDataModel:
    path = Path(cluster_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    description = raw.get("description", "")
    cognitive_role = raw.get("cognitive_role", "")
    files = raw.get("files")
    total_models = raw.get("total_models")

    nways: list[NWayDefinitionModel] = []
    for key, val in raw.items():
        if isinstance(key, str) and key.startswith("NWAY_") and isinstance(val, dict):
            nways.append(
                NWayDefinitionModel(
                    id=key,
                    title=val.get("title"),
                    models=val.get("models", []) or [],
                    interactions=val.get("interactions", {}) or {},
                    consultant_priority=val.get("consultant_priority", {}) or {},
                    consultant_personas=val.get("consultant_personas", {}) or {},
                    system2_triggers=val.get("system2_triggers", {}) or {},
                    metacognitive_prompts=val.get("metacognitive_prompts", {}) or {},
                    cognitive_mastery=val.get("cognitive_mastery"),
                )
            )

    return ClusterDataModel(
        name=cluster_name,
        description=description,
        cognitive_role=cognitive_role,
        files=files,
        total_models=total_models,
        nways=nways,
    )


def load_full_architecture(master_path: str | Path) -> Dict[str, ClusterDataModel]:
    master = load_master_yaml(master_path)
    base_dir = Path(master_path).parent
    clusters: Dict[str, ClusterDataModel] = {}
    
    # Handle case where clusters references external files
    if master.clusters is not None:
        for cluster_key, ref in master.clusters.items():
            cluster_file = (base_dir / ref.file).resolve()
            clusters[cluster_key.upper()] = load_cluster_yaml(cluster_file, cluster_key.upper())
    else:
        # Handle embedded cluster data directly in master YAML
        master_data = yaml.safe_load(Path(master_path).read_text(encoding="utf-8")) or {}
        for key, cluster_data in master_data.items():
            if isinstance(cluster_data, dict) and key.endswith("_CLUSTER"):
                cluster_name = key.upper()
                
                # Build cluster from embedded data
                description = cluster_data.get("description", "")
                cognitive_role = cluster_data.get("cognitive_role", "") 
                files = cluster_data.get("files")
                total_models = cluster_data.get("total_models")
                
                nways: list[NWayDefinitionModel] = []
                for nway_key, nway_val in cluster_data.items():
                    if isinstance(nway_key, str) and nway_key.startswith("NWAY_") and isinstance(nway_val, dict):
                        nways.append(
                            NWayDefinitionModel(
                                id=nway_key,
                                title=nway_val.get("title"),
                                models=nway_val.get("models", []) or [],
                                interactions=nway_val.get("interactions", {}) or {},
                                consultant_priority=nway_val.get("consultant_priority", {}) or {},
                                consultant_personas=nway_val.get("consultant_personas", {}) or {},
                                system2_triggers=nway_val.get("system2_triggers", {}) or {},
                                metacognitive_prompts=nway_val.get("metacognitive_prompts", {}) or {},
                                cognitive_mastery=nway_val.get("cognitive_mastery"),
                            )
                        )
                
                clusters[cluster_name] = ClusterDataModel(
                    name=cluster_name,
                    description=description,
                    cognitive_role=cognitive_role,
                    files=files,
                    total_models=total_models,
                    nways=nways,
                )
    
    return clusters
