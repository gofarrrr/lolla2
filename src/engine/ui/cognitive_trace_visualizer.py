#!/usr/bin/env python3
"""
METIS Cognitive Reasoning Trace Visualizer - P7.2
Visual representation of cognitive reasoning processes with interactive traces

Provides detailed visualization of reasoning flows, mental model applications,
and decision pathways with full transparency into the cognitive process.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from src.engine.models.data_contracts import (
    MetisDataContract,
    ReasoningStep,
    MentalModelDefinition,
)


class TraceNodeType(str, Enum):
    """Types of nodes in cognitive trace"""

    PROBLEM_INPUT = "problem_input"
    MENTAL_MODEL = "mental_model"
    REASONING_STEP = "reasoning_step"
    DECISION_POINT = "decision_point"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    SYNTHESIS = "synthesis"
    OUTPUT = "output"


class TraceEdgeType(str, Enum):
    """Types of edges in cognitive trace"""

    SEQUENTIAL = "sequential"
    DEPENDENCY = "dependency"
    INFLUENCE = "influence"
    VALIDATION = "validation"
    CONTRADICTION = "contradiction"
    SUPPORT = "support"


class VisualizationStyle(str, Enum):
    """Visualization rendering styles"""

    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    FORCE_DIRECTED = "force_directed"
    TIMELINE = "timeline"
    SANKEY = "sankey"
    MATRIX = "matrix"


@dataclass
class TraceNode:
    """Node in the cognitive reasoning trace"""

    node_id: str
    node_type: TraceNodeType
    label: str
    description: str
    timestamp: datetime

    # Node properties
    confidence: float = 0.0
    importance: float = 0.5
    duration_ms: int = 0

    # Visual properties
    position: Dict[str, float] = field(default_factory=dict)  # x, y coordinates
    size: float = 30.0
    color: str = "#2196F3"
    shape: str = "circle"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    mental_model_ref: Optional[str] = None
    reasoning_step_ref: Optional[str] = None

    # Interactive properties
    expandable: bool = False
    expanded: bool = False
    selectable: bool = True
    highlighted: bool = False


@dataclass
class TraceEdge:
    """Edge connecting nodes in cognitive trace"""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: TraceEdgeType

    # Edge properties
    weight: float = 1.0
    confidence: float = 1.0

    # Visual properties
    style: str = "solid"  # solid, dashed, dotted
    color: str = "#757575"
    width: float = 2.0
    animated: bool = False

    # Labels and metadata
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceCluster:
    """Cluster grouping related nodes"""

    cluster_id: str
    label: str
    node_ids: List[str] = field(default_factory=list)

    # Visual properties
    color: str = "#E3F2FD"
    border_color: str = "#1976D2"
    opacity: float = 0.3

    # Cluster type
    cluster_type: str = "mental_model"  # mental_model, phase, hypothesis_group
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveTrace:
    """Complete cognitive reasoning trace visualization"""

    trace_id: UUID
    engagement_id: UUID
    created_at: datetime

    # Graph components
    nodes: List[TraceNode] = field(default_factory=list)
    edges: List[TraceEdge] = field(default_factory=list)
    clusters: List[TraceCluster] = field(default_factory=list)

    # Visualization settings
    style: VisualizationStyle = VisualizationStyle.HIERARCHICAL
    layout_config: Dict[str, Any] = field(default_factory=dict)

    # Interaction state
    selected_nodes: List[str] = field(default_factory=list)
    focused_cluster: Optional[str] = None
    zoom_level: float = 1.0
    pan_position: Dict[str, float] = field(default_factory=dict)

    # Metadata
    total_duration_ms: int = 0
    reasoning_depth: int = 0
    branch_factor: float = 0.0
    cognitive_complexity: float = 0.0


@dataclass
class TraceAnimation:
    """Animation configuration for trace playback"""

    enabled: bool = True
    speed: float = 1.0  # Playback speed multiplier
    current_step: int = 0
    total_steps: int = 0

    # Animation timeline
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    keyframes: List[Dict[str, Any]] = field(default_factory=list)

    # Playback state
    playing: bool = False
    loop: bool = False
    show_trail: bool = True


class CognitiveTraceBuilder:
    """Builds cognitive trace from reasoning steps"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.node_counter = 0
        self.edge_counter = 0

    async def build_trace_from_contract(
        self,
        contract: MetisDataContract,
        style: VisualizationStyle = VisualizationStyle.HIERARCHICAL,
    ) -> CognitiveTrace:
        """Build cognitive trace from data contract"""

        trace = CognitiveTrace(
            trace_id=uuid4(),
            engagement_id=contract.engagement_context.engagement_id,
            created_at=datetime.utcnow(),
            style=style,
        )

        # Create problem input node
        problem_node = await self._create_problem_node(contract)
        trace.nodes.append(problem_node)

        # Create mental model nodes and cluster
        model_nodes, model_cluster = await self._create_mental_model_nodes(
            contract.cognitive_state.selected_mental_models
        )
        trace.nodes.extend(model_nodes)
        if model_cluster:
            trace.clusters.append(model_cluster)

        # Create reasoning step nodes with connections
        reasoning_nodes = await self._create_reasoning_nodes(
            contract.cognitive_state.reasoning_steps, problem_node.node_id, model_nodes
        )
        trace.nodes.extend(reasoning_nodes)

        # Create edges between nodes
        edges = await self._create_trace_edges(
            problem_node,
            model_nodes,
            reasoning_nodes,
            contract.cognitive_state.reasoning_steps,
        )
        trace.edges.extend(edges)

        # Create output/synthesis nodes
        if contract.deliverable_artifacts:
            output_nodes = await self._create_output_nodes(
                contract.deliverable_artifacts, reasoning_nodes
            )
            trace.nodes.extend(output_nodes)

        # Apply layout based on style
        await self._apply_layout(trace)

        # Calculate trace metrics
        await self._calculate_trace_metrics(trace)

        return trace

    async def _create_problem_node(self, contract: MetisDataContract) -> TraceNode:
        """Create the initial problem input node"""

        return TraceNode(
            node_id=f"node_{self._next_node_id()}",
            node_type=TraceNodeType.PROBLEM_INPUT,
            label="Problem Statement",
            description=contract.engagement_context.problem_statement[:200],
            timestamp=contract.engagement_context.created_at,
            confidence=1.0,
            importance=1.0,
            color="#4CAF50",
            shape="hexagon",
            size=40,
            metadata={
                "full_statement": contract.engagement_context.problem_statement,
                "client": contract.engagement_context.client_name,
                "phase": "initiation",
            },
        )

    async def _create_mental_model_nodes(
        self, models: List[MentalModelDefinition]
    ) -> Tuple[List[TraceNode], Optional[TraceCluster]]:
        """Create nodes for mental models"""

        model_nodes = []
        model_node_ids = []

        for model in models:
            node = TraceNode(
                node_id=f"node_{self._next_node_id()}",
                node_type=TraceNodeType.MENTAL_MODEL,
                label=model.name,
                description=model.description,
                timestamp=datetime.utcnow(),
                confidence=model.expected_improvement / 100.0,
                importance=0.8,
                color="#9C27B0",
                shape="diamond",
                size=35,
                mental_model_ref=model.model_id,
                metadata={
                    "category": model.category,
                    "criteria": model.application_criteria,
                    "improvement": model.expected_improvement,
                },
            )
            model_nodes.append(node)
            model_node_ids.append(node.node_id)

        # Create cluster for mental models
        cluster = None
        if model_nodes:
            cluster = TraceCluster(
                cluster_id="cluster_mental_models",
                label="Mental Models Applied",
                node_ids=model_node_ids,
                color="#F3E5F5",
                border_color="#7B1FA2",
                cluster_type="mental_model",
            )

        return model_nodes, cluster

    async def _create_reasoning_nodes(
        self,
        reasoning_steps: List[ReasoningStep],
        problem_node_id: str,
        model_nodes: List[TraceNode],
    ) -> List[TraceNode]:
        """Create nodes for reasoning steps"""

        reasoning_nodes = []

        for i, step in enumerate(reasoning_steps):
            # Find corresponding mental model node
            model_node = None
            for mn in model_nodes:
                if mn.mental_model_ref == step.mental_model_applied:
                    model_node = mn
                    break

            node = TraceNode(
                node_id=f"node_{self._next_node_id()}",
                node_type=TraceNodeType.REASONING_STEP,
                label=f"Step {i+1}: {step.mental_model_applied.replace('_', ' ').title()}",
                description=step.reasoning_text[:150],
                timestamp=step.timestamp,
                confidence=step.confidence_score,
                importance=0.5 + (step.confidence_score * 0.3),
                color=self._confidence_to_color(step.confidence_score),
                shape="rectangle",
                size=30 + (step.confidence_score * 10),
                reasoning_step_ref=step.step_id,
                expandable=True,
                metadata={
                    "full_reasoning": step.reasoning_text,
                    "evidence_sources": step.evidence_sources,
                    "assumptions": step.assumptions_made,
                    "model_applied": step.mental_model_applied,
                },
            )
            reasoning_nodes.append(node)

        return reasoning_nodes

    async def _create_output_nodes(
        self, deliverables: List[Any], reasoning_nodes: List[TraceNode]
    ) -> List[TraceNode]:
        """Create output/synthesis nodes"""

        output_nodes = []

        for deliverable in deliverables:
            node = TraceNode(
                node_id=f"node_{self._next_node_id()}",
                node_type=TraceNodeType.OUTPUT,
                label=f"Output: {deliverable.artifact_type}",
                description="Synthesized deliverable",
                timestamp=deliverable.created_at,
                confidence=0.9,
                importance=0.9,
                color="#FF9800",
                shape="octagon",
                size=35,
                metadata={
                    "artifact_id": str(deliverable.artifact_id),
                    "type": deliverable.artifact_type,
                    "confidence_level": deliverable.confidence_level.value,
                },
            )
            output_nodes.append(node)

        return output_nodes

    async def _create_trace_edges(
        self,
        problem_node: TraceNode,
        model_nodes: List[TraceNode],
        reasoning_nodes: List[TraceNode],
        reasoning_steps: List[ReasoningStep],
    ) -> List[TraceEdge]:
        """Create edges connecting trace nodes"""

        edges = []

        # Connect problem to mental models
        for model_node in model_nodes:
            edge = TraceEdge(
                edge_id=f"edge_{self._next_edge_id()}",
                source_node_id=problem_node.node_id,
                target_node_id=model_node.node_id,
                edge_type=TraceEdgeType.INFLUENCE,
                weight=0.8,
                style="dashed",
                color="#9E9E9E",
                label="applies",
            )
            edges.append(edge)

        # Connect mental models to reasoning steps
        for reasoning_node in reasoning_nodes:
            # Find the mental model this step uses
            step_model = reasoning_node.metadata.get("model_applied")
            for model_node in model_nodes:
                if model_node.mental_model_ref == step_model:
                    edge = TraceEdge(
                        edge_id=f"edge_{self._next_edge_id()}",
                        source_node_id=model_node.node_id,
                        target_node_id=reasoning_node.node_id,
                        edge_type=TraceEdgeType.DEPENDENCY,
                        weight=1.0,
                        confidence=reasoning_node.confidence,
                        style="solid",
                        color="#7B1FA2",
                        animated=True,
                        label="uses",
                    )
                    edges.append(edge)
                    break

        # Connect reasoning steps sequentially
        for i in range(len(reasoning_nodes) - 1):
            edge = TraceEdge(
                edge_id=f"edge_{self._next_edge_id()}",
                source_node_id=reasoning_nodes[i].node_id,
                target_node_id=reasoning_nodes[i + 1].node_id,
                edge_type=TraceEdgeType.SEQUENTIAL,
                weight=1.0,
                style="solid",
                color="#2196F3",
                width=3,
                label="then",
            )
            edges.append(edge)

        return edges

    async def _apply_layout(self, trace: CognitiveTrace):
        """Apply layout algorithm based on visualization style"""

        if trace.style == VisualizationStyle.HIERARCHICAL:
            await self._apply_hierarchical_layout(trace)
        elif trace.style == VisualizationStyle.CIRCULAR:
            await self._apply_circular_layout(trace)
        elif trace.style == VisualizationStyle.TIMELINE:
            await self._apply_timeline_layout(trace)
        else:
            await self._apply_force_directed_layout(trace)

    async def _apply_hierarchical_layout(self, trace: CognitiveTrace):
        """Apply hierarchical layout to nodes"""

        # Group nodes by type/layer
        layers = {
            TraceNodeType.PROBLEM_INPUT: [],
            TraceNodeType.MENTAL_MODEL: [],
            TraceNodeType.REASONING_STEP: [],
            TraceNodeType.OUTPUT: [],
        }

        for node in trace.nodes:
            if node.node_type in layers:
                layers[node.node_type].append(node)

        # Position nodes in layers
        y_spacing = 150
        current_y = 50

        for node_type in [
            TraceNodeType.PROBLEM_INPUT,
            TraceNodeType.MENTAL_MODEL,
            TraceNodeType.REASONING_STEP,
            TraceNodeType.OUTPUT,
        ]:
            nodes_in_layer = layers.get(node_type, [])
            if nodes_in_layer:
                x_spacing = 800 / max(1, len(nodes_in_layer))
                for i, node in enumerate(nodes_in_layer):
                    node.position = {"x": 100 + (i * x_spacing), "y": current_y}
                current_y += y_spacing

    async def _apply_circular_layout(self, trace: CognitiveTrace):
        """Apply circular layout to nodes"""

        import math

        center_x, center_y = 400, 400
        radius = 300

        for i, node in enumerate(trace.nodes):
            angle = (2 * math.pi * i) / len(trace.nodes)
            node.position = {
                "x": center_x + radius * math.cos(angle),
                "y": center_y + radius * math.sin(angle),
            }

    async def _apply_timeline_layout(self, trace: CognitiveTrace):
        """Apply timeline layout based on timestamps"""

        # Sort nodes by timestamp
        sorted_nodes = sorted(trace.nodes, key=lambda n: n.timestamp)

        # Position along timeline
        x_spacing = 800 / max(1, len(sorted_nodes))
        for i, node in enumerate(sorted_nodes):
            # Vary y position by node type for readability
            y_offset = {
                TraceNodeType.PROBLEM_INPUT: 100,
                TraceNodeType.MENTAL_MODEL: 200,
                TraceNodeType.REASONING_STEP: 300,
                TraceNodeType.OUTPUT: 400,
            }.get(node.node_type, 250)

            node.position = {"x": 50 + (i * x_spacing), "y": y_offset}

    async def _apply_force_directed_layout(self, trace: CognitiveTrace):
        """Apply force-directed layout for organic positioning"""

        # Simple force-directed simulation
        import random

        # Initialize random positions
        for node in trace.nodes:
            node.position = {
                "x": random.uniform(100, 700),
                "y": random.uniform(100, 500),
            }

        # Note: In production, would use proper force-directed algorithm
        # like D3.js force simulation or networkx spring layout

    async def _calculate_trace_metrics(self, trace: CognitiveTrace):
        """Calculate cognitive trace metrics"""

        if trace.nodes:
            # Calculate total duration
            timestamps = [n.timestamp for n in trace.nodes]
            if len(timestamps) > 1:
                trace.total_duration_ms = int(
                    (max(timestamps) - min(timestamps)).total_seconds() * 1000
                )

            # Calculate reasoning depth (longest path)
            trace.reasoning_depth = await self._calculate_max_depth(trace)

            # Calculate branch factor (average out-degree)
            trace.branch_factor = len(trace.edges) / max(1, len(trace.nodes))

            # Calculate cognitive complexity
            trace.cognitive_complexity = await self._calculate_complexity(trace)

    async def _calculate_max_depth(self, trace: CognitiveTrace) -> int:
        """Calculate maximum depth of reasoning trace"""

        # Build adjacency list
        adj_list = {node.node_id: [] for node in trace.nodes}
        for edge in trace.edges:
            adj_list[edge.source_node_id].append(edge.target_node_id)

        # Find start nodes (no incoming edges)
        has_incoming = set()
        for edge in trace.edges:
            has_incoming.add(edge.target_node_id)

        start_nodes = [n.node_id for n in trace.nodes if n.node_id not in has_incoming]

        # BFS to find max depth
        max_depth = 0
        for start in start_nodes:
            depth = await self._bfs_depth(start, adj_list)
            max_depth = max(max_depth, depth)

        return max_depth

    async def _bfs_depth(self, start: str, adj_list: Dict[str, List[str]]) -> int:
        """Calculate depth from start node using BFS"""

        from collections import deque

        queue = deque([(start, 0)])
        visited = set()
        max_depth = 0

        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            max_depth = max(max_depth, depth)

            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return max_depth

    async def _calculate_complexity(self, trace: CognitiveTrace) -> float:
        """Calculate cognitive complexity score"""

        # Factors contributing to complexity
        node_factor = len(trace.nodes) * 0.1
        edge_factor = len(trace.edges) * 0.15
        cluster_factor = len(trace.clusters) * 0.2
        depth_factor = trace.reasoning_depth * 0.25
        branch_factor = trace.branch_factor * 0.3

        complexity = min(
            1.0,
            (node_factor + edge_factor + cluster_factor + depth_factor + branch_factor)
            / 5.0,
        )

        return complexity

    def _confidence_to_color(self, confidence: float) -> str:
        """Convert confidence score to color"""
        if confidence >= 0.8:
            return "#4CAF50"  # Green
        elif confidence >= 0.6:
            return "#FFC107"  # Amber
        else:
            return "#F44336"  # Red

    def _next_node_id(self) -> int:
        """Get next node ID"""
        self.node_counter += 1
        return self.node_counter

    def _next_edge_id(self) -> int:
        """Get next edge ID"""
        self.edge_counter += 1
        return self.edge_counter


class CognitiveTraceRenderer:
    """Renders cognitive trace for visualization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def render_to_json(self, trace: CognitiveTrace) -> Dict[str, Any]:
        """Render trace to JSON format for frontend visualization"""

        return {
            "trace_id": str(trace.trace_id),
            "engagement_id": str(trace.engagement_id),
            "created_at": trace.created_at.isoformat(),
            "style": trace.style.value,
            "nodes": [self._node_to_dict(node) for node in trace.nodes],
            "edges": [self._edge_to_dict(edge) for edge in trace.edges],
            "clusters": [self._cluster_to_dict(cluster) for cluster in trace.clusters],
            "metrics": {
                "total_duration_ms": trace.total_duration_ms,
                "reasoning_depth": trace.reasoning_depth,
                "branch_factor": trace.branch_factor,
                "cognitive_complexity": trace.cognitive_complexity,
            },
            "layout_config": trace.layout_config,
            "interaction_state": {
                "selected_nodes": trace.selected_nodes,
                "focused_cluster": trace.focused_cluster,
                "zoom_level": trace.zoom_level,
                "pan_position": trace.pan_position,
            },
        }

    def _node_to_dict(self, node: TraceNode) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            "id": node.node_id,
            "type": node.node_type.value,
            "label": node.label,
            "description": node.description,
            "timestamp": node.timestamp.isoformat(),
            "confidence": node.confidence,
            "importance": node.importance,
            "duration_ms": node.duration_ms,
            "position": node.position,
            "size": node.size,
            "color": node.color,
            "shape": node.shape,
            "metadata": node.metadata,
            "expandable": node.expandable,
            "expanded": node.expanded,
            "selectable": node.selectable,
            "highlighted": node.highlighted,
        }

    def _edge_to_dict(self, edge: TraceEdge) -> Dict[str, Any]:
        """Convert edge to dictionary"""
        return {
            "id": edge.edge_id,
            "source": edge.source_node_id,
            "target": edge.target_node_id,
            "type": edge.edge_type.value,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "style": edge.style,
            "color": edge.color,
            "width": edge.width,
            "animated": edge.animated,
            "label": edge.label,
            "metadata": edge.metadata,
        }

    def _cluster_to_dict(self, cluster: TraceCluster) -> Dict[str, Any]:
        """Convert cluster to dictionary"""
        return {
            "id": cluster.cluster_id,
            "label": cluster.label,
            "node_ids": cluster.node_ids,
            "color": cluster.color,
            "border_color": cluster.border_color,
            "opacity": cluster.opacity,
            "type": cluster.cluster_type,
            "metadata": cluster.metadata,
        }

    async def render_to_svg(self, trace: CognitiveTrace) -> str:
        """Render trace to SVG format"""

        # SVG header
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600" viewBox="0 0 900 600">',
        ]

        # Render clusters as background
        for cluster in trace.clusters:
            svg_parts.append(self._render_cluster_svg(cluster, trace.nodes))

        # Render edges
        for edge in trace.edges:
            svg_parts.append(self._render_edge_svg(edge, trace.nodes))

        # Render nodes
        for node in trace.nodes:
            svg_parts.append(self._render_node_svg(node))

        # Close SVG
        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    def _render_node_svg(self, node: TraceNode) -> str:
        """Render node as SVG element"""

        x = node.position.get("x", 0)
        y = node.position.get("y", 0)

        if node.shape == "circle":
            return f'<circle cx="{x}" cy="{y}" r="{node.size/2}" fill="{node.color}" />'
        elif node.shape == "rectangle":
            return f'<rect x="{x-node.size/2}" y="{y-node.size/2}" width="{node.size}" height="{node.size*0.7}" fill="{node.color}" />'
        elif node.shape == "diamond":
            points = f"{x},{y-node.size/2} {x+node.size/2},{y} {x},{y+node.size/2} {x-node.size/2},{y}"
            return f'<polygon points="{points}" fill="{node.color}" />'
        else:
            return f'<circle cx="{x}" cy="{y}" r="{node.size/2}" fill="{node.color}" />'

    def _render_edge_svg(self, edge: TraceEdge, nodes: List[TraceNode]) -> str:
        """Render edge as SVG line"""

        # Find source and target nodes
        source = next((n for n in nodes if n.node_id == edge.source_node_id), None)
        target = next((n for n in nodes if n.node_id == edge.target_node_id), None)

        if not source or not target:
            return ""

        x1 = source.position.get("x", 0)
        y1 = source.position.get("y", 0)
        x2 = target.position.get("x", 0)
        y2 = target.position.get("y", 0)

        stroke_dasharray = "5,5" if edge.style == "dashed" else ""

        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{edge.color}" stroke-width="{edge.width}" stroke-dasharray="{stroke_dasharray}" />'

    def _render_cluster_svg(self, cluster: TraceCluster, nodes: List[TraceNode]) -> str:
        """Render cluster as SVG group"""

        # Find bounding box of cluster nodes
        cluster_nodes = [n for n in nodes if n.node_id in cluster.node_ids]
        if not cluster_nodes:
            return ""

        min_x = min(n.position.get("x", 0) for n in cluster_nodes) - 20
        max_x = max(n.position.get("x", 0) for n in cluster_nodes) + 20
        min_y = min(n.position.get("y", 0) for n in cluster_nodes) - 20
        max_y = max(n.position.get("y", 0) for n in cluster_nodes) + 20

        width = max_x - min_x
        height = max_y - min_y

        return f'<rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" fill="{cluster.color}" stroke="{cluster.border_color}" opacity="{cluster.opacity}" />'


class InteractiveTraceController:
    """Controls interactive features of cognitive trace"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def handle_node_click(
        self, trace: CognitiveTrace, node_id: str
    ) -> Dict[str, Any]:
        """Handle node click interaction"""

        node = next((n for n in trace.nodes if n.node_id == node_id), None)
        if not node:
            return {"error": "Node not found"}

        # Toggle selection
        if node_id in trace.selected_nodes:
            trace.selected_nodes.remove(node_id)
            node.highlighted = False
        else:
            trace.selected_nodes.append(node_id)
            node.highlighted = True

        # If expandable, toggle expansion
        if node.expandable:
            node.expanded = not node.expanded

        return {
            "node_id": node_id,
            "selected": node_id in trace.selected_nodes,
            "expanded": node.expanded,
            "metadata": node.metadata,
        }

    async def handle_cluster_focus(
        self, trace: CognitiveTrace, cluster_id: str
    ) -> Dict[str, Any]:
        """Handle cluster focus interaction"""

        cluster = next((c for c in trace.clusters if c.cluster_id == cluster_id), None)
        if not cluster:
            return {"error": "Cluster not found"}

        trace.focused_cluster = cluster_id

        # Highlight nodes in cluster
        for node in trace.nodes:
            node.highlighted = node.node_id in cluster.node_ids

        return {
            "cluster_id": cluster_id,
            "focused": True,
            "node_count": len(cluster.node_ids),
        }

    async def handle_zoom(
        self, trace: CognitiveTrace, zoom_delta: float
    ) -> Dict[str, Any]:
        """Handle zoom interaction"""

        new_zoom = trace.zoom_level * (1 + zoom_delta)
        trace.zoom_level = max(0.1, min(5.0, new_zoom))

        return {"zoom_level": trace.zoom_level}

    async def handle_pan(
        self, trace: CognitiveTrace, delta_x: float, delta_y: float
    ) -> Dict[str, Any]:
        """Handle pan interaction"""

        trace.pan_position["x"] = trace.pan_position.get("x", 0) + delta_x
        trace.pan_position["y"] = trace.pan_position.get("y", 0) + delta_y

        return {"pan_position": trace.pan_position}


# Export main classes
__all__ = [
    "CognitiveTraceBuilder",
    "CognitiveTraceRenderer",
    "InteractiveTraceController",
    "CognitiveTrace",
    "TraceNode",
    "TraceEdge",
    "TraceCluster",
    "VisualizationStyle",
    "TraceNodeType",
    "TraceEdgeType",
]
