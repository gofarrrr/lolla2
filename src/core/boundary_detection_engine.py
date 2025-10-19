"""
Boundary Detection Engine - Natural Chunk Boundary Detection
===========================================================

Systematically detects natural chunk boundaries using multiple signal types:
1. Causal seams - where cause-effect relationships change
2. Interface boundaries - system component interactions
3. Temporal boundaries - different timescales
4. Stakeholder boundaries - responsibility/ownership changes
5. Data schema boundaries - information structure changes
6. Decision boundaries - decision point transitions

Based on cognitive science research on natural information boundaries.
"""

import re
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging

from src.integrations.llm.unified_client import UnifiedLLMClient
from src.core.strategic_query_decomposer import ChunkBoundary, BoundaryType

logger = logging.getLogger(__name__)


@dataclass
class BoundarySignal:
    """Represents a detected signal indicating a potential boundary"""

    signal_type: str
    position: int
    strength: float
    context_before: str
    context_after: str
    indicator_text: str
    rationale: str


@dataclass
class BoundaryCluster:
    """Represents a cluster of boundary signals that suggest a strong boundary"""

    start_position: int
    end_position: int
    signals: List[BoundarySignal]
    combined_strength: float
    boundary_types: Set[BoundaryType]
    description: str


class BoundaryDetectionEngine:
    """
    Advanced boundary detection using systematic signal analysis.

    Detects natural chunk boundaries by analyzing multiple types of signals
    that indicate transitions between different logical components.
    """

    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self._initialize_signal_patterns()

        logger.info("🔍 Boundary Detection Engine initialized")

    def _initialize_signal_patterns(self):
        """Initialize patterns for detecting different types of boundary signals"""

        # Causal transition patterns
        self.causal_patterns = [
            # Cause indicators
            r"\b(because|due to|caused by|results from|stems from|originates from)\b",
            r"\b(leads to|results in|causes|triggers|drives|generates)\b",
            r"\b(therefore|thus|consequently|as a result|hence)\b",
            # Effect indicators
            r"\b(impact|effect|consequence|outcome|result)\b",
            # Causal connectors
            r"\b(if.*then|when.*then|given.*therefore)\b",
        ]

        # Temporal transition patterns
        self.temporal_patterns = [
            # Sequential indicators
            r"\b(before|after|during|while|then|next|subsequently)\b",
            r"\b(first|second|third|finally|lastly)\b",
            r"\b(initially|eventually|ultimately|previously)\b",
            # Time horizon indicators
            r"\b(short-term|long-term|immediate|future|near-term)\b",
            r"\b(phase|stage|step|milestone|timeline)\b",
            # Temporal quantities
            r"\b(\d+\s*(days?|weeks?|months?|years?)|quarterly|annually)\b",
        ]

        # Stakeholder transition patterns
        self.stakeholder_patterns = [
            # Responsibility indicators
            r"\b(responsible|owns|manages|decides|approves|executes)\b",
            r"\b(team|department|organization|division|unit)\b",
            r"\b(customer|client|vendor|partner|stakeholder)\b",
            # Role indicators
            r"\b(CEO|CTO|manager|director|analyst|engineer)\b",
            r"\b(board|committee|group|council)\b",
        ]

        # Interface transition patterns
        self.interface_patterns = [
            # System component indicators
            r"\b(system|component|module|service|interface)\b",
            r"\b(input|output|data|information|signal)\b",
            r"\b(integration|connection|interaction|communication)\b",
            # Technical boundaries
            r"\b(API|database|frontend|backend|server|client)\b",
            r"\b(protocol|format|schema|structure)\b",
        ]

        # Decision transition patterns
        self.decision_patterns = [
            # Decision indicators
            r"\b(decide|choose|select|determine|evaluate)\b",
            r"\b(option|alternative|choice|path|approach)\b",
            r"\b(criteria|requirement|constraint|consideration)\b",
            # Decision outcomes
            r"\b(approve|reject|accept|decline|proceed)\b",
            r"\b(go/no-go|yes/no|either.*or)\b",
        ]

        # Data schema transition patterns
        self.data_patterns = [
            # Data structure indicators
            r"\b(data|information|content|document|record)\b",
            r"\b(format|structure|schema|model|template)\b",
            r"\b(field|attribute|property|column|variable)\b",
            # Data flow indicators
            r"\b(process|transform|aggregate|filter|sort)\b",
            r"\b(store|retrieve|update|delete|modify)\b",
        ]

    async def detect_boundaries(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[ChunkBoundary]:
        """
        Main entry point: detect all natural boundaries in the given text.

        Args:
            text: Text to analyze for boundaries
            context: Optional context information

        Returns:
            List of detected chunk boundaries sorted by strength
        """

        # Step 1: Detect individual signals
        signals = self._detect_all_signals(text)

        # Step 2: Cluster signals into coherent boundaries
        clusters = self._cluster_signals(signals)

        # Step 3: Use LLM for semantic boundary validation
        validated_boundaries = await self._validate_boundaries_with_llm(text, clusters)

        # Step 4: Convert to ChunkBoundary objects and sort by strength
        boundaries = self._convert_to_boundaries(validated_boundaries, text)
        boundaries.sort(key=lambda b: b.strength, reverse=True)

        logger.info(f"🔍 Detected {len(boundaries)} natural boundaries")
        return boundaries

    def _detect_all_signals(self, text: str) -> List[BoundarySignal]:
        """Detect all boundary signals in the text"""
        signals = []

        # Detect causal signals
        signals.extend(
            self._detect_pattern_signals(text, self.causal_patterns, "causal")
        )

        # Detect temporal signals
        signals.extend(
            self._detect_pattern_signals(text, self.temporal_patterns, "temporal")
        )

        # Detect stakeholder signals
        signals.extend(
            self._detect_pattern_signals(text, self.stakeholder_patterns, "stakeholder")
        )

        # Detect interface signals
        signals.extend(
            self._detect_pattern_signals(text, self.interface_patterns, "interface")
        )

        # Detect decision signals
        signals.extend(
            self._detect_pattern_signals(text, self.decision_patterns, "decision")
        )

        # Detect data schema signals
        signals.extend(
            self._detect_pattern_signals(text, self.data_patterns, "data_schema")
        )

        return signals

    def _detect_pattern_signals(
        self, text: str, patterns: List[str], signal_type: str
    ) -> List[BoundarySignal]:
        """Detect signals matching the given patterns"""
        signals = []

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                position = match.start()
                matched_text = match.group()

                # Extract context around the match
                context_before = text[max(0, position - 100) : position].strip()
                context_after = text[
                    position + len(matched_text) : position + len(matched_text) + 100
                ].strip()

                # Calculate signal strength based on context richness
                strength = self._calculate_signal_strength(
                    matched_text, context_before, context_after, signal_type
                )

                signal = BoundarySignal(
                    signal_type=signal_type,
                    position=position,
                    strength=strength,
                    context_before=context_before,
                    context_after=context_after,
                    indicator_text=matched_text,
                    rationale=f"{signal_type.title()} indicator '{matched_text}' suggests boundary",
                )
                signals.append(signal)

        return signals

    def _calculate_signal_strength(
        self,
        matched_text: str,
        context_before: str,
        context_after: str,
        signal_type: str,
    ) -> float:
        """Calculate the strength of a boundary signal based on context"""

        base_strength = 0.5

        # Boost strength based on signal type importance
        type_boosts = {
            "causal": 0.8,  # Causal boundaries are very important
            "decision": 0.7,  # Decision boundaries are important
            "temporal": 0.6,  # Temporal boundaries are moderately important
            "stakeholder": 0.5,  # Stakeholder boundaries are useful
            "interface": 0.6,  # Interface boundaries are moderately important
            "data_schema": 0.4,  # Data schema boundaries are less critical
        }

        strength = base_strength * type_boosts.get(signal_type, 0.5)

        # Boost strength if context shows clear transition
        if self._indicates_clear_transition(context_before, context_after):
            strength += 0.2

        # Boost strength for longer, more specific indicators
        if len(matched_text) > 8:
            strength += 0.1

        return min(1.0, strength)

    def _indicates_clear_transition(
        self, context_before: str, context_after: str
    ) -> bool:
        """Check if the context indicates a clear transition between concepts"""

        # Simple heuristic: look for different keywords/concepts in before vs after
        before_words = set(context_before.lower().split())
        after_words = set(context_after.lower().split())

        # If less than 30% word overlap, likely a transition
        overlap = len(before_words.intersection(after_words))
        total_unique = len(before_words.union(after_words))

        if total_unique == 0:
            return False

        overlap_ratio = overlap / total_unique
        return overlap_ratio < 0.3

    def _cluster_signals(self, signals: List[BoundarySignal]) -> List[BoundaryCluster]:
        """Cluster nearby signals into coherent boundary areas"""

        if not signals:
            return []

        # Sort signals by position
        signals.sort(key=lambda s: s.position)

        clusters = []
        current_cluster_signals = [signals[0]]

        for signal in signals[1:]:
            # If signal is within 50 characters of the last signal in current cluster, add to cluster
            last_position = current_cluster_signals[-1].position
            if signal.position - last_position <= 50:
                current_cluster_signals.append(signal)
            else:
                # Create cluster from current signals and start new cluster
                if current_cluster_signals:
                    cluster = self._create_cluster(current_cluster_signals)
                    clusters.append(cluster)
                current_cluster_signals = [signal]

        # Don't forget the last cluster
        if current_cluster_signals:
            cluster = self._create_cluster(current_cluster_signals)
            clusters.append(cluster)

        return clusters

    def _create_cluster(self, signals: List[BoundarySignal]) -> BoundaryCluster:
        """Create a boundary cluster from a group of signals"""

        start_position = min(s.position for s in signals)
        end_position = max(s.position for s in signals)

        # Combine signal strengths (weighted average)
        total_strength = sum(s.strength for s in signals)
        combined_strength = total_strength / len(signals)

        # Identify boundary types represented
        boundary_types = set()
        for signal in signals:
            if signal.signal_type == "causal":
                boundary_types.add(BoundaryType.CAUSAL)
            elif signal.signal_type == "temporal":
                boundary_types.add(BoundaryType.TEMPORAL)
            elif signal.signal_type == "stakeholder":
                boundary_types.add(BoundaryType.STAKEHOLDER)
            elif signal.signal_type == "interface":
                boundary_types.add(BoundaryType.INTERFACE)
            elif signal.signal_type == "decision":
                boundary_types.add(BoundaryType.DECISION)
            elif signal.signal_type == "data_schema":
                boundary_types.add(BoundaryType.DATA_SCHEMA)

        # Create description
        signal_types = [s.signal_type for s in signals]
        unique_types = list(set(signal_types))
        description = f"Boundary cluster with {', '.join(unique_types)} signals"

        return BoundaryCluster(
            start_position=start_position,
            end_position=end_position,
            signals=signals,
            combined_strength=combined_strength,
            boundary_types=boundary_types,
            description=description,
        )

    async def _validate_boundaries_with_llm(
        self, text: str, clusters: List[BoundaryCluster]
    ) -> List[BoundaryCluster]:
        """Use LLM to validate and refine boundary detection"""

        if not clusters:
            return clusters

        # Prepare context for LLM validation
        cluster_descriptions = []
        for i, cluster in enumerate(clusters):
            context_start = max(0, cluster.start_position - 200)
            context_end = min(len(text), cluster.end_position + 200)
            context = text[context_start:context_end]

            cluster_info = {
                "cluster_id": i,
                "boundary_types": [bt.value for bt in cluster.boundary_types],
                "strength": cluster.combined_strength,
                "context": context,
                "signals": [s.indicator_text for s in cluster.signals],
            }
            cluster_descriptions.append(cluster_info)

        prompt = f"""
        Validate and refine boundary detection in this text analysis.
        
        ORIGINAL TEXT: {text[:500]}...
        
        DETECTED BOUNDARY CLUSTERS: {json.dumps(cluster_descriptions, indent=2)}
        
        For each boundary cluster, assess:
        1. Is this a meaningful natural boundary?
        2. What type of boundary is it (causal, temporal, stakeholder, interface, decision, data_schema)?
        3. How strong is this boundary (0.0-1.0)?
        4. What is the specific reason this represents a boundary?
        
        Return validation results as JSON with validated clusters.
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in text analysis and information structure.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Parse LLM response and update cluster strengths
            validation_result = json.loads(response)
            validated_clusters = []

            for result in validation_result.get("validated_clusters", []):
                cluster_id = result.get("cluster_id", 0)
                if cluster_id < len(clusters):
                    cluster = clusters[cluster_id]

                    # Update strength based on LLM assessment
                    llm_strength = result.get("strength", cluster.combined_strength)
                    cluster.combined_strength = (
                        cluster.combined_strength + llm_strength
                    ) / 2

                    # Only keep clusters that LLM validates as meaningful
                    if (
                        result.get("is_meaningful", True)
                        and cluster.combined_strength > 0.3
                    ):
                        validated_clusters.append(cluster)

            return validated_clusters

        except Exception as e:
            logger.warning(
                f"⚠️ LLM boundary validation failed, using original clusters: {e}"
            )
            return clusters

    def _convert_to_boundaries(
        self, clusters: List[BoundaryCluster], text: str
    ) -> List[ChunkBoundary]:
        """Convert boundary clusters to ChunkBoundary objects"""

        boundaries = []

        for cluster in clusters:
            # Determine primary boundary type (most common in cluster)
            type_counts = {}
            for boundary_type in cluster.boundary_types:
                type_counts[boundary_type] = type_counts.get(boundary_type, 0) + 1

            primary_type = (
                max(type_counts.keys(), key=lambda k: type_counts[k])
                if type_counts
                else BoundaryType.CAUSAL
            )

            # Extract context before and after the boundary
            context_before = text[
                max(0, cluster.start_position - 100) : cluster.start_position
            ].strip()
            context_after = text[
                cluster.end_position : cluster.end_position + 100
            ].strip()

            # Create detailed description
            signal_indicators = [s.indicator_text for s in cluster.signals]
            description = (
                f"Natural boundary detected via {', '.join(signal_indicators)}"
            )

            boundary = ChunkBoundary(
                boundary_type=primary_type,
                description=description,
                strength=cluster.combined_strength,
                rationale=cluster.description,
                before_chunk=context_before,
                after_chunk=context_after,
            )
            boundaries.append(boundary)

        return boundaries

    def analyze_boundary_quality(
        self, boundaries: List[ChunkBoundary]
    ) -> Dict[str, Any]:
        """Analyze the quality of detected boundaries"""

        if not boundaries:
            return {
                "total_boundaries": 0,
                "average_strength": 0.0,
                "boundary_types": {},
                "quality_score": 0.0,
            }

        # Calculate statistics
        total_boundaries = len(boundaries)
        average_strength = sum(b.strength for b in boundaries) / total_boundaries

        # Count boundary types
        boundary_types = {}
        for boundary in boundaries:
            type_name = boundary.boundary_type.value
            boundary_types[type_name] = boundary_types.get(type_name, 0) + 1

        # Calculate quality score based on:
        # 1. Number of boundaries (more boundaries = better chunking)
        # 2. Average strength (stronger boundaries = better quality)
        # 3. Diversity of boundary types (more types = better coverage)

        boundary_count_score = min(
            1.0, total_boundaries / 5.0
        )  # Expect ~5 boundaries for good quality
        strength_score = average_strength
        diversity_score = min(
            1.0, len(boundary_types) / 4.0
        )  # Expect ~4 different types

        quality_score = (boundary_count_score + strength_score + diversity_score) / 3.0

        return {
            "total_boundaries": total_boundaries,
            "average_strength": average_strength,
            "boundary_types": boundary_types,
            "quality_score": quality_score,
            "boundary_count_score": boundary_count_score,
            "strength_score": strength_score,
            "diversity_score": diversity_score,
        }


# Global instance
_boundary_engine: Optional[BoundaryDetectionEngine] = None


def get_boundary_detection_engine() -> BoundaryDetectionEngine:
    """Get or create the global boundary detection engine instance"""
    global _boundary_engine
    if _boundary_engine is None:
        _boundary_engine = BoundaryDetectionEngine()
    return _boundary_engine
