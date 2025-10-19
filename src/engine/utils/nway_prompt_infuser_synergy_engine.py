"""
N-Way Prompt Infusion Utility - PROJECT LOLLAPALOOZA SYNERGY ENGINE
====================================================================

This is the ENHANCED version of the N-Way infuser, transformed from a "List Builder"
to an intelligent "Synergy Engine" that creates true emergent cognitive effects.

NEW CAPABILITIES:
- Meta-cognitive analysis of model interactions
- Dynamic synergy detection and conflict resolution
- Emergent meta-directive generation
- A/B testing framework for validation

This implementation achieves the "Lollapalooza Effect" by analyzing relationships
between selected models and synthesizing them into powerful, coherent directives.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from supabase import Client

# Import glass-box evidence collection
from src.engine.adapters.context_stream import UnifiedContextStream  # Migrated, ContextEventType

# Import unified LLM client for meta-cognitive analysis
try:
    from src.integrations.llm.unified_client import UnifiedLLMClient

    LLM_CLIENT_AVAILABLE = True
except ImportError:
    print("⚠️ UnifiedLLMClient not available - synergy analysis will be disabled")
    LLM_CLIENT_AVAILABLE = False
    UnifiedLLMClient = None

# V2 Contracts for enhanced prompt assembly
try:
    from src.contracts.frameworks import V2PromptAssembly, DynamicNWayModel

    V2_CONTRACTS_AVAILABLE = True
except ImportError:
    V2_CONTRACTS_AVAILABLE = False
    V2PromptAssembly = None


@dataclass
class NWayClusterData:
    """
    Data structure for N-Way Interaction cluster information
    """

    interaction_id: str
    cluster_type: str
    models_involved: List[str]
    emergent_effect_summary: str
    instructional_cue_apce: str
    mechanism_description: Optional[str] = None
    synergy_description: Optional[str] = None
    conflict_description: Optional[str] = None


@dataclass
class ModelInteractionAnalysis:
    """
    Result of meta-cognitive analysis of model interactions
    """

    synergy_insight: str
    conflict_insight: str
    meta_directive: str
    confidence_score: float
    analysis_model: str
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class PromptInfusionResult:
    """
    Result of N-Way prompt infusion process with synergy engine enhancements
    """

    original_prompt: Union[str, "V2PromptAssembly"]
    infused_prompt: Union[str, "V2PromptAssembly"]
    applied_clusters: List[str]
    infusion_metadata: Dict[str, Any]
    synergy_analysis: Optional[ModelInteractionAnalysis]
    success: bool
    error_message: Optional[str] = None
    infusion_type: str = "synergy_engine_v1"


class NWayPromptInfuserSynergyEngine:
    """
    SYNERGY ENGINE: Advanced N-Way Prompt Infuser with intelligent model interaction analysis

    This enhanced version transforms the simple concatenation approach into a sophisticated
    synergy engine that:
    1. Analyzes relationships between selected models
    2. Identifies synergies and conflicts
    3. Generates emergent meta-directives
    4. Creates more intelligent, coherent prompts

    The result is achieving the true "Lollapalooza Effect" where the combination of
    models produces insights greater than the sum of their parts.
    """

    def __init__(
        self, supabase_client: Client, llm_client: Optional[UnifiedLLMClient] = None
    ):
        self.supabase = supabase_client
        self.llm_client = llm_client or (
            UnifiedLLMClient() if LLM_CLIENT_AVAILABLE else None
        )
        self.logger = logging.getLogger(__name__)
        self._cluster_cache = {}  # Cache for frequently accessed clusters

        # Synergy Engine configuration
        self.synergy_analysis_enabled = bool(self.llm_client)
        self.analysis_model = "grok-4-fast"  # Default to Grok 4 Fast; client handles reasoning mode heuristics
        self.analysis_timeout_ms = 15000  # 15 second timeout for analysis calls

        if self.synergy_analysis_enabled:
            self.logger.info(
                "✅ SYNERGY ENGINE: Initialized with meta-cognitive analysis capabilities"
            )
        else:
            self.logger.warning(
                "⚠️ SYNERGY ENGINE: LLM client not available - falling back to enhanced list mode"
            )

    async def analyze_model_interactions(
        self,
        core_model: NWayClusterData,
        dynamic_models: List[NWayClusterData],
        context_stream: Optional["UnifiedContextStream"] = None,
    ) -> ModelInteractionAnalysis:
        """
        PHASE 1: Meta-cognitive analysis of model relationships and interactions

        This is the core innovation that transforms the infuser into a synergy engine.
        Uses dedicated LLM call to analyze how the selected models interact, identifying
        synergies, conflicts, and generating an emergent meta-directive.

        Args:
            core_model: The primary consultant's core N-Way model
            dynamic_models: List of additional "booster pack" dynamic models

        Returns:
            ModelInteractionAnalysis with synergy insights and meta-directive
        """
        if not self.synergy_analysis_enabled:
            return ModelInteractionAnalysis(
                synergy_insight="Synergy analysis not available - LLM client missing",
                conflict_insight="Conflict analysis not available - LLM client missing",
                meta_directive="Apply the following mental models in sequence",
                confidence_score=0.0,
                analysis_model="none",
                processing_time_ms=0,
                success=False,
                error_message="LLM client not available for synergy analysis",
            )

        start_time = asyncio.get_event_loop().time() * 1000

        try:
            self.logger.info(
                f"🧠 SYNERGY ENGINE: Analyzing interactions between {len(dynamic_models)} models"
            )

            # Build the meta-analysis prompt
            analysis_prompt = self._build_meta_analysis_prompt(
                core_model, dynamic_models
            )

            # Log LLM provider request for audit trail
            if context_stream:
                context_stream.add_event(
                    "llm_provider_request",
                    {
                        "model_used": self.analysis_model,
                        "purpose": "nway_synergy_meta_analysis",
                        "system_prompt": "You are a meta-cognitive analyst specialized in identifying synergies and conflicts between mental models.",
                        "user_prompt": analysis_prompt,
                        "temperature": None,
                        "max_tokens": None,
                        "consultant_id": "synergy_engine",
                    },
                )

            # Make the meta-cognitive LLM call
            messages = [
                {
                    "role": "system",
                    "content": "You are a meta-cognitive analyst specialized in identifying synergies and conflicts between mental models.",
                },
                {"role": "user", "content": analysis_prompt},
            ]

            response = await self.llm_client.call_llm(
                messages=messages,
                model=self.analysis_model,
                provider="openrouter",  # OPERATION POLISH: Switched from DeepSeek to OpenRouter for reliability
                response_format={"type": "json_object"},
            )

            # PATCH 3: Calculate processing time BEFORE logging for correct order
            processing_time = int((asyncio.get_event_loop().time() * 1000) - start_time)

            # Log LLM provider response for audit trail
            if context_stream:
                context_stream.add_event(
                    "llm_provider_response",
                    {
                        "raw_response": response.content,
                        "completion_tokens": getattr(
                            response, "completion_tokens", None
                        ),
                        "prompt_tokens": getattr(response, "prompt_tokens", None),
                        "actual_cost_usd": getattr(response, "cost", None),
                        "processing_time_ms": processing_time,  # PATCH 3: Use consistent ms format
                        "execution_mode": "synergy_engine_analysis",  # PATCH 3: Add deterministic execution_mode
                        "consultant_id": "synergy_engine",
                        "model_used": self.analysis_model,
                        "operation_type": "nway_meta_analysis",
                    },
                )

            # Parse the structured response
            analysis_data = json.loads(response.content)

            # Extract insights from the analysis
            synergy_insight = analysis_data.get(
                "synergy_insight", "No synergy identified"
            )
            conflict_insight = analysis_data.get(
                "conflict_insight", "No conflicts detected"
            )
            meta_directive = analysis_data.get(
                "meta_directive", "Apply the selected mental models systematically"
            )
            confidence = float(analysis_data.get("confidence_score", 0.5))

            self.logger.info(
                f"✅ SYNERGY ENGINE: Analysis complete in {processing_time}ms with confidence {confidence:.2f}"
            )

            # 🔍 GLASS-BOX EVIDENCE: Record synergy meta-directive
            if context_stream:
                self._record_synergy_meta_directive_evidence(
                    context_stream=context_stream,
                    meta_directive=meta_directive,
                    synergy_insight=synergy_insight,
                    conflict_insight=conflict_insight,
                    confidence_score=confidence,
                    participating_models=[
                        model.interaction_id for model in [core_model] + dynamic_models
                    ],
                    processing_time_ms=processing_time,
                )

            return ModelInteractionAnalysis(
                synergy_insight=synergy_insight,
                conflict_insight=conflict_insight,
                meta_directive=meta_directive,
                confidence_score=confidence,
                analysis_model=(
                    getattr(response.model, "value", response.model)
                    if hasattr(response, "model")
                    else self.analysis_model
                ),
                processing_time_ms=processing_time,
                success=True,
            )

        except Exception as e:
            import traceback

            processing_time = int((asyncio.get_event_loop().time() * 1000) - start_time)
            self.logger.error(f"❌ SYNERGY ENGINE: Meta-analysis failed: {e}")
            self.logger.error(f"🐛 Full traceback: {traceback.format_exc()}")

            # Return fallback analysis
            return ModelInteractionAnalysis(
                synergy_insight="Analysis failed - using standard approach",
                conflict_insight="Unable to detect conflicts - applying models independently",
                meta_directive="Apply the following mental models systematically, paying attention to their individual strengths",
                confidence_score=0.1,
                analysis_model=self.analysis_model,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e),
            )

    def _build_meta_analysis_prompt(
        self, core_model: NWayClusterData, dynamic_models: List[NWayClusterData]
    ) -> str:
        """
        Build the specialized prompt for meta-cognitive analysis of model interactions
        """
        # Collect all model information
        all_models = [core_model] + dynamic_models

        models_description = ""
        for i, model in enumerate(all_models, 1):
            model_type = "CORE MODEL" if model == core_model else f"DYNAMIC MODEL {i-1}"
            models_description += f"""
**{model_type}: {model.interaction_id}**
- Emergent Effect: {model.emergent_effect_summary}
- Instructional Cue: {model.instructional_cue_apce}
- Mechanism: {model.mechanism_description or 'Not specified'}
"""

        return f"""You are a meta-cognitive analyst. Below are several expert mental models that will be applied together in a strategic analysis. Your task is to:

1. **Identify the single most powerful synergistic interaction** between these models. How do they combine to create an effect greater than the sum of their parts?

2. **Identify the single most critical potential conflict or tension** between these models. Where might their directives lead to contradictory approaches or confusion?

3. **Generate a single, overarching 'Meta-Directive'** that the primary AI should follow to optimally leverage the synergies while resolving the conflicts.

**MENTAL MODELS TO ANALYZE:**
{models_description}

**RESPONSE FORMAT:**
Provide your analysis in valid JSON format with exactly these fields:
{{
    "synergy_insight": "Your identification of the most powerful synergistic interaction (2-3 sentences)",
    "conflict_insight": "Your identification of the most critical potential conflict (2-3 sentences)", 
    "meta_directive": "Your emergent meta-directive that leverages synergies and resolves conflicts (2-3 sentences)",
    "confidence_score": 0.85
}}

Focus on creating a meta-directive that transforms these individual models into a coherent, powerful analytical framework."""

    async def infuse_consultant_prompt_with_synergy_engine(
        self,
        original_prompt: str,
        selected_nway_clusters: List[str],
        consultant_id: str,
        context_stream: Optional["UnifiedContextStream"] = None,
    ) -> PromptInfusionResult:
        """
        ENHANCED VERSION: Infuse consultant prompt using the synergy engine approach

        This method replaces the simple list-building with intelligent synergy analysis:
        1. Fetch cluster data
        2. Analyze model interactions for synergies/conflicts
        3. Generate enhanced directive with meta-cognitive insights
        4. Create sophisticated prompt structure
        """
        try:
            if not selected_nway_clusters:
                return PromptInfusionResult(
                    original_prompt=original_prompt,
                    infused_prompt=original_prompt,
                    applied_clusters=[],
                    infusion_metadata={"reason": "No N-Way clusters selected"},
                    synergy_analysis=None,
                    success=True,
                    infusion_type="synergy_engine_v1_no_clusters",
                )

            self.logger.info(
                f"🔥 SYNERGY ENGINE: Processing {len(selected_nway_clusters)} clusters for {consultant_id}"
            )

            # Fetch cluster data
            cluster_data = self.fetch_nway_clusters(selected_nway_clusters)

            if not cluster_data:
                return PromptInfusionResult(
                    original_prompt=original_prompt,
                    infused_prompt=original_prompt,
                    applied_clusters=[],
                    infusion_metadata={"reason": "No cluster data available"},
                    synergy_analysis=None,
                    success=True,
                    infusion_type="synergy_engine_v1_no_data",
                )

            # Emit NWAY_CLUSTER_ACTIVATED evidence for each cluster (Bible: Station 4)
            if context_stream:
                try:
                    for cid, data in cluster_data.items():
                        context_stream.add_event(
                            ContextEventType.NWAY_CLUSTER_ACTIVATED,
                            {
                                "cluster_id": cid,
                                "participating_models": data.models_involved,
                                "instructional_cue_apce": data.instructional_cue_apce,
                                "emergent_effect_summary": data.emergent_effect_summary,
                                "cluster_type": getattr(
                                    data.cluster_type, "value", data.cluster_type
                                ),
                            },
                            metadata={
                                "synergy_engine": "nway_prompt_infuser_v1",
                                "consultant_id": consultant_id,
                            },
                        )
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Could not emit NWAY_CLUSTER_ACTIVATED events: {e}"
                    )

            # Identify core model and dynamic models
            cluster_list = list(cluster_data.values())
            core_model = cluster_list[0]  # First model is treated as core
            dynamic_models = cluster_list[1:] if len(cluster_list) > 1 else []

            # PHASE 1: Analyze model interactions
            synergy_analysis = await self.analyze_model_interactions(
                core_model, dynamic_models, context_stream
            )

            # PHASE 2: Build synergy-enhanced directive section
            synergy_directives = self._build_synergy_enhanced_directives_section(
                cluster_data, synergy_analysis
            )

            # Inject the enhanced directives into the prompt
            infused_prompt = self._inject_directives_into_prompt(
                original_prompt, synergy_directives
            )

            # Build comprehensive metadata
            infusion_metadata = {
                "consultant_id": consultant_id,
                "clusters_applied": len(cluster_data),
                "cluster_types": [
                    getattr(data.cluster_type, "value", data.cluster_type)
                    for data in cluster_data.values()
                ],
                "synergy_engine_enabled": self.synergy_analysis_enabled,
                "synergy_confidence": synergy_analysis.confidence_score,
                "analysis_processing_time_ms": synergy_analysis.processing_time_ms,
                "total_models_involved": len(
                    set(
                        model
                        for data in cluster_data.values()
                        for model in data.models_involved
                    )
                ),
                "directive_length": len(synergy_directives),
                "infusion_success": True,
            }

            self.logger.info(
                f"✅ SYNERGY ENGINE: Enhanced infusion complete for {consultant_id}"
            )

            return PromptInfusionResult(
                original_prompt=original_prompt,
                infused_prompt=infused_prompt,
                applied_clusters=list(cluster_data.keys()),
                infusion_metadata=infusion_metadata,
                synergy_analysis=synergy_analysis,
                success=True,
                infusion_type="synergy_engine_v1",
            )

        except Exception as e:
            self.logger.error(
                f"❌ SYNERGY ENGINE: Error in enhanced infusion for {consultant_id}: {e}"
            )
            return PromptInfusionResult(
                original_prompt=original_prompt,
                infused_prompt=original_prompt,
                applied_clusters=[],
                infusion_metadata={"error": str(e)},
                synergy_analysis=None,
                success=False,
                error_message=str(e),
                infusion_type="synergy_engine_v1_error",
            )

    def _build_synergy_enhanced_directives_section(
        self,
        cluster_data: Dict[str, NWayClusterData],
        synergy_analysis: ModelInteractionAnalysis,
    ) -> str:
        """
        PHASE 2: Build the enhanced cognitive directives section with synergy intelligence

        This completely redesigns the prompt structure to be more sophisticated and coherent:
        1. Meta-Directive (emergent intelligence)
        2. Synergy to Exploit section
        3. Conflict to Resolve section
        4. Component Models (sub-directives)
        """
        if not cluster_data:
            return ""

        # Header with enhanced intelligence indicators
        directives_section = f"""
**🧠 SYNERGY ENGINE: PROPRIETARY COGNITIVE INTELLIGENCE SYSTEM (CRITICAL)**
*Analysis Model: {synergy_analysis.analysis_model} | Confidence: {synergy_analysis.confidence_score:.1%} | Processing: {synergy_analysis.processing_time_ms}ms*

"""

        # SECTION 1: Meta-Directive (The emergent intelligence)
        directives_section += f"""**🎯 PRIMARY META-DIRECTIVE:**
{synergy_analysis.meta_directive}

"""

        # SECTION 2: Synergy to Exploit
        directives_section += f"""**⚡ SYNERGY TO EXPLOIT:**
{synergy_analysis.synergy_insight}

"""

        # SECTION 3: Conflict to Resolve
        if (
            synergy_analysis.conflict_insight
            and "no conflict" not in synergy_analysis.conflict_insight.lower()
        ):
            directives_section += f"""**🔥 CRITICAL TENSION TO RESOLVE:**
{synergy_analysis.conflict_insight}

"""

        # SECTION 4: Component Models (Sub-Directives)
        directives_section += "**📚 COMPONENT COGNITIVE MODELS:**\n"

        for i, (cluster_id, data) in enumerate(cluster_data.items(), 1):
            model_list = (
                ", ".join(data.models_involved)
                if data.models_involved
                else "Multiple Models"
            )

            directive_entry = f"""
**{i}. {cluster_id}**
   - **Function:** {data.emergent_effect_summary}
   - **Application:** {data.instructional_cue_apce}
   - **Models:** {model_list}
"""

            # Add stored synergy/conflict data if available
            if data.synergy_description:
                directive_entry += (
                    f"   - **Known Synergies:** {data.synergy_description}\n"
                )

            if data.conflict_description:
                directive_entry += (
                    f"   - **Known Conflicts:** {data.conflict_description}\n"
                )

            directives_section += directive_entry

        # Application Requirements
        directives_section += """

**🎯 EXECUTION PROTOCOL:**
1. **Follow the Meta-Directive** as your primary operating principle
2. **Exploit the identified synergy** to create insights greater than individual models
3. **Resolve any tensions** between models through synthesis, not abandonment
4. **Reference specific models** throughout your analysis to demonstrate application
5. **Demonstrate emergent intelligence** that shows the models working together

*This is not a checklist - this is an integrated cognitive framework. Your analysis must show these models creating breakthrough insights together.*
"""

        return directives_section

    # Include existing methods from original infuser with enhancements
    def fetch_nway_clusters(self, cluster_ids: List[str]) -> Dict[str, NWayClusterData]:
        """
        Fetch N-Way cluster data from database with local file fallback

        PATCH 2: Added local fallback to ./NWAY/*.txt files when Supabase is unavailable
        """
        try:
            # Check cache first
            cached_clusters = {}
            missing_ids = []

            for cluster_id in cluster_ids:
                if cluster_id in self._cluster_cache:
                    cached_clusters[cluster_id] = self._cluster_cache[cluster_id]
                else:
                    missing_ids.append(cluster_id)

            # Try fetching missing clusters from database
            fetched_clusters = {}
            if missing_ids and self.supabase:
                try:
                    result = (
                        self.supabase.table("nway_interactions")
                        .select(
                            "interaction_id, type, models_involved, emergent_effect_summary, "
                            "instructional_cue_apce, mechanism_description, synergy_description, conflict_description"
                        )
                        .in_("interaction_id", missing_ids)
                        .execute()
                    )

                    for row in result.data:
                        cluster_data = NWayClusterData(
                            interaction_id=row["interaction_id"],
                            cluster_type=(
                                getattr(row["type"], "value", row["type"])
                                if row["type"]
                                else "unknown"
                            ),
                            models_involved=row.get("models_involved", []),
                            emergent_effect_summary=row.get(
                                "emergent_effect_summary", ""
                            ),
                            instructional_cue_apce=row.get(
                                "instructional_cue_apce", ""
                            ),
                            mechanism_description=row.get("mechanism_description"),
                            synergy_description=row.get("synergy_description"),
                            conflict_description=row.get("conflict_description"),
                        )

                        fetched_clusters[cluster_data.interaction_id] = cluster_data
                        # Cache for future use
                        self._cluster_cache[cluster_data.interaction_id] = cluster_data

                    # Remove successfully fetched IDs from missing_ids
                    missing_ids = [
                        id for id in missing_ids if id not in fetched_clusters
                    ]

                except Exception as db_error:
                    self.logger.warning(
                        f"⚠️ Supabase fetch failed: {db_error}, falling back to local files"
                    )

            # PATCH 2: Fallback to local NWAY files for remaining missing IDs
            if missing_ids:
                local_clusters = self._load_nway_clusters_from_local_files(missing_ids)
                fetched_clusters.update(local_clusters)

            # Combine cached and fetched clusters
            all_clusters = {**cached_clusters, **fetched_clusters}

            self.logger.info(
                f"✅ Fetched {len(all_clusters)} N-Way clusters ({len(cached_clusters)} from cache, {len(fetched_clusters)} fetched)"
            )
            return all_clusters

        except Exception as e:
            self.logger.error(f"❌ Error fetching N-Way clusters: {e}")
            # Final fallback: try to load from local files
            return self._load_nway_clusters_from_local_files(cluster_ids)

    def _load_nway_clusters_from_local_files(
        self, cluster_ids: List[str]
    ) -> Dict[str, NWayClusterData]:
        """
        PATCH 2: Load NWAY clusters from local ./NWAY/*.txt files

        This provides a fallback when Supabase is unavailable
        """
        import os

        local_clusters = {}

        try:
            nway_dir = "./NWAY"
            if not os.path.exists(nway_dir):
                self.logger.warning(f"⚠️ NWAY directory not found: {nway_dir}")
                return {}

            for cluster_id in cluster_ids:
                file_path = os.path.join(nway_dir, f"{cluster_id}.txt")

                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()

                        # Create cluster data from file content
                        cluster_data = NWayClusterData(
                            interaction_id=cluster_id,
                            cluster_type="local_file",
                            models_involved=[],
                            emergent_effect_summary=(
                                content[:200] + "..." if len(content) > 200 else content
                            ),
                            instructional_cue_apce=content,
                            mechanism_description=f"Loaded from local file: {file_path}",
                            synergy_description="Local file content",
                            conflict_description="N/A - local file mode",
                        )

                        local_clusters[cluster_id] = cluster_data
                        # Cache for future use
                        self._cluster_cache[cluster_id] = cluster_data

                        self.logger.info(f"📁 Loaded {cluster_id} from local file")

                    except Exception as file_error:
                        self.logger.error(f"❌ Error reading {file_path}: {file_error}")
                else:
                    self.logger.warning(f"⚠️ Local file not found: {file_path}")

            self.logger.info(
                f"📁 Loaded {len(local_clusters)} NWAY clusters from local files"
            )
            return local_clusters

        except Exception as e:
            self.logger.error(f"❌ Error loading from local NWAY files: {e}")
            return {}

    def _inject_directives_into_prompt(
        self, original_prompt: str, nway_directives: str
    ) -> str:
        """
        Inject enhanced directives into the prompt at the optimal location (unchanged logic)
        """
        if not nway_directives.strip():
            return original_prompt

        # Look for common insertion points in consultant prompts
        insertion_points = [
            "Core Strategic Competencies:",
            "Your analysis should:",
            "Please provide:",
            "Instructions:",
            "Methodology:",
            "Framework:",
            "Analysis Requirements:",
        ]

        # Try to find an insertion point
        for insertion_point in insertion_points:
            if insertion_point in original_prompt:
                # Insert before this section
                parts = original_prompt.split(insertion_point, 1)
                if len(parts) == 2:
                    return (
                        parts[0] + nway_directives + "\n\n" + insertion_point + parts[1]
                    )

        # If no insertion point found, append to the beginning after any system prompt
        if original_prompt.startswith("You are"):
            # Find the end of the role description
            lines = original_prompt.split("\n")
            role_end_index = 0
            for i, line in enumerate(lines):
                if (
                    line.strip()
                    and not line.startswith("You are")
                    and not line.startswith("Core")
                ):
                    role_end_index = i
                    break

            if role_end_index > 0:
                role_section = "\n".join(lines[:role_end_index])
                rest_section = "\n".join(lines[role_end_index:])
                return role_section + "\n" + nway_directives + "\n" + rest_section

        # Fallback: append to beginning
        return nway_directives + "\n\n" + original_prompt

    # Backward compatibility methods
    async def infuse_consultant_prompt(
        self,
        original_prompt: str,
        selected_nway_clusters: List[str],
        consultant_id: str,
        context_stream: Optional["UnifiedContextStream"] = None,
    ) -> PromptInfusionResult:
        """
        Backward compatibility wrapper - routes to synergy engine version
        """
        return await self.infuse_consultant_prompt_with_synergy_engine(
            original_prompt, selected_nway_clusters, consultant_id, context_stream
        )

    def get_synergy_engine_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the synergy engine performance
        """
        return {
            "synergy_engine_enabled": self.synergy_analysis_enabled,
            "analysis_model": self.analysis_model,
            "cache_size": len(self._cluster_cache),
            "cached_clusters": list(self._cluster_cache.keys()),
            "llm_client_available": self.llm_client is not None,
            "analysis_timeout_ms": self.analysis_timeout_ms,
        }

    def _record_synergy_meta_directive_evidence(
        self,
        context_stream: "UnifiedContextStream",
        meta_directive: str,
        synergy_insight: str,
        conflict_insight: str,
        confidence_score: float,
        participating_models: List[str],
        processing_time_ms: int,
    ) -> None:
        """Record glass-box evidence of synergy meta-directive generation"""

        evidence_data = {
            "meta_directive": meta_directive,
            "synergy_insight": synergy_insight,
            "conflict_insight": conflict_insight,
            "confidence_score": confidence_score,
            "participating_models": participating_models,
            "instructional_cue_apce": f"Apply meta-directive: {meta_directive}",
            "emergent_effect_summary": f"Synergy: {synergy_insight}. Conflict resolution: {conflict_insight}",
            "model_count": len(participating_models),
            "processing_time_ms": processing_time_ms,
            "synergy_engine_version": "v1",
            "analysis_model": self.analysis_model,
        }

        # Record to UnifiedContextStream
        context_stream.add_event(
            event_type=ContextEventType.SYNERGY_META_DIRECTIVE,
            data=evidence_data,
            metadata={
                "evidence_type": "synergy_directive",
                "audit_level": "complete",
                "trace_id": context_stream.trace_id,
                "synergy_engine": "nway_prompt_infuser_v1",
            },
        )

        self.logger.info(
            f"🔍 Synergy Evidence: Recorded meta-directive with {len(participating_models)} models, confidence {confidence_score:.2f}"
        )


# Factory function for creating synergy engine
def get_nway_synergy_engine(
    supabase_client: Client, llm_client: Optional[UnifiedLLMClient] = None
) -> NWayPromptInfuserSynergyEngine:
    """
    Factory function to create the enhanced synergy engine

    Args:
        supabase_client: Database client for fetching cluster data
        llm_client: Optional LLM client for meta-cognitive analysis

    Returns:
        Fully configured synergy engine instance
    """
    return NWayPromptInfuserSynergyEngine(supabase_client, llm_client)


# Global instance management for backward compatibility
_synergy_engine_instance = None


def get_global_synergy_engine(
    supabase_client: Client = None,
) -> NWayPromptInfuserSynergyEngine:
    """
    Get the global synergy engine instance
    """
    global _synergy_engine_instance

    if _synergy_engine_instance is None:
        if supabase_client is None:
            raise ValueError("supabase_client is required for first initialization")
        _synergy_engine_instance = NWayPromptInfuserSynergyEngine(supabase_client)

    return _synergy_engine_instance


def reset_synergy_engine():
    """Reset the global instance (useful for testing)"""
    global _synergy_engine_instance
    _synergy_engine_instance = None
