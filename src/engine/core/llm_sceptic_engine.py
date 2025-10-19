#!/usr/bin/env python3
"""
LLM Sceptic Engine - Devils Advocate Engine #4 (Hybrid)
Implements LLM-powered deep critique using NWAY_LLM_SCEPTIC_001 framework
Part of the enhanced Devils Advocate system - Project Hybrid Critic
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Import Supabase for N-Way cluster fetching
try:
    from supabase import create_client, Client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase not available - LLM Sceptic Engine will operate in mock mode")

# Import UnifiedLLMClient for deepseek-reasoner calls
try:
    from src.integrations.llm.unified_client import UnifiedLLMClient

    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False
    print("⚠️ UnifiedLLMClient not available - LLM Sceptic Engine disabled")


@dataclass
class ScepticChallenge:
    """Individual challenge from LLM Sceptic analysis"""

    challenge_type: str  # first_principles, assumption_stress, falsifiability, etc.
    challenge_text: str
    severity: float  # 0.0-1.0
    evidence_basis: str
    counter_argument: str
    exploitation_vector: str  # How competitors could exploit this flaw
    mitigation_strategy: str
    metadata: Optional[Dict[str, Any]] = None  # ULTRATHINK: Ensemble metadata


@dataclass
class LLMScepticResult:
    """Complete result from LLM Sceptic analysis"""

    original_analysis: str
    sceptic_challenges: List[ScepticChallenge]
    foundational_assumptions: List[str]  # Identified core assumptions
    falsifiability_criteria: List[str]  # Measurable failure conditions
    second_order_consequences: List[str]  # Unintended negative outcomes
    lollapalooza_bias_combinations: List[str]  # Multiple bias combinations detected
    overall_intellectual_honesty_score: float  # 0.0-1.0
    red_team_exploitation_scenarios: List[str]
    processing_time_ms: float
    confidence_score: float
    reasoning_trace: str  # Complete LLM reasoning (for transparency)


class LLMScepticEngine:
    """
    LLM-Powered Professional Sceptic Engine

    Uses the NWAY_LLM_SCEPTIC_001 cluster to perform deep, creative critique
    of consultant analyses through systematic adversarial reasoning.

    This is the "fourth engine" in the hybrid Devils Advocate system,
    combining the speed of heuristic engines with the depth of LLM analysis.

    ULTRATHINK ENHANCEMENTS:
    - System 2 Persona (proven 13% bias reduction)
    - Temperature ensemble for diversity
    - Feature flags for safe deployment
    """

    def __init__(self, supabase_client=None):
        self.logger = logging.getLogger(__name__)

        # Initialize Supabase for N-Way cluster fetching (hermetic in TEST_FAST)
        test_fast = str(os.getenv("TEST_FAST", "")).lower() in {"1", "true", "yes"}
        if supabase_client:
            self.supabase = supabase_client
        elif SUPABASE_AVAILABLE and not test_fast:
            url = os.getenv("SUPABASE_URL", "https://soztmkgednwjhgzvlzch.supabase.co")
            key = os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNvenRta2dlZG53amhnenZsemNoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDk4MzYxNywiZXhwIjoyMDcwNTU5NjE3fQ.fe-1KftmBOE_sl4uuMrc0P88LWbKqZvCTEa9vimLARQ",
            )
            self.supabase: Client = create_client(url, key)
        else:
            self.supabase = None
            self.logger.warning("Supabase disabled (TEST_FAST) or unavailable - using mock N-Way cluster")

        # Initialize LLM client for deepseek-reasoner calls
        if LLM_CLIENT_AVAILABLE:
            self.llm_client = UnifiedLLMClient()
        else:
            self.llm_client = None
            self.logger.error(
                "UnifiedLLMClient not available - LLM Sceptic Engine disabled"
            )

        # Cache for N-Way cluster data
        self._nway_cluster_cache = None
        self._cache_timestamp = None

        # ULTRATHINK Feature flags for safe deployment - defaults
        self.use_system2_persona = True
        self.use_ensemble = True
        self.track_contradictions = True
        if test_fast:
            # Hermetic: make engine fast and offline
            self.use_ensemble = False

    async def find_creative_flaws(
        self, analysis_text: str, context: Optional[Dict] = None
    ) -> LLMScepticResult:
        """
        Main entry point: Perform deep, creative critique of consultant analysis

        This method implements the complete NWAY_LLM_SCEPTIC_001 protocol:
        1. Fetch N-Way cluster from database
        2. Construct dynamic prompt with persona + instructional cue
        3. Execute LLM call with deepseek-reasoner mode
        4. Parse structured response into critique result

        Args:
            analysis_text: The consultant's analysis to critique
            context: Optional business/engagement context

        Returns:
            LLMScepticResult with comprehensive critique findings
        """
        start_time = datetime.now()

        try:
            # Step 1: Fetch NWAY_LLM_SCEPTIC_001 cluster from database
            nway_cluster = await self._fetch_nway_sceptic_cluster()
            if not nway_cluster:
                # OPERATION AEGIS HYGIENE: attempt to seed a minimal interaction row, then refetch
                try:
                    await self._seed_minimal_sceptic_interaction()
                    nway_cluster = await self._fetch_nway_sceptic_cluster()
                except Exception:
                    nway_cluster = None
                if not nway_cluster:
                    return self._create_fallback_result(
                        analysis_text, "N-Way cluster unavailable"
                    )

            # Step 2: Construct dynamic prompt using cluster's instructional cue
            system_prompt = self._construct_sceptic_prompt(nway_cluster)
            user_prompt = self._format_analysis_for_critique(analysis_text, context)

            # Step 3: Execute LLM call with deepseek-reasoner mode
            if not self.llm_client:
                return self._create_fallback_result(
                    analysis_text, "LLM client unavailable"
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if self.use_ensemble:
                # ULTRATHINK: Temperature ensemble for diversity (3x creativity levels)
                # Use Grok-4-Fast via OpenRouter for all calls per directive
                ensemble_calls = []
                temperatures = [0.3, 0.7, 1.0]
                purposes = [
                    "logical_flaws_detection",
                    "balanced_critique",
                    "creative_issues_discovery",
                ]

                from src.core.unified_context_stream import (
                    get_unified_context_stream,
                    ContextEventType,
                )
                context_stream = get_unified_context_stream()

                for idx, (temp, purpose) in enumerate(zip(temperatures, purposes)):
                    context_stream.add_event(
                        ContextEventType.LLM_PROVIDER_REQUEST,
                        data={
                            "provider": "openrouter",
                            "model": "grok-4-fast",
                            "temperature": temp,
                            "max_tokens": 3500,
                            "system_prompt": "You are Dr. Sarah Chen, a careful, methodical risk auditor with expertise in international business expansion...",
                            "prompt_preview": user_prompt[:200],
                            "analysis_stage": "devils_advocate_ultrathink",
                            "station": "station_6_ultrathink",
                            "ensemble_purpose": purpose,
                            "ensemble_position": idx,
                            "request_timestamp": datetime.now().isoformat(),
                            "operation_type": "temperature_ensemble_critique",
                        },
                        metadata={
                            "agent_contract_id": "llm_sceptic@2.0",
                            "forensic_instrumentation": True,
                        },
                    )

                    ensemble_calls.append(
                        self.llm_client.call_llm(
                            messages=messages,
                            model="grok-4-fast",
                            provider="openrouter",
                            max_tokens=3500,
                            temperature=temp,
                            response_format={"type": "json_object"},
                        )
                    )

                llm_responses = await asyncio.gather(*ensemble_calls)

                # Log each response with temperature metadata
                for i, (response, temp, purpose) in enumerate(
                    zip(llm_responses, temperatures, purposes)
                ):
                    # Attempt to extract metrics if response is an LLMResponse-like object
                    try:
                        tokens_used = getattr(response, "tokens_used", None)
                        latency_ms = getattr(response, "response_time_ms", None)
                    except Exception:
                        tokens_used = None
                        latency_ms = None
                    context_stream.add_event(
                        ContextEventType.LLM_PROVIDER_RESPONSE,
                        data={
                            "provider": "deepseek",
                            "model": "deepseek-reasoner",
                            "temperature": temp,
                            "response_preview": (
                                str(response)[:200] if response else "No response"
                            ),
                            "response_length": len(str(response)) if response else 0,
                            "analysis_stage": "devils_advocate_ultrathink",
                            "station": "station_6_ultrathink",
                            "ensemble_purpose": purpose,
                            "ensemble_position": i,
                            "response_timestamp": datetime.now().isoformat(),
                            "success": response is not None,
                            **({"tokens_used": tokens_used} if tokens_used is not None else {}),
                            **({"response_time_ms": latency_ms} if latency_ms is not None else {}),
                            "operation_type": "temperature_ensemble_critique",
                        },
                        metadata={
                            "agent_contract_id": "llm_sceptic@2.0",
                            "forensic_instrumentation": True,
                        },
                    )
                # Merge ensemble results
                llm_response = await self._merge_ensemble_responses(llm_responses)
            else:
                # Single call using Grok-4-Fast per directive
                llm_response = await self.llm_client.call_llm(
                    messages=messages,
                    model="grok-4-fast",
                    provider="openrouter",
                    max_tokens=3500,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )

            # Step 4: Parse LLM response into structured result
            result = self._parse_llm_response_to_result(
                llm_response, analysis_text, start_time
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time

            self.logger.info(
                f"✅ LLM Sceptic analysis completed in {processing_time:.1f}ms"
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ LLM Sceptic Engine failed: {e}")
            return self._create_error_result(analysis_text, str(e), start_time)

    async def _seed_minimal_sceptic_interaction(self) -> None:
        """Best-effort seed for NWAY_LLM_SCEPTIC_001 when missing."""
        if not self.supabase:
            return
        try:
            from datetime import datetime
            payload = {
                "interaction_id": "NWAY_LLM_SCEPTIC_001",
                "title": "LLM Sceptic Protocol",
                "description": "Seeded minimal sceptic interaction for Operation Aegis",
                "updated_at": datetime.now().isoformat(),
            }
            # Check if exists first, then insert
            exists = (
                self.supabase.table("nway_interactions")
                .select("interaction_id")
                .eq("interaction_id", "NWAY_LLM_SCEPTIC_001")
                .execute()
            )
            if not exists.data:
                self.supabase.table("nway_interactions").insert(payload).execute()
                self.logger.info("🧩 Seeded NWAY_LLM_SCEPTIC_001 minimal row")
            else:
                self.logger.info("🧩 NWAY_LLM_SCEPTIC_001 already present; no seed needed")
        except Exception as e:
            self.logger.warning(f"Failed to seed NWAY_LLM_SCEPTIC_001: {e}")

    async def _fetch_nway_sceptic_cluster(self) -> Optional[Dict]:
        """Fetch NWAY_LLM_SCEPTIC_001 cluster from Supabase database"""

        # Check cache first (cache for 1 hour)
        if (
            self._nway_cluster_cache
            and self._cache_timestamp
            and (datetime.now() - self._cache_timestamp).total_seconds() < 3600
        ):
            return self._nway_cluster_cache

        if not self.supabase:
            return self._get_mock_nway_cluster()

        try:
            result = (
                self.supabase.table("nway_interactions")
                .select("*")
                .eq("interaction_id", "NWAY_LLM_SCEPTIC_001")
                .execute()
            )

            if result.data and len(result.data) > 0:
                cluster_data = result.data[0]
                self._nway_cluster_cache = cluster_data
                self._cache_timestamp = datetime.now()
                self.logger.info("✅ Loaded NWAY_LLM_SCEPTIC_001 from database")
                return cluster_data
            else:
                self.logger.warning(
                    "⚠️ NWAY_LLM_SCEPTIC_001 not found in database, using mock"
                )
                return self._get_mock_nway_cluster()

        except Exception as e:
            self.logger.error(f"❌ Error fetching N-Way cluster: {e}")
            return self._get_mock_nway_cluster()

    async def _merge_ensemble_responses(self, responses: List[Any]) -> Any:
        """
        ULTRATHINK: Merge ensemble responses from multiple temperatures

        This merges the responses from different temperature settings to get
        the best diversity of challenges while avoiding duplication.
        """

        try:
            # Parse all responses to get structured data
            parsed_responses = []
            for i, response in enumerate(responses):
                try:
                    response_text = response.content
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1

                    if json_start != -1 and json_end > 0:
                        json_text = response_text[json_start:json_end]
                        parsed_data = json.loads(json_text)
                        parsed_data["_temperature"] = [0.3, 0.7, 1.0][
                            i
                        ]  # Track temperature
                        parsed_responses.append(parsed_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse ensemble response {i}: {e}")
                    continue

            if not parsed_responses:
                # Fallback to first response if parsing fails
                return responses[0]

            # Merge the parsed responses
            merged_data = {
                "foundational_assumptions": [],
                "assumption_challenges": [],
                "falsifiability_criteria": [],
                "red_team_exploitations": [],
                "second_order_consequences": [],
                "lollapalooza_effects": [],
                "overall_intellectual_honesty_score": 0.0,
                "key_concerns_summary": "",
            }

            # Aggregate all challenges with vote counting
            challenge_votes = {}
            total_honesty_score = 0.0
            summary_parts = []

            for response in parsed_responses:
                temp = response.get("_temperature", 0.7)

                # Collect foundational assumptions
                for assumption in response.get("foundational_assumptions", []):
                    if assumption not in merged_data["foundational_assumptions"]:
                        merged_data["foundational_assumptions"].append(assumption)

                # Collect assumption challenges with vote counting
                for challenge in response.get("assumption_challenges", []):
                    challenge_key = challenge.get("assumption", "")
                    if challenge_key not in challenge_votes:
                        challenge_votes[challenge_key] = {
                            "challenge": challenge,
                            "votes": 0,
                            "temperatures": [],
                        }
                    challenge_votes[challenge_key]["votes"] += 1
                    challenge_votes[challenge_key]["temperatures"].append(temp)

                # Collect other fields
                merged_data["falsifiability_criteria"].extend(
                    response.get("falsifiability_criteria", [])
                )
                merged_data["red_team_exploitations"].extend(
                    response.get("red_team_exploitations", [])
                )
                merged_data["second_order_consequences"].extend(
                    response.get("second_order_consequences", [])
                )
                merged_data["lollapalooza_effects"].extend(
                    response.get("lollapalooza_effects", [])
                )

                # Aggregate honesty scores
                total_honesty_score += response.get(
                    "overall_intellectual_honesty_score", 0.5
                )
                summary_parts.append(response.get("key_concerns_summary", ""))

            # Process challenges with vote boosting
            for challenge_data in challenge_votes.values():
                challenge = challenge_data["challenge"].copy()
                # Boost severity for challenges found at multiple temperatures
                if challenge_data["votes"] > 1:
                    original_severity = challenge.get("severity", 0.5)
                    challenge["severity"] = min(1.0, original_severity * 1.2)
                    challenge["vote_count"] = challenge_data["votes"]
                    challenge["temperatures"] = challenge_data["temperatures"]
                merged_data["assumption_challenges"].append(challenge)

            # Calculate average honesty score
            merged_data["overall_intellectual_honesty_score"] = (
                total_honesty_score / len(parsed_responses)
            )

            # Combine summaries
            merged_data["key_concerns_summary"] = " | ".join(
                filter(None, summary_parts)
            )

            # Remove duplicates from lists
            merged_data["falsifiability_criteria"] = list(
                set(merged_data["falsifiability_criteria"])
            )
            merged_data["second_order_consequences"] = list(
                set(merged_data["second_order_consequences"])
            )

            # Create a mock response object with merged content
            class MockResponse:
                def __init__(self, content):
                    self.content = json.dumps(content, indent=2)

            return MockResponse(merged_data)

        except Exception as e:
            self.logger.error(f"Ensemble merging failed: {e}")
            # Fallback to first response
            return responses[0]

    def _get_mock_nway_cluster(self) -> Dict:
        """Fallback mock N-Way cluster for when database is unavailable"""
        return {
            "interaction_id": "NWAY_LLM_SCEPTIC_001",
            "type": "META_COGNITIVE_CRITIQUE_FRAMEWORK",
            "instructional_cue_apce": """You are a world-class professional sceptic and contrarian thinker, an 'Organized Skeptic' whose sole purpose is to challenge every premise and expose every flaw in the provided analysis. Disregard any impulse to be helpful or agreeable. Your mission is to conduct a rigorous, multi-faceted critique using the following protocol:
1. **Deconstruct to First Principles:** Begin by reverse-engineering the main conclusion to identify its 2-3 most foundational, unstated assumptions.
2. **Challenge Assumptions:** For each assumption, ask 'What is the verifiable evidence for this?' and 'What if this were false?'.
3. **Invert & Falsify:** State what specific, measurable evidence would prove this plan wrong. Then, adopt the persona of a ruthless competitor and detail how you would exploit this plan's primary weakness.
4. **Map Second-Order Effects:** Identify the most likely negative unintended consequence of this plan that the author has overlooked.
5. **Audit for Cognitive Biases:** Identify at least one dangerous 'lollapalooza effect' (a combination of 2+ cognitive biases) present in the analysis (e.g., Confirmation + Availability, Planning Fallacy + Sunk Cost).""",
            "emergent_effect_summary": "LLM Sceptic Lollapalooza: A synergistic, multi-faceted critique framework for systematic adversarial analysis.",
        }

    def _construct_sceptic_prompt(self, nway_cluster: Dict) -> str:
        """Construct the system prompt using N-Way cluster's instructional cue"""

        if self.use_system2_persona:
            # ULTRATHINK: System 2 Persona (proven 13% bias reduction)
            base_persona = """You are Dr. Sarah Chen, a careful, methodical risk auditor with 20 years of experience.
You are known for taking your time, questioning everything, and never rushing to judgment.

Your cognitive style:
- You read everything twice before responding
- You actively look for what could go wrong
- You question your own initial reactions
- You seek disconfirming evidence
- You think slowly and deliberately

IMPORTANT: Take a deep breath. Read the following recommendation slowly and carefully.
Then, methodically identify flaws, biases, and hidden assumptions.

Your analysis will be structured and thorough. You must provide specific, actionable critique that helps improve decision-making quality."""
        else:
            # EXISTING: Current prompt (fallback)
            base_persona = """You are an elite professional sceptic and contrarian analyst. You specialize in finding flaws, exposing assumptions, and challenging conventional thinking through systematic adversarial reasoning.

Your analysis will be structured and thorough. You must provide specific, actionable critique that helps improve decision-making quality."""

        instructional_cue = nway_cluster.get("instructional_cue_apce", "")

        structured_output_format = """
Your response must be a valid JSON object with this exact structure:
{
    "foundational_assumptions": ["assumption 1", "assumption 2", "assumption 3"],
    "assumption_challenges": [
        {
            "assumption": "specific assumption text",
            "challenge": "why this assumption is questionable",
            "evidence_required": "what evidence would validate this",
            "severity": 0.8
        }
    ],
    "falsifiability_criteria": ["measurable condition 1", "measurable condition 2"],
    "red_team_exploitations": [
        {
            "weakness": "specific weakness identified", 
            "exploitation_method": "how competitors would exploit this",
            "impact_severity": 0.7
        }
    ],
    "second_order_consequences": ["unintended consequence 1", "unintended consequence 2"],
    "lollapalooza_effects": [
        {
            "bias_combination": "Bias1 + Bias2",
            "explanation": "how these biases combine to create flawed reasoning",
            "severity": 0.6
        }
    ],
    "overall_intellectual_honesty_score": 0.65,
    "key_concerns_summary": "Brief summary of the most critical issues found"
}

Ensure all numerical scores are between 0.0 and 1.0, where higher scores indicate more severe problems.
"""

        return f"{base_persona}\n\n{instructional_cue}\n\n{structured_output_format}"

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        from enum import Enum

        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            # Convert dataclass or object to dict
            return self._make_json_serializable(obj.__dict__)
        else:
            # Return as-is for primitives (str, int, float, bool, None)
            return obj

    def _format_analysis_for_critique(
        self, analysis_text: str, context: Optional[Dict] = None
    ) -> str:
        """Format the analysis text with context for LLM critique"""

        context_info = ""
        if context:
            # Convert enums and other non-serializable objects to strings
            serializable_context = self._make_json_serializable(context)
            context_info = (
                f"\n\nBUSINESS CONTEXT:\n{json.dumps(serializable_context, indent=2)}\n"
            )

        return f"""Please perform a comprehensive sceptical analysis of the following business analysis/recommendation:

ANALYSIS TO CRITIQUE:
{analysis_text}
{context_info}

Apply your systematic critique protocol to identify flaws, challenge assumptions, and expose potential risks. Focus on intellectual honesty and strategic vulnerability detection."""

    def _parse_llm_response_to_result(
        self, llm_response, original_analysis: str, start_time: datetime
    ) -> LLMScepticResult:
        """Parse LLM response into structured LLMScepticResult"""

        try:
            # Try to parse JSON response
            response_text = llm_response.content

            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_text = response_text[json_start:json_end]
            parsed_data = json.loads(json_text)

            # Convert parsed data to ScepticChallenge objects
            sceptic_challenges = []

            # Process assumption challenges
            for challenge in parsed_data.get("assumption_challenges", []):
                # ULTRATHINK: Handle ensemble metadata (vote_count, temperatures)
                vote_count = challenge.get("vote_count", 1)
                temperatures = challenge.get("temperatures", [0.7])

                challenge_obj = ScepticChallenge(
                    challenge_type="assumption_stress",
                    challenge_text=challenge.get("challenge", ""),
                    severity=challenge.get("severity", 0.5),
                    evidence_basis=challenge.get("evidence_required", ""),
                    counter_argument=challenge.get("assumption", ""),
                    exploitation_vector="",
                    mitigation_strategy=f"Validate assumption: {challenge.get('evidence_required', '')}",
                )

                # Add ensemble metadata
                challenge_obj.metadata = {
                    "vote_count": vote_count,
                    "temperatures": temperatures,
                }

                sceptic_challenges.append(challenge_obj)

            # Process red team exploitations
            for exploit in parsed_data.get("red_team_exploitations", []):
                sceptic_challenges.append(
                    ScepticChallenge(
                        challenge_type="red_team_exploitation",
                        challenge_text=exploit.get("weakness", ""),
                        severity=exploit.get("impact_severity", 0.5),
                        evidence_basis="Competitive analysis",
                        counter_argument="",
                        exploitation_vector=exploit.get("exploitation_method", ""),
                        mitigation_strategy=f"Address vulnerability: {exploit.get('weakness', '')}",
                    )
                )

            # Build lollapalooza effects list
            lollapalooza_effects = []
            for effect in parsed_data.get("lollapalooza_effects", []):
                lollapalooza_effects.append(
                    f"{effect.get('bias_combination', '')}: {effect.get('explanation', '')}"
                )

            # Construct comprehensive result
            return LLMScepticResult(
                original_analysis=original_analysis,
                sceptic_challenges=sceptic_challenges,
                foundational_assumptions=parsed_data.get(
                    "foundational_assumptions", []
                ),
                falsifiability_criteria=parsed_data.get("falsifiability_criteria", []),
                second_order_consequences=parsed_data.get(
                    "second_order_consequences", []
                ),
                lollapalooza_bias_combinations=lollapalooza_effects,
                overall_intellectual_honesty_score=1.0
                - parsed_data.get(
                    "overall_intellectual_honesty_score", 0.5
                ),  # Invert for consistency
                red_team_exploitation_scenarios=[
                    exploit.get("exploitation_method", "")
                    for exploit in parsed_data.get("red_team_exploitations", [])
                ],
                processing_time_ms=0.0,  # Will be set by caller
                confidence_score=0.85,  # LLM-based analysis has high confidence
                reasoning_trace=response_text,  # Full LLM response for transparency
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to parse LLM response: {e}")
            # Return basic result with raw response
            return LLMScepticResult(
                original_analysis=original_analysis,
                sceptic_challenges=[
                    ScepticChallenge(
                        challenge_type="parsing_error",
                        challenge_text=f"Failed to parse LLM response: {e}",
                        severity=0.3,
                        evidence_basis="System error",
                        counter_argument="",
                        exploitation_vector="",
                        mitigation_strategy="Check LLM response format",
                    )
                ],
                foundational_assumptions=[],
                falsifiability_criteria=[],
                second_order_consequences=[],
                lollapalooza_bias_combinations=[],
                overall_intellectual_honesty_score=0.5,
                red_team_exploitation_scenarios=[],
                processing_time_ms=0.0,
                confidence_score=0.2,  # Low confidence due to parsing error
                reasoning_trace=(
                    llm_response.content
                    if hasattr(llm_response, "content")
                    else str(llm_response)
                ),
            )

    def _create_fallback_result(
        self, analysis_text: str, reason: str
    ) -> LLMScepticResult:
        """Create a fallback result when the engine cannot operate normally"""
        return LLMScepticResult(
            original_analysis=analysis_text,
            sceptic_challenges=[
                ScepticChallenge(
                    challenge_type="system_limitation",
                    challenge_text=f"LLM Sceptic Engine unavailable: {reason}",
                    severity=0.1,
                    evidence_basis="System status",
                    counter_argument="",
                    exploitation_vector="",
                    mitigation_strategy="Enable LLM Sceptic Engine dependencies",
                )
            ],
            foundational_assumptions=[],
            falsifiability_criteria=[],
            second_order_consequences=[],
            lollapalooza_bias_combinations=[],
            overall_intellectual_honesty_score=0.5,  # Neutral when engine unavailable
            red_team_exploitation_scenarios=[],
            processing_time_ms=1.0,
            confidence_score=0.1,  # Very low confidence for fallback
            reasoning_trace=f"Fallback mode: {reason}",
        )

    def _create_error_result(
        self, analysis_text: str, error_message: str, start_time: datetime
    ) -> LLMScepticResult:
        """Create an error result when the engine encounters an exception"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return LLMScepticResult(
            original_analysis=analysis_text,
            sceptic_challenges=[
                ScepticChallenge(
                    challenge_type="engine_error",
                    challenge_text=f"Analysis failed due to engine error: {error_message}",
                    severity=0.2,
                    evidence_basis="System error",
                    counter_argument="",
                    exploitation_vector="",
                    mitigation_strategy="Review engine configuration and dependencies",
                )
            ],
            foundational_assumptions=[],
            falsifiability_criteria=[],
            second_order_consequences=[],
            lollapalooza_bias_combinations=[],
            overall_intellectual_honesty_score=0.5,  # Neutral when error occurs
            red_team_exploitation_scenarios=[],
            processing_time_ms=processing_time,
            confidence_score=0.1,  # Very low confidence due to error
            reasoning_trace=f"Engine error: {error_message}",
        )

    # ULTRATHINK: Feature flag configuration methods
    def enable_system2_persona(self, enabled: bool = True):
        """Enable/disable System 2 persona (proven 13% bias reduction)"""
        self.use_system2_persona = enabled
        self.logger.info(f"System 2 persona: {'ENABLED' if enabled else 'DISABLED'}")

    def enable_temperature_ensemble(self, enabled: bool = True):
        """Enable/disable temperature ensemble for diversity"""
        self.use_ensemble = enabled
        self.logger.info(
            f"Temperature ensemble: {'ENABLED' if enabled else 'DISABLED'}"
        )

    def enable_contradiction_tracking(self, enabled: bool = True):
        """Enable/disable contradiction tracking"""
        self.track_contradictions = enabled
        self.logger.info(
            f"Contradiction tracking: {'ENABLED' if enabled else 'DISABLED'}"
        )

    def configure_ultrathink_features(
        self,
        system2_persona: bool = True,
        temperature_ensemble: bool = True,
        contradiction_tracking: bool = True,
    ):
        """Configure all ULTRATHINK features at once"""
        self.enable_system2_persona(system2_persona)
        self.enable_temperature_ensemble(temperature_ensemble)
        self.enable_contradiction_tracking(contradiction_tracking)

        enabled_features = []
        if system2_persona:
            enabled_features.append("System2")
        if temperature_ensemble:
            enabled_features.append("Ensemble")
        if contradiction_tracking:
            enabled_features.append("Contradictions")

        feature_list = ", ".join(enabled_features) if enabled_features else "None"
        self.logger.info(f"🚀 ULTRATHINK features enabled: {feature_list}")

    def get_feature_status(self) -> Dict[str, bool]:
        """Get current status of all ULTRATHINK features"""
        return {
            "system2_persona": self.use_system2_persona,
            "temperature_ensemble": self.use_ensemble,
            "contradiction_tracking": self.track_contradictions,
        }


# Test function for development
async def test_llm_sceptic_engine():
    """Test the LLM Sceptic Engine with sample analysis"""

    engine = LLMScepticEngine()

    sample_analysis = """
    We recommend expanding into the European market immediately. The market opportunity is significant, 
    with 450 million potential customers and growing demand for our product category. Our competitors 
    have not yet established strong footholds, giving us a first-mover advantage. We should invest 
    $50M in the next 18 months to capture 15% market share and achieve $200M in annual revenue by year 3.
    
    The risks are manageable: regulatory compliance costs are well-understood, and our existing product 
    fits the European market perfectly. European customers have shown strong interest in our value proposition 
    through preliminary market research. Success is almost guaranteed given our competitive advantages.
    """

    context = {
        "company_size": "mid_market",
        "industry": "technology",
        "urgency": "high",
        "engagement_id": "test_001",
    }

    print("🧪 Testing LLM Sceptic Engine...")
    result = await engine.find_creative_flaws(sample_analysis, context)

    print(f"📊 Analysis completed in {result.processing_time_ms:.1f}ms")
    print(
        f"🎯 Intellectual Honesty Score: {result.overall_intellectual_honesty_score:.2f}"
    )
    print(f"⚠️ Challenges Found: {len(result.sceptic_challenges)}")
    print(f"🧠 Foundational Assumptions: {len(result.foundational_assumptions)}")
    print(f"🔍 Falsifiability Criteria: {len(result.falsifiability_criteria)}")
    print(f"💭 Second-Order Consequences: {len(result.second_order_consequences)}")
    print(f"🎭 Lollapalooza Effects: {len(result.lollapalooza_bias_combinations)}")

    print("\n🎯 Sample Critique Challenges:")
    for i, challenge in enumerate(result.sceptic_challenges[:3]):
        print(f"   {i+1}. [{challenge.challenge_type}] {challenge.challenge_text}")
        print(f"      Severity: {challenge.severity:.2f}")

    print("\n🤔 Sample Foundational Assumptions:")
    for assumption in result.foundational_assumptions[:3]:
        print(f"   • {assumption}")

    print("\n✅ LLM Sceptic Engine test completed!")


if __name__ == "__main__":
    asyncio.run(test_llm_sceptic_engine())
