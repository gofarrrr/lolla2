"""Navigator state handlers extracted from the orchestrator."""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any, Awaitable, Callable, Dict

from .models import NavigatorState
from .runtime import HandlerResult, NavigatorRuntime

logger = logging.getLogger(__name__)

HandlerCallable = Callable[[str, Dict[str, Any]], Awaitable[HandlerResult]]


def build_state_handlers(runtime: NavigatorRuntime) -> Dict[NavigatorState, HandlerCallable]:
    """Build mapping of navigator state handlers bound to the runtime."""

    return {
        NavigatorState.INITIAL: partial(handle_initial, runtime),
        NavigatorState.CLARIFYING: partial(handle_clarifying, runtime),
        NavigatorState.CONTEXT_GATHERING: partial(handle_context_gathering, runtime),
        NavigatorState.MODEL_DISCOVERY: partial(handle_model_discovery, runtime),
        NavigatorState.MODEL_SELECTION: partial(handle_model_selection, runtime),
        NavigatorState.MODEL_EXPLANATION: partial(handle_model_explanation, runtime),
        NavigatorState.APPLICATION_DESIGN: partial(handle_application_design, runtime),
        NavigatorState.IMPLEMENTATION_GUIDANCE: partial(handle_implementation_guidance, runtime),
        NavigatorState.VALIDATION_FRAMEWORK: partial(handle_validation_framework, runtime),
        NavigatorState.NEXT_STEPS: partial(handle_next_steps, runtime),
        NavigatorState.COMPLETED: partial(handle_completed, runtime),
    }


async def handle_initial(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle initial welcome and goal understanding."""

    runtime.session.user_goal = message.strip()

    prompt = f"""
    Welcome to the Mental Model Navigator! I'm here to help you discover and apply relevant mental models to your situation.

    I understand you're interested in: "{runtime.session.user_goal}"

    To provide the most relevant mental models and guidance, I need to understand your situation better. 
    Let me ask you a few clarifying questions to ensure I give you exactly what you need.

    First, could you tell me more about the specific context or challenge you're facing? 
    For example:
    - What field or domain are you working in?
    - What specific problem are you trying to solve?
    - What's the scale or scope of this challenge?
    """

    runtime.advance_state()

    return HandlerResult(
        response=prompt.strip(),
        state=runtime.session.state.value,
        suggested_actions=[
            "Describe your specific context or domain",
            "Explain the problem you're trying to solve",
            "Share the scope or scale of your challenge",
        ],
    )


async def handle_clarifying(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle Socratic clarification loop."""

    clarification_prompt = f"""
    You are an expert at Socratic questioning to understand problems deeply. 
    The user is seeking help with: "{runtime.session.user_goal}"

    They just provided this additional context: "{message}"

    Generate 2-3 insightful clarifying questions that will help you understand:
    1. The root cause or core challenge
    2. The desired outcome or success criteria
    3. Any constraints or important context

    Make the questions specific and actionable. End with a summary of what you understand so far.
    """

    try:
        response = await runtime.llm_client.generate_response(
            prompt=clarification_prompt,
            system_prompt="You are a helpful mental model navigator assistant specializing in Socratic questioning.",
        )

        if runtime._is_clarification_sufficient(message):
            runtime.advance_state()
            next_prompt = "\n\nGreat! I think I have a good understanding of your situation. Let me gather some domain-specific context to find the most relevant mental models for you."
            response += next_prompt

        return HandlerResult(
            response=response,
            state=runtime.session.state.value,
            suggested_actions=[
                "Answer the clarifying questions",
                "Provide more specific details",
                "Clarify your desired outcome",
            ],
        )

    except Exception as exc:
        logger.error("Error in clarifying state: %s", exc)
        runtime.advance_state()
        return HandlerResult(
            response="Thank you for that context. Let me now gather some domain-specific information to find the most relevant mental models for your situation.",
            state=runtime.session.state.value,
        )


async def handle_context_gathering(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle domain context gathering."""

    runtime.session.domain_context = message.strip()

    context_prompt = f"""
    Based on your context: "{message}"

    I'm now going to search our mental models knowledge base to find relevant frameworks and models 
    that apply to your situation: "{runtime.session.user_goal}" in the context of "{runtime.session.domain_context}".

    This will help me identify the most powerful mental models that can provide insight into your challenge.
    Give me a moment to search through our comprehensive collection of mental models...
    """

    runtime.advance_state()

    return HandlerResult(
        response=context_prompt.strip(),
        state=runtime.session.state.value,
        metadata={"domain_context": runtime.session.domain_context},
    )


async def handle_model_discovery(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle mental model discovery using enhanced RAG."""

    try:
        search_query = f"{runtime.session.user_goal} {runtime.session.domain_context} mental models frameworks decision making"
        logger.info("🔍 Searching knowledge base with query: %s", search_query)

        search_results = None
        # Prefer structured results if supported to preserve categories/metadata
        if hasattr(runtime.knowledge_service, "search_knowledge_base_structured"):
            try:
                search_results = await runtime.knowledge_service.search_knowledge_base_structured(
                    query=search_query,
                    top_k=8,
                )
            except Exception as exc:
                logger.warning("Structured RAG search failed: %s, falling back to plain search", exc)
                search_results = None
        if search_results is None and hasattr(runtime.knowledge_service, "search_knowledge_base"):
            try:
                search_results = await runtime.knowledge_service.search_knowledge_base(
                    query=search_query,
                    top_k=8,
                )
            except Exception as exc:
                logger.warning("RAG search failed: %s, using fallback", exc)

        if search_results and isinstance(search_results, list) and len(search_results) > 0:
            # If structured, map to expected model dicts directly
            if isinstance(search_results[0], dict) and "title" in search_results[0]:
                models = [
                    {
                        "title": r.get("title", "Unknown Model"),
                        "description": r.get("description", ""),
                        "relevance": r.get("relevance", 0.8),
                        "category": r.get("category"),
                        "source": r.get("source", "rag_knowledge_base"),
                        "content": r.get("content", ""),
                    }
                    for r in search_results
                ]
            else:
                models = runtime._process_rag_results(search_results)
        else:
            models = runtime._get_curated_mental_models()

        runtime.session.metadata["discovered_models"] = models

        discovery_response = f"""
## Mental Model Discovery Results

I've discovered {len(models)} mental models highly relevant to your situation:

{runtime._format_discovered_models_enhanced(models)}

**Selection Criteria**: These models were chosen based on their relevance to "{runtime.session.user_goal}" within your "{runtime.session.domain_context}" context.

**Next Step**: I'll help you select the 3-5 most powerful models for your specific challenge. Which models resonate most with your situation, or would you like me to recommend an optimal selection?
        """

        runtime.advance_state()

        return HandlerResult(
            response=discovery_response.strip(),
            state=runtime.session.state.value,
            metadata={"discovered_models": [m["title"] for m in models]},
            suggested_actions=[
                "Select specific models for deep dive",
                "Request my recommendation for optimal selection",
                "Ask questions about any model",
            ],
        )

    except Exception as exc:
        logger.error("Error in model discovery: %s", exc)
        fallback_models = runtime._get_curated_mental_models()[:5]
        runtime.session.metadata["discovered_models"] = fallback_models
        runtime.advance_state()
        return HandlerResult(
            response=f"I've identified {len(fallback_models)} powerful mental models for your situation. Let me help you select the most applicable ones for your specific challenge.",
            state=runtime.session.state.value,
            metadata={"discovered_models": [m["title"] for m in fallback_models]},
        )

async def handle_model_selection(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle intelligent mental model selection process."""

    try:
        discovered_models = runtime.session.metadata.get("discovered_models", [])

        selection_prompt = f"""
You are an expert mental model advisor. The user has this goal: "{runtime.session.user_goal}"
In this context: "{runtime.session.domain_context}"

Available mental models discovered:
{json.dumps([{"title": m["title"], "description": m["description"], "relevance": m.get("relevance", 0.8)} for m in discovered_models], indent=2)}

User's response about preferences: "{message}"

Select the 3-5 most powerful and complementary mental models for this specific situation. Return ONLY a JSON response with this format:
{{
  "selected_models": [
    {{
      "title": "Model Name",
      "rationale": "Why this model is perfect for their situation",
      "primary_application": "How they'll use it",
      "confidence": 0.95
    }}
  ],
  "selection_reasoning": "Overall explanation of why these models work together"
}}
        """

        selected_models = []
        reasoning = "These models complement each other well."

        try:
            llm_response = await runtime.llm_client.generate_response(
                prompt=selection_prompt,
                system_prompt="You are a mental model selection expert. Always return valid JSON.",
            )
            selection_data = runtime._parse_json_response(llm_response)
            selected_models = selection_data.get("selected_models", [])
            reasoning = selection_data.get("selection_reasoning", reasoning)
        except Exception as llm_exc:
            logger.warning("Model selection LLM call failed: %s", llm_exc)

        if not selected_models:
            user_preferences = runtime._extract_model_preferences(message)
            logger.info("Using heuristic selection, user preferences: %s", user_preferences)

            if user_preferences:
                selected_models = [
                    {
                        "title": model,
                        "rationale": "You highlighted this model as important to your situation.",
                        "primary_application": "Strategic thinking",
                        "confidence": 0.85,
                    }
                    for model in user_preferences
                ]
                reasoning = "Prioritized the models you called out explicitly."
                runtime.session.metadata["selection_source"] = "user_preferences"
            elif discovered_models:
                selected_models = runtime._heuristic_model_selection(discovered_models, message)
                reasoning = "Selected using heuristics based on relevance and context match."
                runtime.session.metadata["selection_source"] = "heuristic"
            else:
                selected_models = runtime._get_fallback_selection()
                reasoning = "Using foundational models to ensure forward progress."
                runtime.session.metadata["selection_source"] = "fallback"
        else:
            runtime.session.metadata["selection_source"] = "llm"

        runtime.session.selected_models = selected_models
        runtime.session.metadata["selection_reasoning"] = reasoning

        selection_response = f"""
## Optimal Mental Model Selection

{runtime._format_selected_models_enhanced(selected_models)}

**Why these models**: {reasoning}

**Next Step**: I'll generate comprehensive explanations tailored to your goal and context. Let me know if you'd like to adjust any selections before we dive deeper.
        """

        runtime.advance_state()

        return HandlerResult(
            response=selection_response.strip(),
            state=runtime.session.state.value,
            metadata={
                "selected_models": [model["title"] for model in selected_models],
                "selection_reasoning": reasoning,
            },
            suggested_actions=[
                "Confirm these selections",
                "Request adjustments",
                "Ask for alternative models",
            ],
        )

    except Exception as exc:
        logger.error("Error in model selection: %s", exc)
        runtime.session.selected_models = runtime._get_fallback_selection()
        runtime.session.metadata["selection_source"] = "fallback_error"
        runtime.advance_state()
        return HandlerResult(
            response="I've selected foundational mental models to ensure we can keep moving. Let's explore them in depth.",
            state=runtime.session.state.value,
            metadata={
                "selected_models": [model["title"] for model in runtime.session.selected_models],
                "selection_reasoning": "Fallback selection due to processing error.",
            },
        )

async def handle_model_explanation(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle generation of comprehensive model explanations."""

    try:
        explanations = await runtime._generate_comprehensive_explanations_with_rag()
        runtime.session.model_explanations = explanations

        formatted_explanations = runtime._format_structured_explanations(explanations)

        runtime.advance_state()

        return HandlerResult(
            response=formatted_explanations,
            state=runtime.session.state.value,
            metadata={
                "explanations_generated": len(explanations),
                "models": [exp.title for exp in explanations],
            },
            suggested_actions=[
                "Confirm understanding",
                "Ask follow-up questions",
                "Request alternative explanations",
            ],
        )

    except Exception as exc:
        logger.error("Error generating explanations: %s", exc)
        runtime.session.model_explanations = []
        runtime.advance_state()
        return HandlerResult(
            response="I've prepared detailed explanations of your selected mental models. Let me create your personalized Mental Model Map to show how they work together.",
            state=runtime.session.state.value,
        )

async def handle_application_design(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle Mental Model Map synthesis and application design."""

    try:
        mental_model_map = await runtime._generate_enhanced_mental_model_map()
        runtime.session.mental_model_map = mental_model_map

        application_response = runtime._format_enhanced_mental_model_map(mental_model_map)

        runtime.advance_state()

        return HandlerResult(
            response=application_response,
            state=runtime.session.state.value,
            metadata={
                "mental_model_map_created": True,
                "structured_content": {
                    "type": "mental_model_map",
                    "core_models_count": len(mental_model_map.core_models),
                    "relationships_count": len(mental_model_map.relationships),
                    "synergies_count": len(mental_model_map.synergies),
                },
                "map_summary": {
                    "models": mental_model_map.core_models,
                    "key_synergies": mental_model_map.synergies[:2],
                    "application_sequence": mental_model_map.application_sequence,
                },
            },
            suggested_actions=[
                "Get implementation roadmap",
                "Explore model synergies",
                "Address potential conflicts",
            ],
        )

    except Exception as exc:
        logger.error("Error creating mental model map: %s", exc)
        runtime.advance_state()
        return HandlerResult(
            response="I've created your Mental Model Map showing how your selected models work together. Let me provide your implementation roadmap.",
            state=runtime.session.state.value,
        )

async def handle_implementation_guidance(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle implementation guidance generation."""

    implementation_prompt = f"""
    Here's your step-by-step implementation guidance:

    {await runtime._generate_implementation_steps()}

    These steps provide a practical roadmap for applying your mental models to achieve your goal.

    Would you like me to help you create a validation framework to measure your progress?
    """

    runtime.advance_state()

    return HandlerResult(
        response=implementation_prompt.strip(),
        state=runtime.session.state.value,
        suggested_actions=[
            "Create validation framework",
            "Clarify specific steps",
            "Add timeline considerations",
        ],
    )

async def handle_validation_framework(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle validation framework creation."""

    validation_prompt = f"""
    Here's a validation framework to measure your progress:

    {await runtime._generate_validation_framework()}

    This framework will help you track the effectiveness of your mental model application.

    Let me provide you with next steps and additional resources to support your journey.
    """

    runtime.advance_state()

    return HandlerResult(
        response=validation_prompt.strip(),
        state=runtime.session.state.value,
        suggested_actions=[
            "Review next steps",
            "Access additional resources",
            "Schedule follow-up planning",
        ],
    )

async def handle_next_steps(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle reflection prompts and final steps."""

    try:
        reflection_prompts = await runtime._generate_structured_reflection_prompts()
        next_steps_response = runtime._format_reflection_and_next_steps(reflection_prompts)
        runtime.advance_state()

        return HandlerResult(
            response=next_steps_response,
            state=runtime.session.state.value,
            metadata={
                "journey_completed": True,
                "structured_content": {
                    "type": "reflection_and_completion",
                    "reflection_prompts_count": len(reflection_prompts.prompts),
                    "action_items_count": len(reflection_prompts.action_items),
                },
                "session_summary": {
                    "goal": runtime.session.user_goal,
                    "models_learned": [m.get("title", "Unknown") for m in runtime.session.selected_models],
                    "has_explanations": len(runtime.session.model_explanations) > 0,
                    "has_mental_map": runtime.session.mental_model_map is not None,
                    "total_conversation_turns": len(runtime.session.conversation_history),
                },
            },
            suggested_actions=[
                "Start new navigation session",
                "Review complete session summary",
                "Export Mental Model Map",
                "Access learning resources",
            ],
        )

    except Exception as exc:
        logger.error("Error in reflection and next steps: %s", exc)
        runtime.advance_state()
        return HandlerResult(
            response="Congratulations! You've completed the Mental Model Navigator journey. You now have a powerful toolkit for approaching complex challenges.",
            state=runtime.session.state.value,
            metadata={"journey_completed": True},
        )

async def handle_completed(runtime: NavigatorRuntime, message: str, context: Dict[str, Any]) -> HandlerResult:
    """Handle completed session state."""

    return HandlerResult(
        response="This session has been completed. Would you like to start a new Mental Model Navigator session to explore different challenges or mental models?",
        state=runtime.session.state.value,
        suggested_actions=[
            "Start new session",
            "Review previous session",
            "Explore model library",
        ],
    )
