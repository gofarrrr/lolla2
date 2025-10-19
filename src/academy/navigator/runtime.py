"""Navigator runtime utilities and helper methods."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.integrations.llm.unified_client import UnifiedLLMClient
from src.services.knowledge_retrieval_service import KnowledgeRetrievalService

from .models import (
    ModelExplanation,
    MentalModelMap,
    NavigatorSession,
    NavigatorState,
    ReflectionPrompts,
)

logger = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """Normalized handler response payload."""

    response: str
    state: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {"response": self.response, "state": self.state}
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.suggested_actions:
            payload["suggested_actions"] = self.suggested_actions
        if self.error:
            payload["error"] = self.error
        return payload


class NavigatorRuntime:
    """Runtime façade exposing orchestrator helper logic for state handlers."""

    def __init__(
        self,
        session: NavigatorSession,
        knowledge_service: KnowledgeRetrievalService,
        llm_client: UnifiedLLMClient,
        state_transitions: Dict[NavigatorState, NavigatorState],
    ) -> None:
        self.session = session
        self.knowledge_service = knowledge_service
        self.llm_client = llm_client
        self._state_transitions = state_transitions

    @property
    def state_transitions(self) -> Dict[NavigatorState, NavigatorState]:
        """Expose configured state transitions."""

        return self._state_transitions

    def advance_state(self) -> None:
        """Public wrapper to advance the navigator state."""

        self._advance_state()

    # Helper methods
    
    def _advance_state(self):
        """Advance to next state in the workflow"""
        next_state = self.state_transitions.get(self.session.state)
        if next_state:
            self.session.state = next_state
            self.session.current_step += 1
            logger.info(f"🔄 Advanced to state: {self.session.state.value} (step {self.session.current_step})")
    
    def _is_clarification_sufficient(self, message: str) -> bool:
        """Determine if we have sufficient clarification to proceed"""
        # Simple heuristic - in production, this could be more sophisticated
        return len(message.split()) > 20 or any(word in message.lower() for word in 
                                               ["specifically", "problem", "challenge", "goal", "outcome", "result"])
    
    def _format_discovered_models(self, models: List[Dict[str, Any]]) -> str:
        """Format discovered models for display"""
        formatted = []
        for i, model in enumerate(models, 1):
            title = model.get("title", "Unknown Model")
            description = model.get("description", "No description available")
            relevance = model.get("relevance", 0) * 100
            formatted.append(f"{i}. **{title}** (Relevance: {relevance:.0f}%)\n   {description}")
        
        return "\n\n".join(formatted)
    
    def _format_selected_models(self, models: List[str]) -> str:
        """Format selected models for display"""
        return "\n".join([f"• {model}" for model in models])
    
    def _format_selected_models_enhanced(self, models: List[Dict[str, Any]]) -> str:
        """Format selected models with enhanced details"""
        formatted = []
        for i, model in enumerate(models, 1):
            title = model.get("title", "Unknown Model")
            rationale = model.get("rationale", "Highly relevant to your situation")
            application = model.get("primary_application", "Strategic thinking")
            confidence = model.get("confidence", 0.8) * 100
            
            formatted.append(f"""
**{i}. {title}** (Confidence: {confidence:.0f}%)
   - **Why Selected**: {rationale}
   - **Primary Use**: {application}
            """.strip())
        
        return "\n\n".join(formatted)
    
    def _extract_model_preferences(self, message: str) -> List[str]:
        """Extract model preferences from user message"""
        # Simple extraction - in production, use NLP/LLM to extract preferences
        common_models = [
            "Systems Thinking", "First Principles Thinking", "80/20 Principle",
            "Opportunity Cost", "Confirmation Bias", "Mental Models", "Framework"
        ]
        
        selected = []
        message_lower = message.lower()
        for model in common_models:
            if model.lower() in message_lower:
                selected.append(model)
        
        # Default selection if none found
        if not selected:
            selected = ["Systems Thinking", "First Principles Thinking", "80/20 Principle"]
        
        return selected[:3]  # Limit to 3 models for focus
    
    async def _generate_model_explanations(self) -> str:
        """Generate detailed explanations of selected models"""
        explanations = []
        for model in self.session.selected_models:
            explanation = f"""
            ### {model}
            
            **Core Concept**: [Detailed explanation of {model}]
            **Application to Your Situation**: How {model} helps with "{self.session.user_goal}"
            **Key Questions to Ask**: Specific questions this model helps you explore
            **Common Pitfalls**: What to watch out for when applying this model
            """
            explanations.append(explanation.strip())
        
        return "\n\n".join(explanations)
    
    async def _generate_application_strategy(self) -> str:
        """Generate application strategy"""
        return f"""
        ### Integrated Application Strategy
        
        1. **Analysis Phase**: Use {self.session.selected_models[0] if self.session.selected_models else 'Systems Thinking'} to map the current situation
        2. **Solution Design**: Apply remaining models to generate solution options
        3. **Decision Making**: Evaluate options using multiple mental model lenses
        4. **Implementation**: Execute with continuous model-based feedback
        
        This strategy leverages the complementary strengths of your selected mental models.
        """
    
    async def _generate_implementation_steps(self) -> str:
        """Generate implementation steps"""
        return f"""
        ### Implementation Roadmap
        
        **Week 1-2**: Foundation Setting
        - Map current situation using your selected mental models
        - Identify key stakeholders and constraints
        
        **Week 3-4**: Solution Development
        - Generate multiple solution options
        - Evaluate using mental model frameworks
        
        **Week 5-6**: Execution Planning
        - Create detailed action plans
        - Set up feedback mechanisms
        
        **Ongoing**: Continuous Application
        - Regular mental model reviews
        - Adjust strategy based on results
        """
    
    async def _generate_validation_framework(self) -> str:
        """Generate validation framework"""
        return f"""
        ### Validation Framework
        
        **Leading Indicators**:
        - Mental model application frequency
        - Quality of decision-making process
        - Stakeholder feedback on approach
        
        **Lagging Indicators**:
        - Achievement of stated goals
        - Problem resolution effectiveness
        - Long-term sustainability of solutions
        
        **Review Schedule**:
        - Weekly: Process refinement
        - Monthly: Outcome assessment
        - Quarterly: Strategy adjustment
        """
    
    async def _generate_next_steps(self) -> str:
        """Generate next steps and resources"""
        selected_model_names = [m.get('title', 'Mental Model') for m in self.session.selected_models]
        return f"""
        1. **Immediate Actions** (Next 48 hours):
           - Review your selected mental models: {', '.join(selected_model_names)}
           - Begin applying the first step of your implementation plan
        
        2. **Short-term Goals** (Next 2 weeks):
           - Practice using your mental models daily
           - Track your progress using the validation framework
        
        3. **Long-term Development** (Next 3 months):
           - Expand your mental model toolkit
           - Share learnings with colleagues or peers
        
        4. **Resources for Continued Learning**:
           - Recommended books on mental models
           - Practice exercises and scenarios
           - Community forums for discussion
        """
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status with enhanced information"""
        return {
            "state": self.session.state.value,
            "current_step": self.session.current_step,
            "total_steps": 11,
            "last_activity": self.session.last_activity.isoformat(),
            "created_at": self.session.created_at.isoformat(),
            "user_goal": self.session.user_goal,
            "domain_context": self.session.domain_context,
            "selected_models": [m.get('title', 'Unknown') for m in self.session.selected_models],
            "has_mental_model_map": self.session.mental_model_map is not None,
            "has_model_explanations": len(self.session.model_explanations) > 0,
            "conversation_turns": len(self.session.conversation_history),
            "progress_summary": {
                "goal_defined": bool(self.session.user_goal),
                "context_gathered": bool(self.session.domain_context),
                "models_selected": len(self.session.selected_models) > 0,
                "explanations_generated": len(self.session.model_explanations) > 0,
                "map_created": self.session.mental_model_map is not None,
                "journey_completed": self.session.state == NavigatorState.COMPLETED
            }
        }
    
    async def reset_session(self):
        """Reset session to initial state"""
        self.session.state = NavigatorState.INITIAL
        self.session.current_step = 1
        self.session.user_goal = ""
        self.session.domain_context = ""
        self.session.selected_models = []
        self.session.conversation_history = []
        self.session.last_activity = datetime.now()
        
        logger.info(f"🔄 Session reset - Session: {self.session.session_id}")
    
    def _process_rag_results(self, rag_results: List[str]) -> List[Dict[str, Any]]:
        """Process RAG results into structured mental model data"""
        discovered_models = []
        
        for result in rag_results:
            try:
                # Extract model name from content (look for common patterns)
                lines = result.split('\n')
                model_name = "Unknown Model"
                description = result[:200] + "..." if len(result) > 200 else result
                
                # Try to extract model name from the content
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['model:', 'framework:', 'principle:', 'law:', 'effect:']):
                        model_name = line.split(':')[-1].strip() if ':' in line else line.strip()
                        break
                    elif line.strip() and len(line.strip()) < 100:  # Likely a title
                        model_name = line.strip()
                        break
                
                discovered_models.append({
                    'title': model_name,
                    'description': description,
                    'source': 'rag_knowledge_base',
                    'relevance': 0.8,  # Default relevance
                    'content': result
                })
            except Exception as e:
                logger.warning(f"Error processing RAG result: {e}")
                continue
                
        return discovered_models
    
    def _get_curated_mental_models(self) -> List[Dict[str, Any]]:
        """Get curated list of essential mental models as fallback"""
        return [
            {
                'title': 'First Principles Thinking',
                'description': 'Breaking down complex problems into fundamental components and building up from there.',
                'source': 'curated_collection',
                'relevance': 0.9,
                'category': 'Problem Solving'
            },
            {
                'title': 'Systems Thinking',
                'description': 'Understanding how different parts of a system interact and influence each other.',
                'source': 'curated_collection', 
                'relevance': 0.85,
                'category': 'Strategic Analysis'
            },
            {
                'title': 'Opportunity Cost',
                'description': 'The value of the best alternative foregone when making a choice.',
                'source': 'curated_collection',
                'relevance': 0.8,
                'category': 'Decision Making'
            },
            {
                'title': 'Confirmation Bias',
                'description': 'The tendency to search for, interpret, and recall information that confirms pre-existing beliefs.',
                'source': 'curated_collection',
                'relevance': 0.75,
                'category': 'Cognitive Bias'
            },
            {
                'title': 'Pareto Principle (80/20 Rule)',
                'description': 'Roughly 80% of effects come from 20% of causes.',
                'source': 'curated_collection',
                'relevance': 0.7,
                'category': 'Prioritization'
            }
        ]
    
    def _format_discovered_models_enhanced(self, models: List[Dict[str, Any]]) -> str:
        """Format discovered models with enhanced presentation"""
        if not models:
            return "No specific mental models were found. Let me provide some fundamental models that apply to most situations."
        
        formatted = "🧠 **DISCOVERED MENTAL MODELS:**\n\n"
        
        for i, model in enumerate(models[:5], 1):  # Limit to top 5
            formatted += f"{i}. **{model['title']}**\n"
            formatted += f"   └ {model['description']}\n"
            if 'category' in model:
                formatted += f"   └ Category: {model['category']}\n"
            formatted += f"   └ Relevance: {model.get('relevance', 0.5):.1%}\n\n"
        
        formatted += "Which of these mental models would you like me to explain in detail, or would you like me to search for more specific models?"
        return formatted
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with fallback handling"""
        try:
            # Try to find JSON within the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON found, create structured response from text
            return {
                'selected_models': [],
                'reasoning': response_text,
                'confidence': 0.5
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                'selected_models': [],
                'reasoning': response_text,
                'confidence': 0.3
            }
    
    def _heuristic_model_selection(self, discovered_models: List[Dict[str, Any]], user_input: str) -> List[Dict[str, Any]]:
        """Heuristic-based model selection as LLM fallback"""
        if not discovered_models:
            discovered_models = self._get_curated_mental_models()
        
        # Simple scoring based on relevance and context keywords
        context_text = f"{self.session.user_goal} {self.session.domain_context} {user_input}".lower()
        
        scored_models = []
        for model in discovered_models:
            score = model.get('relevance', 0.5)
            
            # Boost score based on context matching
            model_text = f"{model['title']} {model['description']}".lower()
            
            # Look for keyword matches
            if any(keyword in context_text for keyword in ['decision', 'choice', 'option']):
                if any(word in model_text for word in ['decision', 'choice', 'opportunity']):
                    score += 0.2
            
            if any(keyword in context_text for keyword in ['system', 'complex', 'interaction']):
                if 'system' in model_text:
                    score += 0.2
                    
            if any(keyword in context_text for keyword in ['bias', 'assumption', 'belief']):
                if any(word in model_text for word in ['bias', 'assumption', 'belief']):
                    score += 0.2
            
            scored_models.append((score, model))
        
        # Select top 3 models and format for selected_models structure
        scored_models.sort(key=lambda x: x[0], reverse=True)
        selected = []
        for score, model in scored_models[:3]:
            selected.append({
                'title': model['title'],
                'rationale': f"Selected based on {score:.1%} relevance to your situation",
                'primary_application': model.get('category', 'Strategic thinking'),
                'confidence': min(score, 1.0)
            })
        return selected
    
    def _get_fallback_selection(self) -> List[Dict[str, Any]]:
        """Get fallback selection when all other methods fail"""
        curated = self._get_curated_mental_models()
        return [{
            'title': model['title'],
            'rationale': 'Essential mental model for strategic thinking',
            'primary_application': model.get('category', 'Problem solving'),
            'confidence': model.get('relevance', 0.8)
        } for model in curated[:3]]  # Return top 3 as fallback
    
    async def _generate_comprehensive_explanations_with_rag(self) -> List[ModelExplanation]:
        """Generate comprehensive explanations for selected models using RAG and LLM"""
        explanations = []
        
        for model in self.session.selected_models:
            try:
                model_title = model['title']
                logger.info(f"📚 Generating RAG-enhanced explanation for: {model_title}")
                
                # Step 1: Search knowledge base for detailed model information
                rag_content = await self._get_model_rag_content(model_title)
                
                # Step 2: Generate structured explanation using LLM + RAG
                structured_explanation = await self._generate_structured_explanation(
                    model, rag_content
                )
                
                explanations.append(structured_explanation)
                
            except Exception as e:
                logger.error(f"Error generating explanation for {model['title']}: {e}")
                # Fallback to basic explanation
                fallback_explanation = ModelExplanation(
                    title=model['title'],
                    core_concept=model.get('rationale', 'A powerful mental model for strategic thinking.'),
                    application=f"This model can help with your goal: {self.session.user_goal}",
                    examples=["Example will be provided based on your specific context"],
                    pitfalls=["Be aware of oversimplification", "Consider multiple perspectives"],
                    key_questions=[
                        "How does this model change your perspective?",
                        "What new insights does it reveal?",
                        "What actions might you take differently?"
                    ],
                    data={"source": "fallback", "model_data": model}
                )
                explanations.append(fallback_explanation)
        
        return explanations
    
    async def _get_model_rag_content(self, model_title: str) -> str:
        """Get detailed content for a mental model from RAG knowledge base"""
        try:
            # Enhanced search query for specific model
            search_query = f'{model_title} mental model definition examples application framework'
            
            if hasattr(self.knowledge_service, 'search_knowledge_base'):
                rag_results = await self.knowledge_service.search_knowledge_base(
                    query=search_query,
                    top_k=3  # Focus on top relevant results
                )
                
                if rag_results and len(rag_results) > 0:
                    # Combine top results into comprehensive content
                    combined_content = "\n\n".join(rag_results[:3])
                    return combined_content
            
            # Fallback to curated model knowledge
            return self._get_curated_model_content(model_title)
            
        except Exception as e:
            logger.warning(f"RAG content retrieval failed for {model_title}: {e}")
            return self._get_curated_model_content(model_title)
    
    def _get_curated_model_content(self, model_title: str) -> str:
        """Get curated content for common mental models"""
        curated_content = {
            "First Principles Thinking": """
First Principles Thinking is a problem-solving technique that involves breaking down complex problems into their most fundamental components and building solutions from the ground up. Instead of reasoning by analogy or accepting conventional wisdom, this approach questions every assumption until you reach the basic truths or first principles.

Key aspects:
- Deconstruction: Break problems into fundamental elements
- Questioning assumptions: Challenge what seems obvious
- Reconstruction: Build solutions from basic truths
- Independent reasoning: Think beyond conventional approaches

Origins: Developed by Aristotle, popularized by Elon Musk and other innovators.

Applications: Product development, business strategy, scientific research, personal decision-making.""",
            
            "Systems Thinking": """
Systems Thinking is a holistic approach to analysis that focuses on understanding the relationships and interactions between components of a complex system, rather than examining individual parts in isolation.

Key principles:
- Holistic perspective: See the whole rather than parts
- Interconnectedness: Understand relationships and dependencies
- Feedback loops: Identify reinforcing and balancing cycles
- Dynamic complexity: Focus on patterns of behavior over time
- Non-linear thinking: Small changes can have big effects

Applications: Organizational change, supply chain management, environmental issues, healthcare systems.""",
            
            "Opportunity Cost": """
Opportunity Cost represents the value of the best alternative that must be forgone when making a choice. Every decision involves trade-offs, and opportunity cost helps quantify what you're giving up.

Core concepts:
- Scarcity: Resources (time, money, attention) are limited
- Trade-offs: Choosing one option means not choosing others
- Value assessment: Compare the benefits of alternatives
- Decision optimization: Choose options with highest net benefit

Applications: Resource allocation, investment decisions, time management, strategic planning."""
        }
        
        return curated_content.get(model_title, f"Mental model: {model_title} - A framework for better thinking and decision-making.")
    
    async def _generate_structured_explanation(self, model: Dict[str, Any], rag_content: str) -> ModelExplanation:
        """Generate structured explanation using LLM with RAG content"""
        try:
            prompt = f"""
You are an expert in mental models. Using the provided information, create a comprehensive explanation of "{model['title']}" for someone working on: "{self.session.user_goal}" in the context of "{self.session.domain_context}".

Source Material:
{rag_content}

User's Goal: {self.session.user_goal}
Domain Context: {self.session.domain_context}

Provide a JSON response with this exact structure:
{{
  "core_concept": "Clear, comprehensive definition and explanation",
  "application": "How this specifically applies to their goal and context",
  "examples": ["Practical example 1", "Practical example 2", "Practical example 3"],
  "pitfalls": ["Common mistake 1", "Common mistake 2", "Common mistake 3"],
  "key_questions": ["Key question 1", "Key question 2", "Key question 3"]
}}

Make it highly practical and specific to their situation.
"""
            
            response = await self.llm_client.generate_response(
                prompt=prompt,
                system_prompt="You are a mental models expert. Always return valid JSON with the requested structure."
            )
            
            # Parse the JSON response
            parsed_data = self._parse_json_response(response)
            
            return ModelExplanation(
                title=model['title'],
                core_concept=parsed_data.get('core_concept', f"Core concept of {model['title']}"),
                application=parsed_data.get('application', f"Application to {self.session.user_goal}"),
                examples=parsed_data.get('examples', ["Context-specific example"]),
                pitfalls=parsed_data.get('pitfalls', ["Avoid oversimplification"]),
                key_questions=parsed_data.get('key_questions', ["How does this change your approach?"]),
                data={"source": "rag_enhanced", "model_data": model, "rag_content_length": len(rag_content)}
            )
            
        except Exception as e:
            logger.warning(f"Structured explanation generation failed for {model['title']}: {e}")
            # Return fallback structured explanation
            return ModelExplanation(
                title=model['title'],
                core_concept=f"{model['title']}: {model.get('rationale', 'A powerful framework for strategic thinking')}",
                application=f"This model helps you approach '{self.session.user_goal}' by providing a structured thinking framework.",
                examples=[
                    f"Apply {model['title']} to analyze your situation",
                    f"Use {model['title']} to generate alternative solutions",
                    f"Evaluate decisions through the {model['title']} lens"
                ],
                pitfalls=[
                    "Don't apply this model in isolation",
                    "Consider the limitations of this framework",
                    "Validate insights with other mental models"
                ],
                key_questions=[
                    f"How does {model['title']} change your perspective on the problem?",
                    "What new insights does this model reveal?",
                    "What actions would you take differently using this framework?"
                ],
                data={"source": "fallback", "model_data": model}
            )
    
    def _format_structured_explanations(self, explanations: List[ModelExplanation]) -> str:
        """Format structured explanations for display"""
        formatted_sections = []
        
        formatted_sections.append("## 🧠 Deep Dive: Your Selected Mental Models\n")
        
        for i, explanation in enumerate(explanations, 1):
            section = f"""
### {i}. {explanation.title}

**🎯 Core Concept:**
{explanation.core_concept}

**🔧 Application to Your Situation:**
{explanation.application}

**📝 Practical Examples:**
{chr(10).join([f"• {example}" for example in explanation.examples])}

**⚠️ Common Pitfalls:**
{chr(10).join([f"• {pitfall}" for pitfall in explanation.pitfalls])}

**🤔 Key Questions to Consider:**
{chr(10).join([f"• {question}" for question in explanation.key_questions])}
"""
            formatted_sections.append(section.strip())
        
        formatted_sections.append("""
## 🔗 Integration Insights

These models work together to provide:
- **Multiple perspectives** on your challenge
- **Complementary analysis frameworks**
- **Synergistic problem-solving approaches**

**Next Step**: Now I'll help you synthesize these models into a unified "Mental Model Map" that shows how they interconnect for maximum impact on your goal.

✨ Ready to create your personalized Mental Model Map?
""")
        
        return "\n\n".join(formatted_sections)
    
    async def _generate_comprehensive_explanations(self) -> str:
        """Legacy method - kept for backward compatibility"""
        explanations = await self._generate_comprehensive_explanations_with_rag()
        return self._format_structured_explanations(explanations)
    
    def _get_fallback_explanation(self, model: Dict[str, Any]) -> str:
        """Generate fallback explanation when LLM is unavailable"""
        return f"""
### {model['title']}

**Core Concept**: {model.get('rationale', 'Mental model for better decision making')}

**Application to Your Situation**: This model can help you analyze "{self.session.user_goal}" by providing a structured framework for thinking through the key factors and relationships involved.

**Key Questions to Consider**:
- How does this model change your perspective on the problem?
- What new insights does it reveal?
- What actions might you take differently as a result?

**Next Steps**: Consider how you might test or validate the insights this model provides in your specific context.
"""
    
    async def _generate_enhanced_mental_model_map(self) -> MentalModelMap:
        """Generate enhanced mental model map with structured output"""
        try:
            model_names = [model['title'] for model in self.session.selected_models]
            logger.info(f"🗺️ Generating enhanced Mental Model Map for: {', '.join(model_names)}")
            
            # Generate comprehensive relationships analysis
            relationships = await self._analyze_comprehensive_relationships()
            
            # Generate enhanced synergies analysis
            synergies = await self._generate_enhanced_synergies()
            
            # Identify potential conflicts
            conflicts = await self._identify_potential_conflicts()
            
            # Generate optimal application sequence
            application_sequence = self._generate_optimal_sequence()
            
            # Generate unified strategy
            unified_strategy = await self._generate_unified_strategy()
            
            return MentalModelMap(
                core_models=model_names,
                relationships=relationships,
                synergies=synergies,
                conflicts=conflicts,
                application_sequence=application_sequence,
                unified_strategy=unified_strategy,
                data={
                    "context": self.session.user_goal,
                    "domain": self.session.domain_context,
                    "generation_method": "enhanced_rag_llm",
                    "models_analyzed": len(self.session.selected_models)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating enhanced mental model map: {e}")
            return MentalModelMap(
                core_models=[model['title'] for model in self.session.selected_models],
                relationships=[],
                synergies=["Your models work together powerfully for comprehensive analysis"],
                conflicts=["No significant conflicts detected"],
                application_sequence=[model['title'] for model in self.session.selected_models],
                unified_strategy="Apply models systematically to gain multiple perspectives on your challenge.",
                data={"error": "Simplified generation due to processing constraints"}
            )
    
    async def _analyze_comprehensive_relationships(self) -> List[Dict[str, str]]:
        """Analyze relationships between mental models comprehensively"""
        relationships = []
        
        for i, model1 in enumerate(self.session.selected_models):
            for j, model2 in enumerate(self.session.selected_models[i+1:], i+1):
                try:
                    # Use LLM to analyze relationship between models
                    relationship_prompt = f"""
Analyze the relationship between these two mental models in the context of "{self.session.user_goal}":

Model 1: {model1['title']}
Description: {model1.get('rationale', 'Strategic thinking framework')}

Model 2: {model2['title']}
Description: {model2.get('rationale', 'Strategic thinking framework')}

User Goal: {self.session.user_goal}

Provide a JSON response:
{{
  "relationship_type": "foundation|complementary|reinforcing|checking|sequential",
  "strength": "weak|moderate|strong",
  "description": "How these models interact and support each other",
  "application_insight": "Practical insight about using them together"
}}
"""
                    
                    response = await self.llm_client.generate_response(
                        prompt=relationship_prompt,
                        system_prompt="You are a mental models expert. Always return valid JSON."
                    )
                    
                    parsed_relationship = self._parse_json_response(response)
                    
                    relationships.append({
                        "model1": model1['title'],
                        "model2": model2['title'],
                        "relationship_type": parsed_relationship.get('relationship_type', 'complementary'),
                        "strength": parsed_relationship.get('strength', 'moderate'),
                        "description": parsed_relationship.get('description', 'These models work together effectively'),
                        "application_insight": parsed_relationship.get('application_insight', 'Use both for comprehensive analysis')
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze relationship between {model1['title']} and {model2['title']}: {e}")
                    # Fallback relationship
                    basic_relationship = self._analyze_model_relationship(model1, model2)
                    if basic_relationship:
                        relationships.append({
                            "model1": model1['title'],
                            "model2": model2['title'],
                            "relationship_type": basic_relationship['type'],
                            "strength": "moderate",
                            "description": basic_relationship['description'],
                            "application_insight": "Apply both models for enhanced perspective"
                        })
        
        return relationships
    
    async def _generate_enhanced_synergies(self) -> List[str]:
        """Generate enhanced synergies analysis using LLM"""
        try:
            model_names = [model['title'] for model in self.session.selected_models]
            
            synergies_prompt = f"""
Analyze the synergies between these mental models for the goal: "{self.session.user_goal}"

Selected Models: {', '.join(model_names)}
Context: {self.session.domain_context}

Identify 3-5 key synergies where these models enhance each other. For each synergy, explain:
1. Which models are involved
2. How they amplify each other
3. Practical benefit for the user's goal

Provide as JSON array:
[
  "Synergy description 1: Models X and Y create powerful combination by...",
  "Synergy description 2: Models Y and Z together enable..."
]
"""
            
            response = await self.llm_client.generate_response(
                prompt=synergies_prompt,
                system_prompt="You are a mental models expert. Return a JSON array of synergy descriptions."
            )
            
            # Parse JSON array of synergies
            import json
            parsed_synergies = json.loads(response) if response.strip().startswith('[') else []
            
            if parsed_synergies:
                return parsed_synergies
                
        except Exception as e:
            logger.warning(f"Enhanced synergies generation failed: {e}")
        
        # Fallback to basic synergies
        basic_synergies = self._identify_synergies(self.session.selected_models)
        return [s['description'] for s in basic_synergies] if basic_synergies else [
            "Your selected models provide multiple complementary perspectives",
            "Models work together to reduce blind spots and biases",
            "Combined framework enables both analytical and intuitive thinking"
        ]
    
    async def _identify_potential_conflicts(self) -> List[str]:
        """Identify potential conflicts or tensions between models"""
        try:
            model_names = [model['title'] for model in self.session.selected_models]
            
            conflicts_prompt = f"""
Analyze potential conflicts or tensions between these mental models:

Models: {', '.join(model_names)}
Context: {self.session.user_goal} in {self.session.domain_context}

Identify any potential conflicts, contradictions, or tensions that might arise when using these models together. Consider:
1. Philosophical differences
2. Practical application conflicts
3. Time/resource conflicts
4. Contradictory recommendations

Return as JSON array of conflict descriptions, or empty array if no significant conflicts:
["Conflict description 1", "Conflict description 2"]
"""
            
            response = await self.llm_client.generate_response(
                prompt=conflicts_prompt,
                system_prompt="You are a mental models expert. Return a JSON array of conflict descriptions or empty array."
            )
            
            import json
            parsed_conflicts = json.loads(response) if response.strip().startswith('[') else []
            return parsed_conflicts
            
        except Exception as e:
            logger.warning(f"Conflicts analysis failed: {e}")
            return []  # No conflicts identified
    
    def _generate_optimal_sequence(self) -> List[str]:
        """Generate optimal application sequence for mental models"""
        # Enhanced heuristic for optimal sequencing
        foundation_models = []
        analysis_models = []
        decision_models = []
        validation_models = []
        
        for model in self.session.selected_models:
            name = model['title'].lower()
            if any(word in name for word in ['first principles', 'systems', 'fundamental']):
                foundation_models.append(model['title'])
            elif any(word in name for word in ['bias', 'assumption', 'analysis', 'pareto']):
                analysis_models.append(model['title'])
            elif any(word in name for word in ['opportunity', 'cost', 'decision', 'choice']):
                decision_models.append(model['title'])
            else:
                validation_models.append(model['title'])
        
        # Optimal sequence: Foundation → Analysis → Decision → Validation
        optimal_sequence = foundation_models + analysis_models + decision_models + validation_models
        
        # Ensure all models are included
        all_model_names = [model['title'] for model in self.session.selected_models]
        for model_name in all_model_names:
            if model_name not in optimal_sequence:
                optimal_sequence.append(model_name)
        
        return optimal_sequence
    
    async def _generate_unified_strategy(self) -> str:
        """Generate unified application strategy"""
        try:
            model_titles = [model['title'] for model in self.session.selected_models]
            
            strategy_prompt = f"""
Create a unified strategy for applying these mental models to achieve: "{self.session.user_goal}" in "{self.session.domain_context}".

Selected Models: {', '.join(model_titles)}

Provide a concise, actionable strategy (3-4 sentences) that explains:
1. How to integrate all models effectively
2. The recommended approach or sequence
3. Key principles for maximum impact

Make it practical and specific to their goal.
"""
            
            strategy = await self.llm_client.generate_response(
                prompt=strategy_prompt,
                system_prompt="You are a strategic thinking expert. Provide clear, actionable guidance."
            )
            
            return strategy.strip()
            
        except Exception as e:
            logger.warning(f"Unified strategy generation failed: {e}")
            return f"Apply your selected models systematically: start with foundational analysis, then use complementary frameworks to validate insights and guide decision-making for '{self.session.user_goal}'."
    
    def _format_enhanced_mental_model_map(self, mental_map: MentalModelMap) -> str:
        """Format enhanced mental model map for display"""
        formatted_sections = []
        
        # Header
        formatted_sections.append("🗺️ **YOUR PERSONALIZED MENTAL MODEL MAP**\n")
        
        # Core Models Table
        formatted_sections.append("### 🧠 Core Models for Your Challenge")
        formatted_sections.append(f"**Goal:** {mental_map.data.get('context', self.session.user_goal)}\n")
        
        models_table = "| # | Mental Model | Role in Your Toolkit |\n|---|---|---|\n"
        for i, model in enumerate(mental_map.core_models, 1):
            # Find the model details from selected_models
            model_details = next((m for m in self.session.selected_models if m['title'] == model), {})
            role = model_details.get('primary_application', 'Strategic Analysis')
            models_table += f"| {i} | **{model}** | {role} |\n"
        
        formatted_sections.append(models_table)
        
        # Application Sequence
        if mental_map.application_sequence:
            formatted_sections.append("### 🔄 Recommended Application Sequence")
            sequence_text = " → ".join([f"**{i}. {model}**" for i, model in enumerate(mental_map.application_sequence, 1)])
            formatted_sections.append(sequence_text + "\n")
        
        # Synergies
        if mental_map.synergies:
            formatted_sections.append("### ✨ Key Synergies")
            for synergy in mental_map.synergies:
                formatted_sections.append(f"• {synergy}")
            formatted_sections.append("")
        
        # Relationships
        if mental_map.relationships:
            formatted_sections.append("### 🔗 Model Relationships")
            for rel in mental_map.relationships:
                strength_emoji = "🔥" if rel['strength'] == 'strong' else "🔌" if rel['strength'] == 'moderate' else "🔵"
                formatted_sections.append(f"{strength_emoji} **{rel['model1']} ↔ {rel['model2']}** ({rel['relationship_type']})")
                formatted_sections.append(f"   {rel['description']}")
            formatted_sections.append("")
        
        # Potential Conflicts
        if mental_map.conflicts:
            formatted_sections.append("### ⚠️ Potential Tensions to Manage")
            for conflict in mental_map.conflicts:
                formatted_sections.append(f"• {conflict}")
            formatted_sections.append("")
        
        # Unified Strategy
        formatted_sections.append("### 💯 Unified Application Strategy")
        formatted_sections.append(mental_map.unified_strategy + "\n")
        
        # Next Steps
        formatted_sections.append("🚀 **Next Step**: I'll provide specific implementation guidance with actionable steps tailored to your goal.\n")
        formatted_sections.append("✨ Ready for your implementation roadmap?")
        
        return "\n\n".join(formatted_sections)
    
    async def _generate_mental_model_map(self) -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility"""
        enhanced_map = await self._generate_enhanced_mental_model_map()
        
        # Convert back to legacy format
        return {
            'core_models': enhanced_map.core_models,
            'relationships': enhanced_map.relationships,
            'synergies': "\n".join([f"• {s}" for s in enhanced_map.synergies]),
            'conflicts': "\n".join([f"• {c}" for c in enhanced_map.conflicts]) if enhanced_map.conflicts else "No significant conflicts detected.",
            'unified_strategy': enhanced_map.unified_strategy,
            'application_sequence': enhanced_map.application_sequence,
            'context': self.session.user_goal
        }
    
    def _analyze_model_relationship(self, model1: Dict[str, Any], model2: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Analyze relationship between two mental models"""
        name1, name2 = model1['title'].lower(), model2['title'].lower()
        
        # Predefined relationship patterns
        relationships = [
            {
                'keywords1': ['first principles', 'fundamental'],
                'keywords2': ['systems', 'complex'],
                'type': 'foundation',
                'description': 'First principles provides the foundation for systems analysis'
            },
            {
                'keywords1': ['opportunity cost', 'cost'],
                'keywords2': ['decision', 'choice'],
                'type': 'constraint',
                'description': 'Opportunity cost constrains decision-making options'
            },
            {
                'keywords1': ['bias', 'cognitive'],
                'keywords2': ['decision', 'choice', 'thinking'],
                'type': 'check',
                'description': 'Bias awareness serves as a check on decision processes'
            }
        ]
        
        for rel in relationships:
            if (any(kw in name1 for kw in rel['keywords1']) and any(kw in name2 for kw in rel['keywords2'])) or \
               (any(kw in name2 for kw in rel['keywords1']) and any(kw in name1 for kw in rel['keywords2'])):
                return {
                    'type': rel['type'],
                    'description': rel['description']
                }
        
        return None
    
    def _suggest_application_sequence(self, models: List[Dict[str, Any]]) -> List[str]:
        """Suggest optimal sequence for applying mental models"""
        # Simple heuristic: foundation models first, then analysis, then decision
        foundation_models = []
        analysis_models = []
        decision_models = []
        
        for model in models:
            name = model['title'].lower()
            if any(word in name for word in ['first principles', 'systems', 'fundamental']):
                foundation_models.append(model['title'])
            elif any(word in name for word in ['bias', 'assumption', 'analysis']):
                analysis_models.append(model['title'])
            else:
                decision_models.append(model['title'])
        
        return foundation_models + analysis_models + decision_models
    
    def _identify_synergies(self, models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Identify synergies between mental models"""
        synergies = []
        
        model_names = [model['title'].lower() for model in models]
        
        # Common synergy patterns
        if any('systems' in name for name in model_names) and any('first principles' in name for name in model_names):
            synergies.append({
                'description': 'Systems thinking combined with first principles creates powerful problem decomposition',
                'models': 'Systems Thinking + First Principles'
            })
        
        if any('bias' in name for name in model_names) and any('decision' in name or 'opportunity' in name for name in model_names):
            synergies.append({
                'description': 'Bias awareness improves decision-making quality and opportunity cost analysis',
                'models': 'Bias Awareness + Decision Making'
            })
        
        return synergies
    
    def _format_mental_model_map(self, mental_map: Dict[str, Any]) -> str:
        """Format mental model map for display"""
        formatted = "🗺️ **MENTAL MODEL MAP**\n\n"
        
        # Core models
        formatted += f"**Core Models for: {mental_map['context']}**\n"
        for i, model in enumerate(mental_map['core_models'], 1):
            formatted += f"{i}. {model}\n"
        
        # Application sequence
        if mental_map.get('application_sequence'):
            formatted += "\n**Recommended Application Sequence:**\n"
            for i, model in enumerate(mental_map['application_sequence'], 1):
                formatted += f"{i} → {model}\n"
        
        # Relationships
        if mental_map.get('relationships'):
            formatted += "\n**Model Relationships:**\n"
            for rel in mental_map['relationships']:
                formatted += f"• {rel['model1']} ↔ {rel['model2']}: {rel['description']}\n"
        
        # Synergies
        if mental_map.get('synergies') and mental_map['synergies'] != 'Your models work together powerfully.':
            formatted += "\n**Key Synergies:**\n"
            formatted += mental_map['synergies'] + "\n"
        
        return formatted
    
    async def _generate_structured_reflection_prompts(self) -> ReflectionPrompts:
        """Generate structured reflection prompts based on the journey"""
        try:
            model_names = [model['title'] for model in self.session.selected_models]
            
            # Use LLM to generate personalized reflection prompts
            reflection_prompt = f"""
Generate thoughtful reflection questions for someone who just completed a Mental Model Navigator session.

Their journey:
- Goal: {self.session.user_goal}
- Domain: {self.session.domain_context}
- Models learned: {', '.join(model_names)}
- Mental Model Map created: {'Yes' if self.session.mental_model_map else 'No'}

Generate 5-7 reflection questions that help them:
1. Consolidate their learning
2. Identify key insights
3. Plan practical application
4. Recognize mindset shifts

Also suggest 3-5 immediate action items they should take.

Provide as JSON:
{{
  "reflection_prompts": ["Question 1", "Question 2", ...],
  "model_specific_questions": [
    {{"model": "Model Name", "question": "Specific question about this model"}}
  ],
  "action_items": ["Action 1", "Action 2", ...]
}}
"""
            
            response = await self.llm_client.generate_response(
                prompt=reflection_prompt,
                system_prompt="You are a learning and reflection expert. Generate insightful questions and actionable items."
            )
            
            parsed_reflection = self._parse_json_response(response)
            
            # Extract model-specific questions
            model_specific = parsed_reflection.get('model_specific_questions', [])
            if not model_specific and model_names:
                # Generate fallback model-specific questions
                for model_name in model_names[:3]:  # Limit to 3
                    model_specific.append({
                        "model": model_name,
                        "question": f"How will you specifically apply {model_name} to your work going forward?"
                    })
            
            return ReflectionPrompts(
                prompts=parsed_reflection.get('reflection_prompts', self._get_fallback_reflection_prompts()),
                model_specific_questions=model_specific,
                action_items=parsed_reflection.get('action_items', self._get_fallback_action_items()),
                data={
                    "session_context": self.session.user_goal,
                    "models_count": len(model_names),
                    "generation_method": "llm_personalized"
                }
            )
            
        except Exception as e:
            logger.warning(f"Structured reflection generation failed: {e}")
            return self._get_fallback_reflection_prompts_structured()
    
    def _get_fallback_reflection_prompts(self) -> List[str]:
        """Get fallback reflection prompts"""
        return [
            f"How has your understanding of '{self.session.user_goal}' evolved through this process?",
            "Which mental model provided the most surprising insight, and why?",
            "What assumptions did you discover you were making that you hadn't recognized before?",
            "How might you apply these mental models to other challenges you're facing?",
            "What would you do differently if you were to approach this problem again?",
            "How will you remember to use these mental models in future decisions?"
        ]
    
    def _get_fallback_action_items(self) -> List[str]:
        """Get fallback action items"""
        return [
            "Choose one mental model to apply within the next 48 hours",
            "Schedule a weekly review of your mental model toolkit",
            "Share one insight with a colleague or friend",
            "Identify your next learning opportunity in mental models"
        ]
    
    def _get_fallback_reflection_prompts_structured(self) -> ReflectionPrompts:
        """Get fallback structured reflection prompts"""
        model_names = [model['title'] for model in self.session.selected_models]
        
        model_specific = []
        for model_name in model_names[:3]:
            model_specific.append({
                "model": model_name,
                "question": f"What specific situation will you apply {model_name} to first?"
            })
        
        return ReflectionPrompts(
            prompts=self._get_fallback_reflection_prompts(),
            model_specific_questions=model_specific,
            action_items=self._get_fallback_action_items(),
            data={"source": "fallback"}
        )
    
    def _format_reflection_and_next_steps(self, reflection_prompts: ReflectionPrompts) -> str:
        """Format reflection prompts and next steps for display"""
        formatted_sections = []
        
        # Header
        formatted_sections.append("🎆 **REFLECTION & NEXT STEPS**\n")
        
        # Journey completion celebration
        formatted_sections.append("🎉 **Congratulations!** You've successfully completed the Mental Model Navigator journey!\n")
        
        # Session summary
        formatted_sections.append("### 📋 Your Mental Model Journey Summary")
        formatted_sections.append(f"**🎯 Goal:** {self.session.user_goal}")
        model_names = [m.get('title', 'Unknown') for m in self.session.selected_models]
        formatted_sections.append(f"**🧠 Models Mastered:** {', '.join(model_names)}")
        formatted_sections.append(f"**🗺️ Mental Model Map:** {'✅ Created' if self.session.mental_model_map else '❌ Not created'}")
        formatted_sections.append(f"**📚 Detailed Explanations:** {'✅ Generated' if self.session.model_explanations else '❌ Not generated'}\n")
        
        # Reflection questions
        formatted_sections.append("### 🤔 Reflection Questions")
        formatted_sections.append("Take a moment to reflect on your learning journey:\n")
        
        for i, prompt in enumerate(reflection_prompts.prompts, 1):
            formatted_sections.append(f"**{i}.** {prompt}")
        formatted_sections.append("")
        
        # Model-specific questions
        if reflection_prompts.model_specific_questions:
            formatted_sections.append("### 🎯 Model-Specific Applications")
            for item in reflection_prompts.model_specific_questions:
                formatted_sections.append(f"**{item['model']}:** {item['question']}")
            formatted_sections.append("")
        
        # Action items
        formatted_sections.append("### 🚀 Immediate Action Items")
        formatted_sections.append("Your next steps to maximize learning:")
        for i, action in enumerate(reflection_prompts.action_items, 1):
            formatted_sections.append(f"{i}. {action}")
        formatted_sections.append("")
        
        # Toolkit reference
        if self.session.mental_model_map:
            formatted_sections.append("### 📝 Your Toolkit Reference")
            formatted_sections.append("**Mental Model Map:** Use the map above as your quick reference guide")
            formatted_sections.append("**Application Sequence:** Follow the recommended sequence for best results")
            formatted_sections.append("**Synergies:** Leverage the identified synergies for compound benefits\n")
        
        # Continuous learning
        formatted_sections.append("### 📚 Continue Your Mental Models Journey")
        formatted_sections.append("• **Start a new session** to explore different mental models")
        formatted_sections.append("• **Apply these models** to new challenges as they arise")
        formatted_sections.append("• **Share your insights** with others to deepen understanding")
        formatted_sections.append("• **Build the habit** of asking 'What mental models apply here?'\n")
        
        # Closing
        formatted_sections.append("🌟 **You now have a powerful toolkit for approaching complex challenges with mental models!**\n")
        formatted_sections.append("🔄 **Ready for a new challenge?** Start a new session anytime to explore different mental models for other situations.")
        
        return "\n\n".join(formatted_sections)
    
    async def _generate_reflection_prompts(self) -> str:
        """Legacy method - kept for backward compatibility"""
        structured_prompts = await self._generate_structured_reflection_prompts()
        
        formatted = ""
        for i, prompt in enumerate(structured_prompts.prompts[:5], 1):
            formatted += f"{i}. {prompt}\n"
        
        return formatted
    
    async def _generate_comprehensive_next_steps(self) -> str:
        """Generate comprehensive next steps"""
        next_steps = "🎯 **YOUR MENTAL MODEL TOOLKIT**\n\n"
        
        # Model summary
        if self.session.selected_models:
            next_steps += "**Models You've Learned:**\n"
            for model in self.session.selected_models:
                next_steps += f"• {model['title']}: Ready to apply\n"
        
        # Practical actions
        next_steps += "\n**Immediate Actions:**\n"
        next_steps += "1. Choose one mental model to apply this week\n"
        next_steps += "2. Identify a specific decision where you can test the model\n"
        next_steps += "3. Document what insights the model reveals\n"
        
        # Mental model map reference
        if self.session.mental_model_map:
            next_steps += "\n**Reference Your Mental Model Map:**\n"
            next_steps += "Use the map above to understand how your models connect and reinforce each other.\n"
        
        next_steps += "\n**Continue Learning:**\n"
        next_steps += "• Start a new session to explore different mental models\n"
        next_steps += "• Apply these models to new challenges as they arise\n"
        next_steps += "• Share your insights with others to deepen your understanding\n"
        
        return next_steps
