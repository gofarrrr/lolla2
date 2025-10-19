"""
Knowledge-Enhanced Prompt Templates - Three-Step Cognitive Gauntlet
=================================================================

Advanced prompt templates for the Plan-Synthesize-Analyze workflow that grants
AI consultants access to the mental models knowledge base.

Part of the Great Knowledge Infusion - Build Order KB-03
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class KnowledgeEnhancedPrompts:
    """
    Prompt templates for the three-step cognitive gauntlet:
    1. Research Planner Prompt
    2. Knowledge Synthesizer Prompt  
    3. Final Analysis Prompt (knowledge-augmented)
    """
    
    @staticmethod
    def create_research_planner_prompt(
        consultant_type: str,
        specialization: str,
        user_query: str,
        assigned_nways: List[str] = None
    ) -> str:
        """
        Step 1: Research Planner Prompt
        
        Creates a prompt that instructs the consultant to formulate targeted
        research questions for the knowledge base search.
        
        Args:
            consultant_type: Type of consultant (e.g., "strategic_analyst")
            specialization: Consultant's specialization area
            user_query: The original user query
            assigned_nways: List of assigned N-way dimensions
            
        Returns:
            Formatted research planner prompt
        """
        nway_context = ""
        if assigned_nways:
            nway_context = f"\n\nYour assigned cognitive dimensions: {', '.join(assigned_nways)}"
        
        return f"""You are a {consultant_type} with specialization in {specialization}. You have been assigned the strategic problem: "{user_query}"{nway_context}

Before writing your analysis, you must formulate a research plan. You have a toolbox with two available tools:

🧠 **knowledge_base_search(query: str)**: Use this to query our internal, proprietary library of deep mental models for timeless strategic wisdom and cognitive frameworks.

🌐 **live_internet_research(query: str)**: Use this to get real-time, up-to-the-minute data, market statistics, news, or current developments from the live internet.

Your task is to formulate a research plan. Generate a list of 1-3 tool calls to gather the context you need. You must intelligently choose the right tool for each question:

**Tool Selection Guidelines:**
- Use **knowledge_base_search** for: strategic frameworks, mental models, cognitive biases, timeless principles, decision-making frameworks
- Use **live_internet_research** for: current market data, recent events, statistics, competitive intelligence, trends, regulatory changes

**Examples of Good Tool Selection:**
- "What mental models apply to market entry?" → knowledge_base_search
- "What is Amazon's current market share in cloud computing?" → live_internet_research  
- "How does Systems Thinking apply to competitive analysis?" → knowledge_base_search
- "What are the latest AI startup funding trends in 2024?" → live_internet_research

If the user's query can be answered from your general knowledge without tools, respond with an empty list of tool_calls.

Your output MUST be a JSON object with a single key "tool_calls", containing a list of tool call objects.

Example format:
{{
  "tool_calls": [
    {{
      "tool_name": "knowledge_base_search",
      "parameters": {{
        "query": "How does Second-Order Thinking apply to market entry timing decisions?",
        "top_k": 3
      }}
    }},
    {{
      "tool_name": "live_internet_research", 
      "parameters": {{
        "query": "Q3 2024 cloud computing market share data and competitive landscape",
        "research_type": "market_intelligence",
        "tier": "testing"
      }}
    }}
  ]
}}

Generate your intelligent research plan now:"""

    @staticmethod
    def create_knowledge_synthesizer_prompt(
        user_query: str,
        raw_rag_results: Any,
        consultant_type: str
    ) -> str:
        """
        Step 2: Knowledge Synthesizer Prompt
        
        Creates a prompt that instructs the assistant to synthesize the RAG results
        into a concise, powerful briefing memo.
        
        Args:
            user_query: The original user query
            raw_rag_results: Raw results from knowledge base search (list, dict, or error)
            consultant_type: Type of consultant for context
            
        Returns:
            Formatted knowledge synthesizer prompt
        """
        # Format the RAG results for the prompt
        if isinstance(raw_rag_results, list):
            if raw_rag_results:
                results_text = "\n\n".join([f"Result {i+1}:\n{result}" for i, result in enumerate(raw_rag_results)])
            else:
                results_text = "EMPTY_RESULTS: No relevant mental models found in the knowledge base."
        elif isinstance(raw_rag_results, dict) and "error" in raw_rag_results:
            results_text = f"ERROR: {raw_rag_results['error']}"
        else:
            results_text = f"UNKNOWN_FORMAT: {str(raw_rag_results)}"
            
        return f"""You are a research analyst preparing a briefing memo for a senior {consultant_type}.

The strategic problem is: "{user_query}"

You have received the following raw intelligence from your query of our internal knowledge base:

{results_text}

Your task is to reflect on this intelligence and synthesize it into a concise, powerful briefing memo (max 4 bullet points).

Guidelines:
- If you received relevant intelligence, extract only the most critical insights, quotes, and heuristics
- Focus on actionable frameworks and decision-making tools
- Highlight potential blind spots or contrarian perspectives
- If you received an empty list or an error, your synthesis must explicitly state that the knowledge base provided no relevant information on this topic

Format your response as a clear briefing memo that will be given to the senior consultant to ground their final analysis.

Intelligence Briefing Memo:"""

    @staticmethod
    def create_final_analysis_prompt(
        consultant_type: str,
        specialization: str,
        user_query: str,
        briefing_memo: str,
        assigned_nways: List[str] = None,
        framework: Dict[str, Any] = None,
        inquiry_complex_block: str = "",
        include_memo_models: bool = True,
        include_ask_back: bool = True,
    ) -> str:
        """
        Step 3: Final Analysis Prompt (knowledge-augmented)

        OPERATION PHOENIX: Prompt optimized for Grok 4 Fast reasoning mode.

        KEY PRINCIPLES FROM RESEARCH:
        1. Keep prompts simple and direct
        2. Avoid chain-of-thought instructions ("think step-by-step")
        3. Provide rich context but avoid prescribing reasoning steps
        4. Use clear delimiters for structure
        5. Define explicit success criteria

        Args:
            consultant_type: Type of consultant
            specialization: Consultant's specialization area
            user_query: The original user query
            briefing_memo: Synthesized intelligence briefing from Step 2
            assigned_nways: List of assigned N-way dimensions
            framework: Analysis framework (optional)
            inquiry_complex_block: Optional pre-formatted inquiry complex block
            include_memo_models: If True, include MeMo mental-model selection section
            include_ask_back: If True, include an ask-back catalytic question prompt

        Returns:
            Formatted final analysis prompt
        """
        # Dimension assignment (MANDATORY for cognitive diversity)
        dimension_context = ""
        if assigned_nways:
            dimension_context = f"""
<dimensions>
MANDATORY FOCUS: You are EXCLUSIVELY analyzing these dimensions: {', '.join(assigned_nways)}

COGNITIVE DIVERSITY REQUIREMENT:
- Other consultants are analyzing DIFFERENT dimensions (not yours)
- Do NOT provide generic observations that any consultant would notice
- Do NOT repeat insights from other analytical lenses
- Your value comes from your UNIQUE perspective on {', '.join(assigned_nways)}

Focus ONLY on your assigned dimensions. Ignore all other dimensions.
</dimensions>
"""
        else:
            # If no dimensions assigned, add warning
            dimension_context = """
<dimensions>
WARNING: No specific dimensions assigned. You must still provide a unique analytical perspective.
Avoid generic observations. Focus on insights unique to your specialization.
</dimensions>
"""

        # Framework context (optional)
        framework_context = ""
        if framework:
            framework_context = f"""
<framework>
Analysis Framework: {framework.get('name', 'Strategic Analysis')}
</framework>
"""

        # Optional MeMo (mental model selection) section
        memo_block = ""
        if include_memo_models:
            memo_block = (
                "<mental_models>\n"
                "From the following mental models, select 1–3 that best apply and explain why in 1–2 sentences each:\n"
                "[first principles, inversion, step-back, MECE, Fermi estimation, systems thinking, base rates, opportunity cost]\n"
                "Then apply them explicitly in the analysis.\n"
                "</mental_models>\n\n"
            )

        # Optional ask-back catalytic question
        ask_back_block = ""
        if include_ask_back:
            ask_back_block = (
                "\n<ask_back>End with one catalytic question you need answered to increase confidence.</ask_back>\n"
            )

        # The prompt (SIMPLE and DIRECT)
        return f"""<role>
You are a {consultant_type} specializing in {specialization}.
</role>

<task>
Analyze this strategic problem: {user_query}
</task>

{dimension_context}{framework_context}{inquiry_complex_block}
<intelligence_briefing>
{briefing_memo}
</intelligence_briefing>

{memo_block}<success_criteria>
Provide:
- 5+ specific insights with quantitative support (numbers, metrics, timelines)
- 3+ concrete recommendations with implementation details
- 3+ risk factors with mitigation strategies
- Explicit references to mental models (from <mental_models> if present)

Minimum 800 words. Be specific, not generic.
</success_criteria>

<output_format>
Use markdown with clear headings. Include specific numbers, budgets, timelines.
</output_format>{ask_back_block}

Analyze and recommend.
"""

    @staticmethod
    def format_tool_calls_for_execution(tool_calls_json: str) -> List[Dict[str, Any]]:
        """
        Parse and validate tool calls JSON from the research planner
        
        Args:
            tool_calls_json: JSON string containing tool calls
            
        Returns:
            List of validated tool call dictionaries
        """
        try:
            parsed = json.loads(tool_calls_json)
            
            if not isinstance(parsed, dict) or "tool_calls" not in parsed:
                return []
                
            tool_calls = parsed["tool_calls"]
            
            if not isinstance(tool_calls, list):
                return []
                
            # Validate each tool call (KB-04: support both internal RAG and external Perplexity tools)
            validated_calls = []
            for call in tool_calls:
                if not isinstance(call, dict) or not isinstance(call.get("parameters"), dict):
                    continue
                    
                tool_name = call.get("tool_name")
                parameters = call.get("parameters", {})
                
                # Validate knowledge_base_search calls
                if (tool_name == "knowledge_base_search" and "query" in parameters):
                    validated_calls.append(call)
                
                # Validate live_internet_research calls 
                elif (tool_name == "live_internet_research" and "query" in parameters):
                    validated_calls.append(call)
                    
            return validated_calls
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log the error but return empty list to gracefully handle parsing failures
            return []
            
    @staticmethod
    def extract_research_plan_from_response(response_content: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the research planner response
        
        Args:
            response_content: Raw response content from the research planner
            
        Returns:
            List of tool call dictionaries
        """
        try:
            # Try to find JSON in the response
            import re
            
            # Look for complete JSON block with proper nesting (improved regex for multi-tool arrays)
            json_match = re.search(r'\{.*?"tool_calls"\s*:\s*\[.*?\]\s*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return KnowledgeEnhancedPrompts.format_tool_calls_for_execution(json_str)
                
            # Alternative: extract just the response content if it looks like pure JSON
            stripped = response_content.strip()
            if stripped.startswith('{') and 'tool_calls' in stripped:
                return KnowledgeEnhancedPrompts.format_tool_calls_for_execution(stripped)
                
            return []
            
        except Exception:
            return []


# Convenience functions for easy access
def create_research_planner_prompt(
    consultant_type: str,
    specialization: str, 
    user_query: str,
    assigned_nways: List[str] = None
) -> str:
    """Convenience function for creating research planner prompt"""
    return KnowledgeEnhancedPrompts.create_research_planner_prompt(
        consultant_type, specialization, user_query, assigned_nways
    )


def create_knowledge_synthesizer_prompt(
    user_query: str,
    raw_rag_results: Any,
    consultant_type: str
) -> str:
    """Convenience function for creating knowledge synthesizer prompt"""
    return KnowledgeEnhancedPrompts.create_knowledge_synthesizer_prompt(
        user_query, raw_rag_results, consultant_type
    )


def create_final_analysis_prompt(
    consultant_type: str,
    specialization: str,
    user_query: str, 
    briefing_memo: str,
    assigned_nways: List[str] = None,
    framework: Dict[str, Any] = None
) -> str:
    """Convenience function for creating final analysis prompt"""
    return KnowledgeEnhancedPrompts.create_final_analysis_prompt(
        consultant_type, specialization, user_query, briefing_memo, assigned_nways, framework
    )