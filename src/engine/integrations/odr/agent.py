"""
ODR Research Agent base class
"""

import logging

from .config import ODRConfiguration

# ODR Dependencies (will be installed)
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain_anthropic import ChatAnthropic
    from langchain_tavily import TavilySearch
    from langchain.schema import SystemMessage, HumanMessage
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain.prompts import PromptTemplate

    ODR_AVAILABLE = True
except ImportError:
    ODR_AVAILABLE = False

logger = logging.getLogger(__name__)


class ODRResearchAgent:
    """Open Deep Research autonomous agent"""

    def __init__(self, config: ODRConfiguration):
        self.config = config
        self.llm = None
        self.search_tool = None
        self.agent = None
        self.demo_mode = False

        # Initialize templates
        try:
            from src.intelligence.research_templates import get_research_templates

            self.templates = get_research_templates()
        except ImportError:
            self.templates = None

        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the ODR research agent"""
        if not ODR_AVAILABLE:
            # In demo mode, just set attributes without failing
            self.llm = None
            self.search_tool = None
            self.demo_mode = True
            logger.info("🧪 ODR running in DEMO mode (no langchain dependencies)")
            return

        try:
            # Initialize LLM
            self.llm = ChatAnthropic(
                anthropic_api_key=self.config.anthropic_api_key,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Initialize search tool (with demo mode support)
            if self.config.tavily_api_key == "demo":
                # Demo mode - create mock search tool
                self.search_tool = None
                self.demo_mode = True
                logger.info("🧪 ODR running in DEMO mode (no real search)")
            else:
                self.search_tool = TavilySearch(
                    max_results=self.config.max_search_results,
                    search_depth=self.config.search_depth,
                )
                self.demo_mode = False

            logger.info("✅ ODR Research Agent initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ODR agent: {e}")
            # Fall back to demo mode instead of raising
            self.llm = None
            self.search_tool = None
            self.demo_mode = True
            logger.info("🧪 Falling back to ODR demo mode")
