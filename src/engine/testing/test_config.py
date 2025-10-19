#!/usr/bin/env python3
"""
Test Configuration System
Manages cost-efficient testing configurations
"""

import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class TestMode(str, Enum):
    """Test execution modes with different cost profiles"""

    MOCK_ALL = "mock_all"  # $0.00 - All APIs mocked
    HYBRID = "hybrid"  # $0.10 - Critical paths real, rest mocked
    PRODUCTION = "production"  # $0.50+ - All real API calls
    SELECTIVE = "selective"  # $0.15 - Only specific stages real


@dataclass
class TestConfiguration:
    """Configuration for test execution"""

    mode: TestMode
    enable_perplexity: bool
    enable_deepseek: bool
    enable_claude_fallback: bool
    max_research_queries: int
    cache_research_results: bool
    estimated_cost_usd: float


class TestConfigManager:
    """Manages test configurations to control costs"""

    def __init__(self):
        self.configs = {
            TestMode.MOCK_ALL: TestConfiguration(
                mode=TestMode.MOCK_ALL,
                enable_perplexity=False,
                enable_deepseek=False,
                enable_claude_fallback=False,
                max_research_queries=0,
                cache_research_results=True,
                estimated_cost_usd=0.0,
            ),
            TestMode.HYBRID: TestConfiguration(
                mode=TestMode.HYBRID,
                enable_perplexity=True,
                enable_deepseek=False,  # Mock expensive DeepSeek calls
                enable_claude_fallback=True,
                max_research_queries=5,  # Limit research calls
                cache_research_results=True,
                estimated_cost_usd=0.10,
            ),
            TestMode.SELECTIVE: TestConfiguration(
                mode=TestMode.SELECTIVE,
                enable_perplexity=True,
                enable_deepseek=True,
                enable_claude_fallback=False,
                max_research_queries=8,
                cache_research_results=True,
                estimated_cost_usd=0.15,
            ),
            TestMode.PRODUCTION: TestConfiguration(
                mode=TestMode.PRODUCTION,
                enable_perplexity=True,
                enable_deepseek=True,
                enable_claude_fallback=True,
                max_research_queries=25,
                cache_research_results=False,
                estimated_cost_usd=0.50,
            ),
        }

    def get_config(self, mode: Optional[TestMode] = None) -> TestConfiguration:
        """Get configuration for specified mode"""
        if mode is None:
            # Determine mode from environment
            mode_str = os.getenv("TEST_MODE", "mock_all").lower()
            mode = TestMode(mode_str)

        return self.configs[mode]

    def apply_config(self, config: TestConfiguration):
        """Apply configuration to environment"""
        os.environ["ENABLE_PERPLEXITY_RESEARCH"] = str(config.enable_perplexity).lower()
        os.environ["ENABLE_DEEPSEEK_CALLS"] = str(config.enable_deepseek).lower()
        os.environ["ENABLE_CLAUDE_FALLBACK"] = str(
            config.enable_claude_fallback
        ).lower()
        os.environ["MAX_RESEARCH_QUERIES"] = str(config.max_research_queries)
        os.environ["CACHE_RESEARCH_RESULTS"] = str(
            config.cache_research_results
        ).lower()

        print(f"🎯 Test Configuration Applied: {config.mode.value}")
        print(f"   Estimated cost: ${config.estimated_cost_usd:.2f}")
        print(f"   Perplexity: {'✅' if config.enable_perplexity else '❌'}")
        print(f"   DeepSeek: {'✅' if config.enable_deepseek else '❌'}")
        print(f"   Max research queries: {config.max_research_queries}")


# Global configuration manager
_config_manager = TestConfigManager()


def get_test_config(mode: Optional[TestMode] = None) -> TestConfiguration:
    """Get test configuration"""
    return _config_manager.get_config(mode)


def apply_test_config(mode: TestMode):
    """Apply test configuration"""
    config = _config_manager.get_config(mode)
    _config_manager.apply_config(config)
    return config
