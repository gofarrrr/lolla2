"""
Intelligence Contracts - METIS 2.0 System Contract Definitions
============================================================

Defines formal contracts for all intelligence systems in METIS 2.0:
- RAG System Contract for semantic search and knowledge storage
- Web Intelligence Contract for web scraping and data extraction
- Enhanced Research Contract for unified research orchestration
- Cost Optimization Contract for provider selection and cost management

These contracts provide standardized interfaces and guarantees for
all intelligence capabilities in the METIS platform.
"""

from typing import List, Dict, Any, Optional

# Import base contract classes
from src.core.contracts.system_contract import (
    SystemContract,
    Scope,
    Inputs,
    Rules,
    OutputSpec,
    Evaluation,
)


class RAGSystemContract(SystemContract):
    """
    Contract for RAG (Retrieval-Augmented Generation) system

    Defines capabilities, constraints, and behavior guarantees for
    semantic search, document storage, and knowledge retrieval.
    """

    contract_id: str = "rag_system@1.0"
    scope: Scope = Scope(
        job="retrieve_and_store_knowledge",
        audience="research_analysts_and_knowledge_workers",
        in_scope=[
            "Semantic search across stored documents",
            "Document storage with automatic embedding generation",
            "Knowledge deduplication and similarity detection",
            "Multi-modal content processing (text, structured data)",
            "Contextual relevance scoring and ranking",
            "Conversation memory integration",
            "Cross-reference and citation tracking",
        ],
        out_of_scope=[
            "Real-time web scraping (handled by WebIntelligenceContract)",
            "Direct LLM inference (handled by model providers)",
            "User authentication and access control",
            "Bulk data migration from external systems",
            "Image or video content analysis",
        ],
        success=[
            "Documents are stored with accurate embeddings within 5 seconds",
            "Search queries return relevant results with >80% accuracy",
            "System maintains <2 second response time for queries",
            "Duplicate detection identifies >95% of similar content",
            "Storage system maintains 99.9% uptime and data integrity",
        ],
        assumptions=[
            "Input documents are primarily text-based or structured data",
            "Voyage AI embedding service is available and responsive",
            "Milvus vector database is properly configured and operational",
            "Supabase and Zep storage systems are accessible",
            "Users provide well-formed search queries in English",
        ],
        guardrails=[
            "No storage of personally identifiable information without explicit consent",
            "Automatic content filtering for inappropriate or harmful material",
            "Rate limiting to prevent system abuse (100 requests/minute per user)",
            "Data retention policies enforced (documents expire after 1 year unless refreshed)",
            "Embedding dimension consistency enforced (1024 dimensions)",
        ],
    )

    inputs: Inputs = Inputs(
        required=[
            "search_query: string (1-500 characters)",
            "user_id: string (valid user identifier)",
            "document_content: string or structured_data (for storage operations)",
        ],
        optional=[
            "max_results: integer (1-100, default 10)",
            "similarity_threshold: float (0.0-1.0, default 0.7)",
            "freshness_required: boolean (default false)",
            "max_age_hours: integer (1-8760, default 24)",
            "metadata_filters: object (additional search constraints)",
        ],
        normalization=[
            "Query text is lowercased and stripped of extra whitespace",
            "Document content is chunked into optimal embedding segments",
            "Metadata fields are validated against defined schema",
            "Timestamps are normalized to ISO 8601 format",
        ],
        on_failure="diagnostic-json-with-retry-guidance",
    )

    process: List[str] = [
        "1. Validate input parameters against schema constraints",
        "2. For storage: Generate embeddings using Voyage AI (1024 dimensions)",
        "3. For storage: Detect duplicates using similarity scoring (>0.95 threshold)",
        "4. For storage: Store content in Milvus (vectors) and Supabase (metadata)",
        "5. For search: Generate query embedding and perform vector similarity search",
        "6. For search: Apply metadata filters and freshness constraints",
        "7. For search: Rank results by relevance and user context",
        "8. Return results with confidence scores and source attribution",
        "9. Log operation metrics for monitoring and optimization",
    ]

    rules: Rules = Rules(
        rounding=3,
        signed_numbers=True,
        tone="technical_precise",
        language="en_us",
        pii="aggregate_only",
    )

    output: OutputSpec = OutputSpec(
        formats=["json", "structured_response"], json_schema_version="draft-07"
    )

    evaluation: Evaluation = Evaluation(
        rubric=[
            "Response time: <2 seconds for search, <5 seconds for storage",
            "Accuracy: >80% relevant results in top 10 search results",
            "Completeness: All requested fields populated in response",
            "Consistency: Same query returns consistent results across sessions",
            "Error handling: Graceful degradation with informative error messages",
        ],
        auto_repair=True,
    )


class WebIntelligenceContract(SystemContract):
    """
    Contract for Web Intelligence system

    Defines capabilities for web scraping, content extraction,
    and intelligent provider selection (Firecrawl vs Apify).
    """

    contract_id: str = "web_intelligence@1.0"
    scope: Scope = Scope(
        job="gather_web_intelligence",
        audience="research_analysts_and_market_intelligence_teams",
        in_scope=[
            "URL scraping with markdown extraction",
            "Website crawling with depth and breadth controls",
            "Structured data parsing from web pages",
            "Intelligent provider selection (Firecrawl vs Apify)",
            "Content deduplication and quality filtering",
            "Rate limiting and ethical scraping practices",
            "Batch processing of multiple URLs",
            "Actor-based scraping for specialized content types",
        ],
        out_of_scope=[
            "Authentication bypass or circumventing access controls",
            "Scraping of illegal or copyrighted content",
            "Real-time monitoring or continuous surveillance",
            "Social media account impersonation",
            "Bulk email harvesting or contact scraping",
            "Breaking website terms of service",
        ],
        success=[
            "Successfully extract content from >90% of accessible web pages",
            "Provider selection algorithm achieves >85% cost efficiency",
            "Respect robots.txt and rate limiting for >99% of requests",
            "Content extraction maintains formatting and structure fidelity",
            "System processes batches of 100+ URLs within 10 minutes",
        ],
        assumptions=[
            "Target websites are publicly accessible",
            "Firecrawl and Apify services are available and responding",
            "Network connectivity is stable and unrestricted",
            "Websites follow standard HTML/CSS structure patterns",
            "Content is primarily in English or commonly supported languages",
        ],
        guardrails=[
            "Automatic robots.txt compliance checking",
            "Rate limiting: maximum 10 requests per second per domain",
            "Content filtering for adult, harmful, or illegal material",
            "Maximum crawl depth of 5 levels to prevent infinite recursion",
            "Timeout limits: 30 seconds per page, 10 minutes per batch operation",
            "Respect website rate limiting and 429 response codes",
        ],
    )

    inputs: Inputs = Inputs(
        required=[
            "urls: array[string] (valid HTTP/HTTPS URLs)",
            "task_description: string (scraping objective, 10-200 characters)",
        ],
        optional=[
            "max_pages: integer (1-1000, default 5)",
            "complexity: string (low|medium|high, default medium)",
            "preferred_provider: string (firecrawl|apify|auto, default auto)",
            "extract_formats: array[string] (markdown|html|structured, default [markdown])",
            "follow_links: boolean (default false)",
            "timeout_seconds: integer (10-300, default 30)",
        ],
        normalization=[
            "URLs are validated and normalized (add https:// if missing)",
            "Task description is sanitized and truncated to 200 characters",
            "Provider preference is validated against available options",
            "Timeout values are capped at system maximums",
        ],
        on_failure="diagnostic-json-with-fallback-provider",
    )

    process: List[str] = [
        "1. Validate input URLs and accessibility",
        "2. Analyze website complexity and content type",
        "3. Select optimal provider (Firecrawl for simple, Apify for complex)",
        "4. Configure provider-specific parameters and constraints",
        "5. Execute scraping with automatic retry logic",
        "6. Extract and normalize content to specified formats",
        "7. Apply content quality filters and deduplication",
        "8. Track costs and performance metrics",
        "9. Return structured results with metadata and provenance",
    ]

    rules: Rules = Rules(
        rounding=2,
        signed_numbers=False,
        tone="technical_objective",
        language="en_us",
        pii="exclude_all",
    )

    output: OutputSpec = OutputSpec(
        formats=["json", "markdown"], json_schema_version="draft-07"
    )

    evaluation: Evaluation = Evaluation(
        rubric=[
            "Success rate: >90% successful content extraction",
            "Cost efficiency: Provider selection saves >15% vs always using expensive option",
            "Speed: Average response time <30 seconds per URL",
            "Quality: Extracted content maintains >95% of original formatting",
            "Ethics: 100% compliance with robots.txt and rate limiting",
        ],
        auto_repair=True,
    )


class EnhancedResearchContract(SystemContract):
    """
    Contract for Enhanced Research system

    Defines unified research orchestration across RAG, web intelligence,
    and external research providers with cost optimization.
    """

    contract_id: str = "enhanced_research@1.0"
    scope: Scope = Scope(
        job="orchestrate_comprehensive_research",
        audience="strategic_analysts_and_decision_makers",
        in_scope=[
            "Unified research across multiple intelligence sources",
            "Automatic gap analysis and research planning",
            "Provider fallback chains for reliability",
            "Cost-optimized provider selection",
            "Real-time learning from research results",
            "Conversation memory integration",
            "Cross-source fact verification and synthesis",
            "Transparent research process documentation",
        ],
        out_of_scope=[
            "Original content creation or generation",
            "Financial investment advice or predictions",
            "Medical or legal professional guidance",
            "Personal data investigation or background checks",
            "Bulk data collection for commercial resale",
            "Automated decision-making without human oversight",
        ],
        success=[
            "Research completeness: >90% of user requirements addressed",
            "Source diversity: Utilizes 3+ different information sources",
            "Cost efficiency: Stays within 10% of allocated budget",
            "Response time: Complete research delivered within 2 minutes",
            "Learning integration: All results stored in RAG for future queries",
        ],
        assumptions=[
            "Multiple research providers (Perplexity, Exa, etc.) are available",
            "RAG system and web intelligence contracts are operational",
            "Cost optimization system has current provider pricing",
            "User queries are well-formed and specific enough for targeted research",
            "Budget allocations are sufficient for comprehensive research",
        ],
        guardrails=[
            "Automatic budget monitoring with alerts at 80% usage",
            "Research scope validation to prevent overly broad queries",
            "Source credibility scoring and bias detection",
            "Factual accuracy validation across multiple sources",
            "Privacy protection: no storage of sensitive personal information",
            "Rate limiting across all providers to prevent abuse",
        ],
    )

    inputs: Inputs = Inputs(
        required=[
            "research_query: string (10-1000 characters)",
            "user_id: string (valid user identifier)",
        ],
        optional=[
            "priority: string (low|medium|high, default medium)",
            "budget_limit: float (0.01-10.00 USD, default 1.00)",
            "needs_fresh_data: boolean (default false)",
            "max_sources: integer (1-10, default 5)",
            "research_depth: string (quick|standard|comprehensive, default standard)",
            "preferred_providers: array[string] (provider preferences)",
            "time_limit_minutes: integer (1-30, default 5)",
        ],
        normalization=[
            "Query text is analyzed for intent and complexity scoring",
            "Budget limits are validated against daily/monthly caps",
            "Provider preferences are checked against availability",
            "Time limits are adjusted based on query complexity",
        ],
        on_failure="diagnostic-json-with-partial-results",
    )

    process: List[str] = [
        "1. Analyze research query and identify information gaps",
        "2. Check RAG system for existing relevant knowledge",
        "3. Create intelligent research plan with provider selection",
        "4. Execute parallel research across selected providers",
        "5. Apply cost optimization and budget monitoring",
        "6. Synthesize results from multiple sources",
        "7. Perform fact verification and credibility assessment",
        "8. Store all new information in RAG for learning",
        "9. Update conversation memory and user context",
        "10. Return comprehensive results with source attribution",
    ]

    rules: Rules = Rules(
        rounding=2,
        signed_numbers=True,
        tone="analytical_professional",
        language="en_us",
        pii="contextual_only",
    )

    output: OutputSpec = OutputSpec(
        formats=["json", "markdown", "structured_analysis"],
        json_schema_version="draft-07",
    )

    evaluation: Evaluation = Evaluation(
        rubric=[
            "Comprehensiveness: >90% of research objectives met",
            "Source quality: Average credibility score >7/10",
            "Cost effectiveness: Within 110% of budget allocation",
            "Timeliness: Results delivered within specified time limit",
            "Learning value: New information successfully integrated into knowledge base",
        ],
        auto_repair=True,
    )


class CostOptimizationContract(SystemContract):
    """
    Contract for Cost Optimization system

    Defines intelligent cost management across all providers
    with budget controls and optimization algorithms.
    """

    contract_id: str = "cost_optimization@1.0"
    scope: Scope = Scope(
        job="optimize_provider_costs_and_budget_management",
        audience="system_administrators_and_finance_teams",
        in_scope=[
            "Multi-provider cost tracking and analysis",
            "Intelligent provider selection based on cost-effectiveness",
            "Budget allocation and monitoring across service categories",
            "Usage pattern analysis and forecasting",
            "Automatic cost alerts and budget controls",
            "ROI analysis and value optimization",
            "Provider performance vs cost correlation",
            "Volume discount optimization",
        ],
        out_of_scope=[
            "Financial accounting or bookkeeping",
            "Provider contract negotiation",
            "Payment processing or billing",
            "Tax calculation or compliance",
            "Multi-currency exchange rate management",
            "Subscription or license management",
        ],
        success=[
            "Cost reduction: Achieve 15%+ savings through optimal provider selection",
            "Budget compliance: Stay within allocated budgets 95% of the time",
            "Accuracy: Cost predictions within 10% of actual costs",
            "Response time: Provider selection decisions made within 100ms",
            "Monitoring: Real-time cost tracking with <5 minute delay",
        ],
        assumptions=[
            "Provider cost structures are stable and predictable",
            "Usage patterns have sufficient historical data for modeling",
            "Budget allocations are realistic and properly funded",
            "Provider performance metrics are accurately measured",
            "Volume discounts and pricing tiers are clearly defined",
        ],
        guardrails=[
            "Automatic budget limits with emergency circuit breakers",
            "Cost anomaly detection with immediate alerting",
            "Minimum service quality thresholds (don't sacrifice quality for cost)",
            "Provider diversity requirements (avoid single-provider dependence)",
            "Historical cost data retention for audit and analysis",
            "Transparency in all cost allocation and optimization decisions",
        ],
    )

    inputs: Inputs = Inputs(
        required=[
            "operation_type: string (search|scrape|research|storage)",
            "estimated_volume: integer (number of operations)",
        ],
        optional=[
            "priority_level: string (low|medium|high, default medium)",
            "quality_threshold: float (0.0-1.0, default 0.8)",
            "budget_limit: float (maximum cost in USD)",
            "preferred_providers: array[string] (provider preferences)",
            "time_sensitivity: string (immediate|standard|flexible, default standard)",
        ],
        normalization=[
            "Operation types are validated against supported categories",
            "Volume estimates are checked for reasonableness",
            "Quality thresholds are bounded within valid ranges",
            "Budget limits are validated against available funds",
        ],
        on_failure="diagnostic-json-with-alternative-options",
    )

    process: List[str] = [
        "1. Analyze operation requirements and constraints",
        "2. Calculate costs for all eligible providers",
        "3. Apply quality and performance weighting factors",
        "4. Consider current usage patterns and rate limits",
        "5. Evaluate volume discount opportunities",
        "6. Select optimal provider based on weighted scoring",
        "7. Track actual usage and costs against predictions",
        "8. Update cost models with real-world performance data",
        "9. Generate cost reports and optimization recommendations",
    ]

    rules: Rules = Rules(
        rounding=4,
        signed_numbers=True,
        tone="analytical_precise",
        language="en_us",
        pii="aggregate_only",
    )

    output: OutputSpec = OutputSpec(
        formats=["json", "financial_summary"], json_schema_version="draft-07"
    )

    evaluation: Evaluation = Evaluation(
        rubric=[
            "Cost accuracy: Predictions within 10% of actual costs",
            "Optimization effectiveness: Measurable cost savings achieved",
            "Budget compliance: No unauthorized budget overruns",
            "Performance: Provider selections maintain quality standards",
            "Transparency: Clear documentation of all cost decisions",
        ],
        auto_repair=True,
    )


# Contract Registry for easy access
INTELLIGENCE_CONTRACTS = {
    "rag_system": RAGSystemContract(),
    "web_intelligence": WebIntelligenceContract(),
    "enhanced_research": EnhancedResearchContract(),
    "cost_optimization": CostOptimizationContract(),
}


def get_contract(contract_name: str) -> Optional[SystemContract]:
    """Retrieve a specific intelligence contract by name"""
    return INTELLIGENCE_CONTRACTS.get(contract_name)


def list_contracts() -> List[str]:
    """List all available intelligence contract names"""
    return list(INTELLIGENCE_CONTRACTS.keys())


def validate_contracts() -> Dict[str, Any]:
    """Validate all intelligence contracts and return validation results"""
    results = {
        "valid_contracts": [],
        "invalid_contracts": [],
        "validation_errors": {},
        "total_contracts": len(INTELLIGENCE_CONTRACTS),
    }

    for name, contract in INTELLIGENCE_CONTRACTS.items():
        try:
            # Validate the contract by attempting to access all fields
            _ = contract.contract_id
            _ = contract.scope
            _ = contract.inputs
            _ = contract.process
            _ = contract.rules
            _ = contract.output
            _ = contract.evaluation

            results["valid_contracts"].append(name)

        except Exception as e:
            results["invalid_contracts"].append(name)
            results["validation_errors"][name] = str(e)

    results["validation_summary"] = (
        f"{len(results['valid_contracts'])}/{results['total_contracts']} contracts valid"
    )

    return results


def get_contract_summary() -> Dict[str, Any]:
    """Get a summary of all intelligence contracts and their capabilities"""
    summary = {
        "total_contracts": len(INTELLIGENCE_CONTRACTS),
        "contracts": {},
        "capabilities_matrix": {},
        "validation_status": validate_contracts()["validation_summary"],
    }

    # Extract key information from each contract
    for name, contract in INTELLIGENCE_CONTRACTS.items():
        summary["contracts"][name] = {
            "contract_id": contract.contract_id,
            "job": contract.scope.job,
            "audience": contract.scope.audience,
            "key_capabilities": contract.scope.in_scope[:3],  # First 3 capabilities
            "success_criteria_count": len(contract.scope.success),
            "guardrails_count": len(contract.scope.guardrails),
            "process_steps": len(contract.process),
        }

        # Build capabilities matrix
        for capability in contract.scope.in_scope:
            if capability not in summary["capabilities_matrix"]:
                summary["capabilities_matrix"][capability] = []
            summary["capabilities_matrix"][capability].append(name)

    return summary


# Export all contract classes and utilities
__all__ = [
    "RAGSystemContract",
    "WebIntelligenceContract",
    "EnhancedResearchContract",
    "CostOptimizationContract",
    "INTELLIGENCE_CONTRACTS",
    "get_contract",
    "list_contracts",
    "validate_contracts",
    "get_contract_summary",
]
