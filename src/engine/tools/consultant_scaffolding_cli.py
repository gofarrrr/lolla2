#!/usr/bin/env python3
"""
METIS V2.1 Consultant Scaffolding CLI Tool
Automates consultant onboarding from multi-step manual task to single command

This CLI tool addresses the "Consultant Onboarding Bottleneck" identified in the
architecture assessment by automating:
- Blueprint creation with industry-standard templates
- Database insertion with validation
- N-Way cluster generation and embedding
- Integration testing and validation

ARCHITECTURAL MANDATE COMPLIANCE:
✅ Glass-Box Transparency: All scaffolding operations logged to UnifiedContextStream
✅ Test-Driven Development: Built-in validation and testing capabilities
"""

import asyncio
import logging
import sys
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

# Core METIS imports
try:
    from src.engine.adapters.context_stream import UnifiedContextStream  # Migrated, ContextEventType
    from src.engine.engines.services.blueprint_registry import (
        BlueprintRegistry,
        ConsultantBlueprint,
    )
    from src.engine.engines.services.nway_selection_service import NWaySelectionService
    from src.engine.engines.services.nway_cache_service import NWayCacheService
    from src.engine.core.tool_decision_framework import ToolDecisionFramework

    # Database integration
    from supabase import Client

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ METIS dependencies not available: {e}")
    print("🛠️ Run from project root: python -m src.tools.consultant_scaffolding_cli")
    DEPENDENCIES_AVAILABLE = False


class ConsultantScaffoldingCLI:
    """
    Consultant Scaffolding CLI - Automates consultant onboarding

    Transforms multi-step manual consultant creation into single command execution
    with complete validation, testing, and Glass-Box audit trail.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from src.engine.adapters.context_stream import get_unified_context_stream  # Migrated
        self.context_stream = get_unified_context_stream() if DEPENDENCIES_AVAILABLE else None
        self.blueprint_registry: Optional[BlueprintRegistry] = None
        self.nway_service: Optional[NWaySelectionService] = None
        self.supabase_client: Optional[Client] = None

        # Industry-standard consultant templates
        self.consultant_templates = self._load_consultant_templates()

        # CLI execution tracking
        self.execution_id = str(uuid4())
        self.operations_log: List[Dict[str, Any]] = []

    def _load_consultant_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load industry-standard consultant blueprint templates"""

        return {
            "strategy": {
                "name_template": "{specialty} Strategy Consultant",
                "specialization": "strategic_analysis",
                "expertise_template": "{domain} strategy, competitive analysis, market positioning",
                "persona_template": "You are a {specialty} strategy consultant with deep expertise in {domain}. You specialize in strategic analysis, competitive positioning, and long-term planning.",
                "frameworks": [
                    "Porter's Five Forces",
                    "BCG Growth-Share Matrix",
                    "Ansoff Matrix",
                    "SWOT Analysis",
                    "Blue Ocean Strategy",
                ],
                "triggers": [
                    "strategy",
                    "strategic",
                    "competitive",
                    "market",
                    "positioning",
                ],
            },
            "operations": {
                "name_template": "{specialty} Operations Consultant",
                "specialization": "operational_excellence",
                "expertise_template": "{domain} operations, process optimization, operational efficiency",
                "persona_template": "You are an {specialty} operations consultant focused on {domain}. You excel at process optimization, operational efficiency, and systematic improvement.",
                "frameworks": [
                    "Lean Six Sigma",
                    "Theory of Constraints",
                    "Process Mapping",
                    "Kaizen Methodology",
                    "Operational Excellence",
                ],
                "triggers": [
                    "operations",
                    "process",
                    "efficiency",
                    "optimization",
                    "lean",
                ],
            },
            "technology": {
                "name_template": "{specialty} Technology Consultant",
                "specialization": "digital_transformation",
                "expertise_template": "{domain} technology, digital transformation, innovation strategy",
                "persona_template": "You are a {specialty} technology consultant specializing in {domain}. You focus on digital transformation, technology strategy, and innovation implementation.",
                "frameworks": [
                    "Digital Transformation Framework",
                    "Technology Adoption Lifecycle",
                    "Agile Methodology",
                    "DevOps Practices",
                    "Innovation Pipeline",
                ],
                "triggers": [
                    "technology",
                    "digital",
                    "innovation",
                    "transformation",
                    "tech",
                ],
            },
            "finance": {
                "name_template": "{specialty} Financial Consultant",
                "specialization": "financial_analysis",
                "expertise_template": "{domain} finance, financial modeling, investment analysis",
                "persona_template": "You are a {specialty} financial consultant with expertise in {domain}. You specialize in financial analysis, modeling, and strategic financial planning.",
                "frameworks": [
                    "Financial Modeling",
                    "DCF Valuation",
                    "Ratio Analysis",
                    "Capital Structure Optimization",
                    "Investment Analysis",
                ],
                "triggers": [
                    "finance",
                    "financial",
                    "investment",
                    "valuation",
                    "capital",
                ],
            },
            "marketing": {
                "name_template": "{specialty} Marketing Consultant",
                "specialization": "market_development",
                "expertise_template": "{domain} marketing, brand strategy, customer acquisition",
                "persona_template": "You are a {specialty} marketing consultant focused on {domain}. You excel at brand strategy, customer acquisition, and market development.",
                "frameworks": [
                    "Marketing Mix (4Ps)",
                    "Customer Journey Mapping",
                    "Brand Positioning Framework",
                    "Growth Hacking",
                    "Market Segmentation",
                ],
                "triggers": ["marketing", "brand", "customer", "acquisition", "growth"],
            },
        }

    async def initialize_services(
        self, supabase_client: Optional[Client] = None
    ) -> bool:
        """Initialize METIS services for consultant scaffolding"""

        if not DEPENDENCIES_AVAILABLE:
            print("❌ Cannot initialize services - dependencies unavailable")
            return False

        try:
            self.supabase_client = supabase_client

            # Initialize services with dependency injection
            self.blueprint_registry = BlueprintRegistry(
                context_stream=self.context_stream, supabase_client=supabase_client
            )

            self.nway_service = NWaySelectionService(
                context_stream=self.context_stream,
                tool_framework=None,  # Optional
                supabase_client=supabase_client,
            )

            # Glass-Box: Log service initialization
            self.context_stream.add_event(
                event_type=ContextEventType.SYSTEM_STATE,
                data={
                    "cli_tool": "consultant_scaffolding",
                    "execution_id": self.execution_id,
                    "services_initialized": True,
                    "database_connected": bool(supabase_client),
                },
                metadata={
                    "tool": "ConsultantScaffoldingCLI",
                    "operation": "initialize_services",
                },
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            return False

    def scaffold_consultant(
        self,
        consultant_type: str,
        specialty: str,
        domain: str,
        consultant_id: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Scaffold a new consultant with industry-standard configuration

        Args:
            consultant_type: Type from templates (strategy, operations, technology, finance, marketing)
            specialty: Specific specialty (e.g., "Healthcare", "Fintech", "Manufacturing")
            domain: Domain expertise (e.g., "digital health", "payment systems", "supply chain")
            consultant_id: Optional custom ID (auto-generated if not provided)
            custom_config: Optional custom configuration overrides

        Returns:
            Dict containing scaffolded consultant blueprint and metadata
        """

        # Generate consultant ID if not provided
        if not consultant_id:
            consultant_id = f"{consultant_type}_{specialty.lower().replace(' ', '_')}_{uuid4().hex[:8]}"

        # Get template for consultant type
        if consultant_type not in self.consultant_templates:
            available = list(self.consultant_templates.keys())
            raise ValueError(
                f"Unknown consultant type '{consultant_type}'. Available: {available}"
            )

        template = self.consultant_templates[consultant_type]

        # Apply template with specialty and domain
        consultant_config = {
            "consultant_id": consultant_id,
            "name": template["name_template"].format(specialty=specialty),
            "specialization": template["specialization"],
            "expertise": template["expertise_template"].format(domain=domain),
            "persona_prompt": template["persona_template"].format(
                specialty=specialty, domain=domain
            ),
            "stable_frameworks": template["frameworks"],
            "adaptive_triggers": template["triggers"]
            + [specialty.lower(), domain.lower()],
            "effectiveness_score": 0.8,  # Default effectiveness
            "metadata": {
                "scaffolded": True,
                "scaffold_timestamp": datetime.utcnow().isoformat(),
                "template_type": consultant_type,
                "specialty": specialty,
                "domain": domain,
                "execution_id": self.execution_id,
            },
        }

        # Apply custom overrides
        if custom_config:
            consultant_config.update(custom_config)

        # Glass-Box: Log scaffolding operation
        if self.context_stream:
            self.context_stream.add_event(
                event_type=ContextEventType.TOOL_EXECUTION,
                data={
                    "operation": "scaffold_consultant",
                    "consultant_type": consultant_type,
                    "consultant_id": consultant_id,
                    "specialty": specialty,
                    "domain": domain,
                },
                metadata={
                    "tool": "ConsultantScaffoldingCLI",
                    "execution_id": self.execution_id,
                },
            )

        # Log operation
        self.operations_log.append(
            {
                "operation": "scaffold_consultant",
                "consultant_id": consultant_id,
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return consultant_config

    async def register_consultant(self, consultant_config: Dict[str, Any]) -> bool:
        """
        Register scaffolded consultant with METIS services

        Args:
            consultant_config: Consultant configuration from scaffold_consultant()

        Returns:
            True if registration successful, False otherwise
        """

        if not self.blueprint_registry:
            self.logger.error("Blueprint registry not initialized")
            return False

        try:
            # Create ConsultantBlueprint object
            blueprint = ConsultantBlueprint(
                consultant_id=consultant_config["consultant_id"],
                name=consultant_config["name"],
                specialization=consultant_config["specialization"],
                expertise=consultant_config["expertise"],
                persona_prompt=consultant_config["persona_prompt"],
                stable_frameworks=consultant_config["stable_frameworks"],
                adaptive_triggers=consultant_config["adaptive_triggers"],
                effectiveness_score=consultant_config["effectiveness_score"],
                metadata=consultant_config["metadata"],
            )

            # Register with blueprint registry
            success = self.blueprint_registry.add_blueprint(blueprint)

            if success:
                # Glass-Box: Log successful registration
                if self.context_stream:
                    self.context_stream.add_event(
                        event_type=ContextEventType.CONSULTANT_SELECTION,
                        data={
                            "operation": "register_consultant",
                            "consultant_id": blueprint.consultant_id,
                            "consultant_name": blueprint.name,
                            "registration_success": True,
                        },
                        metadata={
                            "tool": "ConsultantScaffoldingCLI",
                            "execution_id": self.execution_id,
                        },
                    )

                self.operations_log.append(
                    {
                        "operation": "register_consultant",
                        "consultant_id": blueprint.consultant_id,
                        "success": True,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                return True
            else:
                self.logger.error(
                    f"Failed to register consultant {blueprint.consultant_id}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error registering consultant: {e}")

            # Glass-Box: Log registration error
            if self.context_stream:
                self.context_stream.add_event(
                    event_type=ContextEventType.ERROR_OCCURRED,
                    data={
                        "operation": "register_consultant",
                        "error": str(e),
                        "consultant_id": consultant_config.get(
                            "consultant_id", "unknown"
                        ),
                    },
                    metadata={
                        "tool": "ConsultantScaffoldingCLI",
                        "execution_id": self.execution_id,
                    },
                )

            self.operations_log.append(
                {
                    "operation": "register_consultant",
                    "consultant_id": consultant_config.get("consultant_id", "unknown"),
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return False

    async def test_consultant(self, consultant_id: str) -> Dict[str, Any]:
        """
        Test scaffolded consultant with METIS services

        Args:
            consultant_id: ID of consultant to test

        Returns:
            Test results with validation metrics
        """

        test_results = {
            "consultant_id": consultant_id,
            "tests": {},
            "overall_success": False,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # Test 1: Blueprint Retrieval
            if self.blueprint_registry:
                blueprint = self.blueprint_registry.get_blueprint(consultant_id)
                test_results["tests"]["blueprint_retrieval"] = {
                    "success": blueprint is not None,
                    "result": (
                        f"Blueprint found: {blueprint.name}"
                        if blueprint
                        else "Blueprint not found"
                    ),
                }

            # Test 2: N-Way Selection Integration
            if (
                self.nway_service
                and test_results["tests"]["blueprint_retrieval"]["success"]
            ):
                # Create test query matching consultant's triggers
                blueprint = self.blueprint_registry.get_blueprint(consultant_id)
                test_query = (
                    f"I need help with {' and '.join(blueprint.adaptive_triggers[:2])}"
                )

                selection_result = (
                    await self.nway_service.select_relevant_nway_clusters(test_query, 3)
                )

                test_results["tests"]["nway_selection"] = {
                    "success": bool(selection_result.selected_clusters),
                    "result": f"Selected clusters: {selection_result.selected_clusters}",
                    "confidence": selection_result.confidence_score,
                }

            # Test 3: Database Persistence (if Supabase available)
            if self.supabase_client:
                try:
                    # Query agent_personas table for consultant
                    result = (
                        self.supabase_client.table("agent_personas")
                        .select("*")
                        .eq("consultant_id", consultant_id)
                        .execute()
                    )

                    test_results["tests"]["database_persistence"] = {
                        "success": len(result.data) > 0,
                        "result": f"Database record found: {len(result.data) > 0}",
                    }
                except Exception as e:
                    test_results["tests"]["database_persistence"] = {
                        "success": False,
                        "result": f"Database test failed: {e}",
                    }

            # Calculate overall success
            successful_tests = sum(
                1 for test in test_results["tests"].values() if test["success"]
            )
            total_tests = len(test_results["tests"])
            test_results["overall_success"] = (
                successful_tests == total_tests and total_tests > 0
            )
            test_results["success_rate"] = (
                successful_tests / total_tests if total_tests > 0 else 0.0
            )

            # Glass-Box: Log test completion
            if self.context_stream:
                self.context_stream.add_event(
                    event_type=ContextEventType.TOOL_EXECUTION,
                    data={
                        "operation": "test_consultant",
                        "consultant_id": consultant_id,
                        "overall_success": test_results["overall_success"],
                        "success_rate": test_results["success_rate"],
                    },
                    metadata={
                        "tool": "ConsultantScaffoldingCLI",
                        "execution_id": self.execution_id,
                    },
                )

            self.operations_log.append(
                {
                    "operation": "test_consultant",
                    "consultant_id": consultant_id,
                    "success": test_results["overall_success"],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return test_results

        except Exception as e:
            self.logger.error(f"Error testing consultant {consultant_id}: {e}")
            test_results["tests"]["error"] = {
                "success": False,
                "result": f"Test error: {e}",
            }
            return test_results

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report of CLI execution"""

        return {
            "execution_id": self.execution_id,
            "execution_timestamp": datetime.utcnow().isoformat(),
            "operations_completed": len(self.operations_log),
            "operations_log": self.operations_log,
            "services_initialized": {
                "blueprint_registry": bool(self.blueprint_registry),
                "nway_service": bool(self.nway_service),
                "database_connected": bool(self.supabase_client),
            },
            "available_templates": list(self.consultant_templates.keys()),
            "context_stream_events": (
                len(self.context_stream.events) if self.context_stream else 0
            ),
        }


def create_consultant_scaffolding_cli() -> ConsultantScaffoldingCLI:
    """Factory function to create ConsultantScaffoldingCLI with proper dependencies"""
    return ConsultantScaffoldingCLI()


async def main():
    """CLI entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="METIS V2.1 Consultant Scaffolding CLI - Automate consultant onboarding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scaffold healthcare strategy consultant
  python -m src.tools.consultant_scaffolding_cli scaffold strategy "Healthcare" "digital health transformation"
  
  # Scaffold fintech operations consultant 
  python -m src.tools.consultant_scaffolding_cli scaffold operations "Fintech" "payment processing optimization"
  
  # List available templates
  python -m src.tools.consultant_scaffolding_cli templates
  
  # Test existing consultant
  python -m src.tools.consultant_scaffolding_cli test strategy_healthcare_abc123
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scaffold command
    scaffold_parser = subparsers.add_parser("scaffold", help="Scaffold new consultant")
    scaffold_parser.add_argument(
        "type",
        choices=["strategy", "operations", "technology", "finance", "marketing"],
        help="Consultant type from available templates",
    )
    scaffold_parser.add_argument(
        "specialty", help='Consultant specialty (e.g., "Healthcare", "Fintech")'
    )
    scaffold_parser.add_argument(
        "domain", help='Domain expertise (e.g., "digital health", "payment systems")'
    )
    scaffold_parser.add_argument(
        "--id", help="Custom consultant ID (auto-generated if not provided)"
    )
    scaffold_parser.add_argument(
        "--register",
        action="store_true",
        help="Register consultant with METIS services",
    )
    scaffold_parser.add_argument(
        "--test", action="store_true", help="Test consultant after scaffolding"
    )

    # Templates command
    templates_parser = subparsers.add_parser(
        "templates", help="List available consultant templates"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test existing consultant")
    test_parser.add_argument("consultant_id", help="ID of consultant to test")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI tool
    cli = create_consultant_scaffolding_cli()

    if args.command == "templates":
        print("📋 Available Consultant Templates:")
        print("=" * 50)
        for template_type, config in cli.consultant_templates.items():
            print(f"\n🔹 {template_type.title()}:")
            print(f"   Specialization: {config['specialization']}")
            print(f"   Frameworks: {', '.join(config['frameworks'][:3])}...")
            print(f"   Triggers: {', '.join(config['triggers'])}")
        return

    # Initialize services for other commands
    print("🚀 Initializing METIS services...")
    services_ready = await cli.initialize_services()

    if not services_ready:
        print(
            "❌ Failed to initialize services. Check dependencies and database connection."
        )
        return

    if args.command == "scaffold":
        print(f"🛠️ Scaffolding {args.type} consultant...")
        print(f"   Specialty: {args.specialty}")
        print(f"   Domain: {args.domain}")

        # Scaffold consultant
        consultant_config = cli.scaffold_consultant(
            consultant_type=args.type,
            specialty=args.specialty,
            domain=args.domain,
            consultant_id=args.id,
        )

        print(f"✅ Consultant scaffolded: {consultant_config['consultant_id']}")
        print(f"   Name: {consultant_config['name']}")
        print(f"   Expertise: {consultant_config['expertise']}")

        # Register if requested
        if args.register:
            print("📝 Registering consultant with METIS services...")
            registration_success = await cli.register_consultant(consultant_config)

            if registration_success:
                print("✅ Consultant registered successfully")
            else:
                print("❌ Failed to register consultant")
                return

        # Test if requested
        if args.test:
            print("🧪 Testing consultant integration...")
            test_results = await cli.test_consultant(consultant_config["consultant_id"])

            print(
                f"📊 Test Results (Success Rate: {test_results['success_rate']:.1%}):"
            )
            for test_name, test_result in test_results["tests"].items():
                status = "✅" if test_result["success"] else "❌"
                print(f"   {status} {test_name}: {test_result['result']}")

    elif args.command == "test":
        print(f"🧪 Testing consultant: {args.consultant_id}")

        test_results = await cli.test_consultant(args.consultant_id)

        print(f"📊 Test Results (Success Rate: {test_results['success_rate']:.1%}):")
        for test_name, test_result in test_results["tests"].items():
            status = "✅" if test_result["success"] else "❌"
            print(f"   {status} {test_name}: {test_result['result']}")

    # Generate summary report
    summary = cli.generate_summary_report()
    print("\n📋 Execution Summary:")
    print(f"   Operations Completed: {summary['operations_completed']}")
    print(f"   Context Events Generated: {summary['context_stream_events']}")
    print(f"   Services Initialized: {summary['services_initialized']}")


if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print("❌ METIS dependencies not available")
        print("🛠️ Run from project root: python -m src.tools.consultant_scaffolding_cli")
        sys.exit(1)

    asyncio.run(main())
