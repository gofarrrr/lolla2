import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from .client import GammaAPIClient
from .builder import PresentationBuilder
from .templates import PresentationType, TemplateEngine
from .storage import PresentationStorage
from .config import GammaConfig
from .exceptions import GammaAPIError, ValidationError

logger = logging.getLogger(__name__)


class GammaPresentationService:
    """
    High-level service for generating presentations from METIS analyses
    Orchestrates the entire presentation generation workflow
    """

    def __init__(
        self,
        config: Optional[GammaConfig] = None,
        storage: Optional[PresentationStorage] = None,
    ):
        self.config = config or GammaConfig()
        self.builder = PresentationBuilder()
        self.storage = storage or PresentationStorage()
        self.template_engine = TemplateEngine()

    async def generate_from_analysis(
        self,
        analysis_result: Dict[str, Any],
        presentation_type: PresentationType = PresentationType.STRATEGY,
        export_formats: List[str] = ["pdf"],
        custom_instructions: Optional[str] = None,
        theme_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate presentation from METIS analysis

        Args:
            analysis_result: METIS cognitive analysis output
            presentation_type: Type of presentation to generate
            export_formats: List of export formats (pdf, pptx)
            custom_instructions: Additional generation instructions
            theme_override: Override default theme

        Returns:
            Dictionary containing presentation URLs and metadata
        """

        generation_id = str(uuid.uuid4())
        logger.info(f"🚀 Starting presentation generation: {generation_id}")

        try:
            # Validate analysis result
            if not self.builder.validate_analysis_result(analysis_result):
                raise ValidationError("Analysis result is missing required fields")

            # Build content from analysis
            gamma_content = self.builder.build_from_analysis(
                analysis_result, presentation_type
            )

            # Apply custom instructions and theme overrides
            if custom_instructions:
                gamma_content["additional_instructions"] = (
                    gamma_content.get("additional_instructions", "")
                    + f" {custom_instructions}"
                )

            if theme_override:
                gamma_content["theme_name"] = theme_override

            # Generate presentations for each format
            results = {}
            generation_errors = []

            async with GammaAPIClient(self.config) as client:
                for format in export_formats:
                    try:
                        logger.info(f"📊 Generating {format.upper()} format...")

                        result = await client.generate_presentation(
                            input_text=gamma_content["input_text"],
                            text_mode=gamma_content.get("text_mode", "generate"),
                            theme_name=gamma_content.get("theme_name"),
                            num_cards=gamma_content.get("num_cards"),
                            additional_instructions=gamma_content.get(
                                "additional_instructions"
                            ),
                            text_options=gamma_content.get("text_options"),
                            image_options=gamma_content.get("image_options"),
                            export_as=format if format in ["pdf", "pptx"] else None,
                        )

                        results[format] = result

                        # Store the result
                        await self.storage.save_presentation(
                            generation_id, format, result, analysis_result
                        )

                    except Exception as e:
                        logger.error(f"❌ Failed to generate {format}: {e}")
                        generation_errors.append({"format": format, "error": str(e)})

            # Check if any formats succeeded
            if not results:
                raise GammaAPIError("All format generations failed")

            # Extract URLs for response
            urls = {}
            for format, result in results.items():
                # Try different URL fields
                url = (
                    result.get("url")
                    or result.get("exportUrl")
                    or result.get("shareUrl")
                    or result.get("webUrl")
                )
                if url:
                    urls[format] = url

            # Compile final response
            response = {
                "generation_id": generation_id,
                "timestamp": datetime.now().isoformat(),
                "presentation_type": presentation_type.value,
                "analysis_id": analysis_result.get("analysis_id"),
                "formats": results,
                "urls": urls,
                "status": "success" if not generation_errors else "partial_success",
                "message": f"Successfully generated {len(results)} presentation format(s)",
                "errors": generation_errors if generation_errors else None,
                "theme_used": gamma_content.get("theme_name"),
                "card_count": gamma_content.get("num_cards"),
            }

            logger.info(f"✅ Presentation generation complete: {generation_id}")
            return response

        except Exception as e:
            logger.error(f"❌ Presentation generation failed: {e}")
            return {
                "generation_id": generation_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "presentation_type": presentation_type.value,
                "analysis_id": analysis_result.get("analysis_id"),
            }

    async def generate_batch(
        self,
        analyses: List[Dict[str, Any]],
        presentation_type: PresentationType = PresentationType.STRATEGY,
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate multiple presentations in batch with concurrency control"""

        logger.info(f"📚 Batch generation started for {len(analyses)} analyses")

        # Create semaphore to limit concurrent generations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(analysis):
            async with semaphore:
                return await self.generate_from_analysis(analysis, presentation_type)

        # Execute with controlled concurrency
        tasks = [generate_with_semaphore(analysis) for analysis in analyses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and count successes
        processed_results = []
        successful = 0

        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "status": "error",
                        "error": str(result),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                processed_results.append(result)
                if result.get("status") in ["success", "partial_success"]:
                    successful += 1

        logger.info(f"✅ Batch complete: {successful}/{len(analyses)} successful")

        return processed_results

    async def regenerate_with_feedback(
        self,
        generation_id: str,
        feedback: str,
        presentation_type: Optional[PresentationType] = None,
        export_formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Regenerate presentation with user feedback"""

        logger.info(f"🔄 Regenerating {generation_id} with feedback")

        # Retrieve original presentation data
        original = await self.storage.get_presentation(generation_id)

        if not original:
            return {
                "status": "error",
                "error": "Original generation not found",
                "generation_id": generation_id,
            }

        # Get original analysis result
        analysis_result = original.get("analysis_result")
        if not analysis_result:
            return {
                "status": "error",
                "error": "Original analysis result not found",
                "generation_id": generation_id,
            }

        # Prepare feedback-enhanced instructions
        custom_instructions = f"""
        User Feedback: {feedback}
        
        Please adjust the presentation based on this feedback while maintaining professional quality and structure.
        """

        # Use original settings or provided overrides
        final_presentation_type = presentation_type or PresentationType(
            original.get("presentation_type", "strategy")
        )
        final_export_formats = (
            export_formats or list(original.get("formats", {}).keys()) or ["pdf"]
        )

        # Regenerate with feedback
        return await self.generate_from_analysis(
            analysis_result,
            final_presentation_type,
            final_export_formats,
            custom_instructions,
        )

    async def get_presentation_status(self, generation_id: str) -> Dict[str, Any]:
        """Get detailed status of a presentation generation"""

        presentation = await self.storage.get_presentation(generation_id)

        if not presentation:
            return {"status": "not_found", "generation_id": generation_id}

        # Check format statuses
        formats_status = {}
        for format_name, format_data in presentation.get("formats", {}).items():
            formats_status[format_name] = {
                "status": format_data.get("status", "unknown"),
                "url": format_data.get("gamma_url"),
                "export_url": format_data.get("export_url"),
                "created_at": format_data.get("created_at"),
                "file_size": format_data.get("file_size", 0),
            }

        return {
            "generation_id": generation_id,
            "created_at": presentation.get("created_at"),
            "analysis_id": presentation.get("analysis_id"),
            "problem_statement": presentation.get("problem_statement"),
            "formats": formats_status,
            "total_formats": len(formats_status),
            "status": "completed" if formats_status else "unknown",
        }

    async def list_templates(self) -> List[Dict[str, Any]]:
        """List available presentation templates"""

        templates = []
        for pres_type in PresentationType:
            template = self.template_engine.get_template(pres_type)
            templates.append(
                {
                    "type": pres_type.value,
                    "name": template.get("name", pres_type.name.title()),
                    "description": self.template_engine.get_template_description(
                        pres_type
                    ),
                    "theme": template.get("theme"),
                    "format": template.get("format"),
                    "typical_cards": template.get("num_cards", "auto"),
                }
            )

        return templates

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the Gamma presentation service"""

        health_data = {
            "service_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check storage
        try:
            storage_stats = await self.storage.get_usage_statistics()
            health_data["components"]["storage"] = {
                "status": "healthy",
                "presentations": storage_stats["total_presentations"],
                "size_mb": storage_stats["total_size_mb"],
            }
        except Exception as e:
            health_data["components"]["storage"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_data["service_status"] = "degraded"

        # Check Gamma API
        try:
            async with GammaAPIClient(self.config) as client:
                api_health = await client.health_check()
                health_data["components"]["gamma_api"] = api_health

                if api_health["status"] != "healthy":
                    health_data["service_status"] = "degraded"

        except Exception as e:
            health_data["components"]["gamma_api"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_data["service_status"] = "unhealthy"

        # Check templates
        try:
            templates = await self.list_templates()
            health_data["components"]["templates"] = {
                "status": "healthy",
                "count": len(templates),
            }
        except Exception as e:
            health_data["components"]["templates"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_data["service_status"] = "degraded"

        return health_data

    async def cleanup_old_presentations(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old presentations"""
        return await self.storage.cleanup_old_presentations(days)

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        return {
            "service": "GammaPresentationService",
            "version": "1.0.0",
            "uptime": datetime.now().isoformat(),
            "config": {
                "default_theme": self.config.default_theme,
                "max_generations_per_month": self.config.max_generations_per_month,
                "storage_dir": str(self.config.storage_dir),
            },
        }
