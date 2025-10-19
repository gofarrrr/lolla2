import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aiofiles
import logging

logger = logging.getLogger(__name__)


class PresentationStorage:
    """
    Manages storage of generated presentations and metadata
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("data/presentations")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load presentation metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"⚠️ Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    async def save_presentation(
        self,
        generation_id: str,
        format: str,
        result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save presentation result and metadata"""

        # Create presentation directory
        pres_dir = self.storage_dir / generation_id
        pres_dir.mkdir(exist_ok=True)

        # Save result data
        result_file = pres_dir / f"{format}_result.json"
        try:
            async with aiofiles.open(result_file, "w") as f:
                await f.write(json.dumps(result, indent=2, default=str))
        except Exception as e:
            logger.error(f"❌ Failed to save result file: {e}")
            raise

        # Save analysis result if provided
        if analysis_result:
            analysis_file = pres_dir / "analysis_result.json"
            try:
                async with aiofiles.open(analysis_file, "w") as f:
                    await f.write(json.dumps(analysis_result, indent=2, default=str))
            except Exception as e:
                logger.warning(f"⚠️ Failed to save analysis result: {e}")

        # Update metadata
        if generation_id not in self.metadata:
            self.metadata[generation_id] = {
                "created_at": datetime.now().isoformat(),
                "formats": {},
                "analysis_id": (
                    analysis_result.get("analysis_id") if analysis_result else None
                ),
                "problem_statement": (
                    self._extract_problem_statement(analysis_result)
                    if analysis_result
                    else None
                ),
            }

        self.metadata[generation_id]["formats"][format] = {
            "file": str(result_file),
            "gamma_url": result.get("url"),
            "export_url": result.get("exportUrl"),
            "status": result.get("status", "completed"),
            "created_at": datetime.now().isoformat(),
            "theme": result.get("_metadata", {}).get("theme_used"),
            "file_size": result_file.stat().st_size if result_file.exists() else 0,
        }

        # Save metadata
        await self._save_metadata()

        logger.info(f"💾 Saved {format} presentation: {generation_id}")
        return str(result_file)

    async def _save_metadata(self):
        """Save metadata to file"""
        try:
            async with aiofiles.open(self.metadata_file, "w") as f:
                await f.write(json.dumps(self.metadata, indent=2, default=str))
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")

    async def get_presentation(
        self, generation_id: str, format: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve presentation data"""

        if generation_id not in self.metadata:
            return None

        pres_data = self.metadata[generation_id].copy()

        # If specific format requested, load the detailed result
        if format and format in pres_data.get("formats", {}):
            result_file = Path(pres_data["formats"][format]["file"])
            if result_file.exists():
                try:
                    async with aiofiles.open(result_file, "r") as f:
                        content = await f.read()
                        detailed_result = json.loads(content)
                        pres_data["detailed_result"] = detailed_result
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load detailed result: {e}")

        # Load analysis result if available
        analysis_file = self.storage_dir / generation_id / "analysis_result.json"
        if analysis_file.exists():
            try:
                async with aiofiles.open(analysis_file, "r") as f:
                    content = await f.read()
                    pres_data["analysis_result"] = json.loads(content)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load analysis result: {e}")

        return pres_data

    async def list_presentations(
        self, limit: int = 10, offset: int = 0, filter_status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all presentations with metadata"""

        presentations = []
        items = list(self.metadata.items())

        # Sort by creation date (newest first)
        items.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)

        # Apply filtering
        if filter_status:
            filtered_items = []
            for gen_id, data in items:
                formats = data.get("formats", {})
                has_status = any(
                    format_data.get("status") == filter_status
                    for format_data in formats.values()
                )
                if has_status:
                    filtered_items.append((gen_id, data))
            items = filtered_items

        # Apply pagination
        for gen_id, data in items[offset : offset + limit]:
            presentation = {"generation_id": gen_id, **data}
            # Add summary statistics
            formats = data.get("formats", {})
            presentation["format_count"] = len(formats)
            presentation["total_size"] = sum(
                format_data.get("file_size", 0) for format_data in formats.values()
            )
            presentations.append(presentation)

        return presentations

    async def search_presentations(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search presentations by problem statement or analysis content"""

        results = []
        query_lower = query.lower()

        for gen_id, data in self.metadata.items():
            # Search in problem statement
            problem = data.get("problem_statement", "")
            if isinstance(problem, str) and query_lower in problem.lower():
                results.append(
                    {
                        "generation_id": gen_id,
                        "match_type": "problem_statement",
                        "match_text": (
                            problem[:200] + "..." if len(problem) > 200 else problem
                        ),
                        **data,
                    }
                )
                continue

            # Search in analysis ID
            analysis_id = data.get("analysis_id", "")
            if isinstance(analysis_id, str) and query_lower in analysis_id.lower():
                results.append(
                    {
                        "generation_id": gen_id,
                        "match_type": "analysis_id",
                        "match_text": analysis_id,
                        **data,
                    }
                )

        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get storage usage statistics"""

        total_presentations = len(self.metadata)
        total_formats = sum(
            len(data.get("formats", {})) for data in self.metadata.values()
        )

        # Calculate total storage size
        total_size = 0
        for data in self.metadata.values():
            for format_data in data.get("formats", {}).values():
                total_size += format_data.get("file_size", 0)

        # Count by format
        format_counts = {}
        for data in self.metadata.values():
            for format_name in data.get("formats", {}):
                format_counts[format_name] = format_counts.get(format_name, 0) + 1

        # Count by status
        status_counts = {}
        for data in self.metadata.values():
            for format_data in data.get("formats", {}).values():
                status = format_data.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_presentations": total_presentations,
            "total_formats": total_formats,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "format_distribution": format_counts,
            "status_distribution": status_counts,
            "storage_path": str(self.storage_dir),
            "metadata_file_size": (
                self.metadata_file.stat().st_size if self.metadata_file.exists() else 0
            ),
        }

    async def cleanup_old_presentations(self, days: int = 30) -> Dict[str, Any]:
        """Remove presentations older than specified days"""

        cutoff = datetime.now() - timedelta(days=days)
        removed_presentations = 0
        removed_files = 0
        freed_space = 0

        for gen_id, data in list(self.metadata.items()):
            try:
                created = datetime.fromisoformat(data["created_at"])
                if created < cutoff:
                    # Calculate freed space before removal
                    pres_dir = self.storage_dir / gen_id
                    if pres_dir.exists():
                        for file in pres_dir.rglob("*"):
                            if file.is_file():
                                freed_space += file.stat().st_size
                                file.unlink()
                                removed_files += 1
                        pres_dir.rmdir()

                    # Remove from metadata
                    del self.metadata[gen_id]
                    removed_presentations += 1

            except (ValueError, OSError) as e:
                logger.warning(f"⚠️ Failed to cleanup {gen_id}: {e}")

        if removed_presentations > 0:
            await self._save_metadata()
            logger.info(
                f"🧹 Cleaned up {removed_presentations} presentations, {removed_files} files, freed {freed_space/1024/1024:.1f}MB"
            )

        return {
            "removed_presentations": removed_presentations,
            "removed_files": removed_files,
            "freed_space_bytes": freed_space,
            "freed_space_mb": round(freed_space / (1024 * 1024), 2),
            "cutoff_date": cutoff.isoformat(),
        }

    async def export_metadata(self, output_file: Optional[Path] = None) -> Path:
        """Export metadata to a backup file"""

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.storage_dir / f"metadata_backup_{timestamp}.json"

        # Add export metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_presentations": len(self.metadata),
            "presentations": self.metadata,
        }

        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(export_data, indent=2, default=str))

        logger.info(f"📥 Exported metadata to {output_file}")
        return output_file

    async def import_metadata(self, import_file: Path) -> Dict[str, Any]:
        """Import metadata from a backup file"""

        try:
            async with aiofiles.open(import_file, "r") as f:
                content = await f.read()
                import_data = json.loads(content)

            imported_presentations = import_data.get("presentations", {})
            original_count = len(self.metadata)

            # Merge with existing metadata (avoid overwriting)
            for gen_id, data in imported_presentations.items():
                if gen_id not in self.metadata:
                    self.metadata[gen_id] = data

            await self._save_metadata()

            imported_count = len(self.metadata) - original_count

            result = {
                "imported_presentations": imported_count,
                "total_presentations": len(self.metadata),
                "import_file": str(import_file),
                "import_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"📤 Imported {imported_count} presentations from {import_file}"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Failed to import metadata: {e}")
            raise

    def _extract_problem_statement(self, analysis_result: Dict[str, Any]) -> str:
        """Extract problem statement from analysis result"""
        problem = analysis_result.get("problem_statement", "")

        if isinstance(problem, dict):
            return problem.get("problem_description", str(problem))[:500]
        elif isinstance(problem, str):
            return problem[:500]
        else:
            return str(problem)[:500]
