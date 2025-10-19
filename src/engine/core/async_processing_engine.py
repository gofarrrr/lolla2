#!/usr/bin/env python3
"""
METIS Async Processing Engine
Enterprise-grade background analysis system with job queues, progress tracking,
and result caching for long-running cognitive tasks.

Key Features:
- Redis-backed job queue for reliable processing
- Progressive result streaming during background execution
- Multi-priority task scheduling (immediate, normal, background)
- Comprehensive failure handling with exponential backoff
- Real-time progress monitoring and cancellation support
- Integration with existing METIS components (caching, checkpointing, HITL)
"""

import asyncio
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict


# Redis imports (mock for demo - would use real Redis in production)
class MockRedis:
    def __init__(self):
        self.data = {}
        self.queues = defaultdict(list)
        self.subscribers = defaultdict(list)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        self.data[key] = {"value": value, "expires": time.time() + ex if ex else None}

    async def get(self, key: str) -> Optional[str]:
        item = self.data.get(key)
        if item and (item["expires"] is None or item["expires"] > time.time()):
            return item["value"]
        return None

    async def lpush(self, queue: str, item: str):
        self.queues[queue].append(item)

    async def brpop(self, queues: List[str], timeout: float = 1.0):
        for queue in queues:
            if self.queues[queue]:
                return queue, self.queues[queue].pop(0)
        await asyncio.sleep(min(timeout, 0.1))
        return None

    async def publish(self, channel: str, message: str):
        # Mock publish - in real Redis would use pub/sub
        pass


class TaskPriority(Enum):
    """Task priority levels"""

    IMMEDIATE = "immediate"  # <5s response time, blocks other tasks
    NORMAL = "normal"  # 5-30s response time, standard queue
    BACKGROUND = "background"  # >30s response time, runs when resources available
    RESEARCH = "research"  # Long-running research tasks, lowest priority


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskType(Enum):
    """Types of async tasks"""

    COGNITIVE_ANALYSIS = "cognitive_analysis"
    RESEARCH_SYNTHESIS = "research_synthesis"
    TREE_SEARCH = "tree_search"
    MENTAL_MODEL_APPLICATION = "mental_model_application"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    SCENARIO_ANALYSIS = "scenario_analysis"


@dataclass
class AsyncTask:
    """Represents an asynchronous processing task"""

    task_id: str
    task_type: TaskType
    priority: TaskPriority
    title: str
    description: str

    # Task parameters
    input_data: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Progress tracking
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 1
    completed_steps: int = 0

    # Results and error handling
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Resource usage
    estimated_duration_seconds: float = 30.0
    actual_duration_seconds: Optional[float] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to values
        data["task_type"] = self.task_type.value
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AsyncTask":
        """Create task from dictionary"""
        task = cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            priority=TaskPriority(data["priority"]),
            title=data["title"],
            description=data["description"],
            input_data=data["input_data"],
        )

        # Restore other fields
        for field_name, value in data.items():
            if hasattr(task, field_name) and field_name not in [
                "task_type",
                "priority",
            ]:
                if field_name == "status":
                    setattr(task, field_name, TaskStatus(value))
                else:
                    setattr(task, field_name, value)

        return task

    def update_progress(self, percentage: float, step_name: str = ""):
        """Update task progress"""
        self.progress_percentage = min(100.0, max(0.0, percentage))
        if step_name:
            self.current_step = step_name

        # Update completed steps if progress increased significantly
        expected_completed = int((self.progress_percentage / 100.0) * self.total_steps)
        if expected_completed > self.completed_steps:
            self.completed_steps = expected_completed

    def calculate_estimated_completion(self) -> Optional[float]:
        """Calculate estimated completion time based on progress"""
        if not self.started_at or self.progress_percentage <= 0:
            return None

        elapsed = time.time() - self.started_at
        if self.progress_percentage > 0:
            total_estimated = elapsed / (self.progress_percentage / 100.0)
            return self.started_at + total_estimated

        return None


class TaskProcessor(ABC):
    """Abstract base class for task processors"""

    @abstractmethod
    async def process_task(self, task: AsyncTask) -> Dict[str, Any]:
        """Process a specific task and return results"""
        pass

    @abstractmethod
    def get_supported_task_types(self) -> List[TaskType]:
        """Return list of task types this processor can handle"""
        pass

    @abstractmethod
    async def estimate_duration(self, task: AsyncTask) -> float:
        """Estimate task duration in seconds"""
        pass


class CognitiveAnalysisProcessor(TaskProcessor):
    """Processes cognitive analysis tasks"""

    async def process_task(self, task: AsyncTask) -> Dict[str, Any]:
        """Process cognitive analysis task"""
        input_data = task.input_data

        # Simulate cognitive analysis steps
        steps = [
            ("problem_structuring", "Structuring the problem using MECE framework"),
            ("hypothesis_generation", "Generating testable hypotheses"),
            ("mental_model_selection", "Selecting appropriate mental models"),
            ("analysis_execution", "Executing deep analysis"),
            ("synthesis", "Synthesizing insights and recommendations"),
        ]

        task.total_steps = len(steps)
        insights = []

        for i, (phase, description) in enumerate(steps):
            task.update_progress((i / len(steps)) * 100, description)

            # Simulate processing time
            await asyncio.sleep(0.5)

            # Generate simulated insights
            insight = (
                f"Insight from {phase}: {description} reveals strategic implications"
            )
            insights.append(
                {
                    "phase": phase,
                    "insight": insight,
                    "confidence": 0.7 + (i * 0.05),
                    "timestamp": time.time(),
                }
            )

        task.update_progress(100.0, "Analysis completed")

        return {
            "analysis_type": "cognitive_analysis",
            "insights": insights,
            "confidence_score": sum(i["confidence"] for i in insights) / len(insights),
            "methodology": "MeMo mental models framework",
            "processing_time": time.time() - (task.started_at or time.time()),
        }

    def get_supported_task_types(self) -> List[TaskType]:
        return [TaskType.COGNITIVE_ANALYSIS, TaskType.MENTAL_MODEL_APPLICATION]

    async def estimate_duration(self, task: AsyncTask) -> float:
        return 15.0  # 15 seconds for cognitive analysis


class ResearchSynthesisProcessor(TaskProcessor):
    """Processes research synthesis tasks"""

    async def process_task(self, task: AsyncTask) -> Dict[str, Any]:
        """Process research synthesis task"""
        steps = [
            ("source_collection", "Collecting relevant research sources"),
            ("credibility_assessment", "Assessing source credibility"),
            ("content_extraction", "Extracting key information"),
            ("synthesis", "Synthesizing research findings"),
            ("validation", "Validating conclusions"),
        ]

        task.total_steps = len(steps)
        sources = []

        for i, (phase, description) in enumerate(steps):
            task.update_progress((i / len(steps)) * 100, description)
            await asyncio.sleep(0.3)

            # Simulate research source processing
            sources.append(
                {
                    "source_id": f"source_{i+1}",
                    "title": f"Research finding from {phase}",
                    "credibility": 0.8 + (i * 0.03),
                    "relevance": 0.75 + (i * 0.04),
                }
            )

        task.update_progress(100.0, "Research synthesis completed")

        return {
            "research_type": "synthesis",
            "sources_processed": len(sources),
            "sources": sources,
            "synthesis_confidence": 0.82,
            "methodology": "Multi-source triangulation",
            "processing_time": time.time() - (task.started_at or time.time()),
        }

    def get_supported_task_types(self) -> List[TaskType]:
        return [TaskType.RESEARCH_SYNTHESIS]

    async def estimate_duration(self, task: AsyncTask) -> float:
        return 25.0  # 25 seconds for research synthesis


@dataclass
class ProcessingStats:
    """Statistics for async processing engine"""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_processing_time: float = 0.0
    active_tasks: int = 0
    queue_sizes: Dict[str, int] = field(default_factory=dict)


class AsyncProcessingEngine:
    """
    Enterprise async processing engine for background cognitive analysis
    """

    def __init__(
        self, redis_client: Optional[MockRedis] = None, max_concurrent_tasks: int = 5
    ):
        self.redis = redis_client or MockRedis()
        self.max_concurrent_tasks = max_concurrent_tasks

        # Task management
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.task_processors: Dict[TaskType, TaskProcessor] = {}
        self.queue_names = {
            TaskPriority.IMMEDIATE: "metis:queue:immediate",
            TaskPriority.NORMAL: "metis:queue:normal",
            TaskPriority.BACKGROUND: "metis:queue:background",
            TaskPriority.RESEARCH: "metis:queue:research",
        }

        # Processing control
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.stats = ProcessingStats()

        # Register default processors
        self._register_default_processors()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _register_default_processors(self):
        """Register default task processors"""
        cognitive_processor = CognitiveAnalysisProcessor()
        for task_type in cognitive_processor.get_supported_task_types():
            self.task_processors[task_type] = cognitive_processor

        research_processor = ResearchSynthesisProcessor()
        for task_type in research_processor.get_supported_task_types():
            self.task_processors[task_type] = research_processor

    async def start(self):
        """Start the async processing engine"""
        if self.is_running:
            return

        self.is_running = True
        self.logger.info(
            f"Starting async processing engine with {self.max_concurrent_tasks} workers"
        )

        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(worker_task)

        print(
            f"🚀 Async processing engine started with {self.max_concurrent_tasks} workers"
        )

    async def stop(self):
        """Stop the async processing engine"""
        self.is_running = False

        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for workers to stop
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()

        print("⏹️ Async processing engine stopped")

    async def submit_task(
        self,
        task_type: TaskType,
        priority: TaskPriority,
        title: str,
        description: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a new task for async processing"""

        task_id = f"task_{int(time.time()*1000)}_{str(uuid.uuid4())[:8]}"

        # Get processor to estimate duration
        processor = self.task_processors.get(task_type)
        estimated_duration = 30.0  # Default
        if processor:
            task_stub = AsyncTask(
                task_id, task_type, priority, title, description, input_data
            )
            estimated_duration = await processor.estimate_duration(task_stub)

        # Create task
        task = AsyncTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            title=title,
            description=description,
            input_data=input_data,
            context=context or {},
            estimated_duration_seconds=estimated_duration,
        )

        # Store task
        await self.redis.set(
            f"metis:task:{task_id}", json.dumps(task.to_dict()), ex=3600
        )

        # Add to appropriate queue
        queue_name = self.queue_names[priority]
        await self.redis.lpush(queue_name, task_id)

        task.status = TaskStatus.QUEUED
        self.stats.total_tasks += 1

        print(f"📋 Task submitted: {task_id} ({task_type.value}, {priority.value})")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task"""
        # Check active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()

        # Check Redis storage
        task_data = await self.redis.get(f"metis:task:{task_id}")
        if task_data:
            task_dict = json.loads(task_data)
            return task_dict

        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        # Check if task is active
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            # In real implementation, would signal worker to stop
            print(f"🚫 Task cancelled: {task_id}")
            return True

        # Check if task exists in storage
        task_data = await self.redis.get(f"metis:task:{task_id}")
        if task_data:
            task_dict = json.loads(task_data)
            task_dict["status"] = TaskStatus.CANCELLED.value
            await self.redis.set(
                f"metis:task:{task_id}", json.dumps(task_dict), ex=3600
            )
            return True

        return False

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        stats = {
            "active_tasks": len(self.active_tasks),
            "total_processed": self.stats.total_tasks,
            "success_rate": (
                self.stats.completed_tasks / max(self.stats.total_tasks, 1)
            )
            * 100,
            "average_processing_time": self.stats.average_processing_time,
            "queue_sizes": {},
            "task_types_active": defaultdict(int),
        }

        # Count active task types
        for task in self.active_tasks.values():
            stats["task_types_active"][task.task_type.value] += 1

        return stats

    async def _worker_loop(self, worker_id: str):
        """Main worker loop for processing tasks"""
        print(f"👷 Worker {worker_id} started")

        while self.is_running:
            try:
                # Try to get task from priority queues
                queue_list = [
                    self.queue_names[TaskPriority.IMMEDIATE],
                    self.queue_names[TaskPriority.NORMAL],
                    self.queue_names[TaskPriority.BACKGROUND],
                    self.queue_names[TaskPriority.RESEARCH],
                ]

                result = await self.redis.brpop(queue_list, timeout=1.0)
                if not result:
                    continue

                queue_name, task_id = result

                # Load task data
                task_data = await self.redis.get(f"metis:task:{task_id}")
                if not task_data:
                    continue

                task = AsyncTask.from_dict(json.loads(task_data))

                # Check if task was cancelled
                if task.status == TaskStatus.CANCELLED:
                    continue

                # Process task
                await self._process_task(task, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)

        print(f"👷 Worker {worker_id} stopped")

    async def _process_task(self, task: AsyncTask, worker_id: str):
        """Process a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.active_tasks[task.task_id] = task

        print(f"⚙️ {worker_id} processing: {task.task_id} ({task.task_type.value})")

        try:
            # Get appropriate processor
            processor = self.task_processors.get(task.task_type)
            if not processor:
                raise ValueError(f"No processor available for {task.task_type.value}")

            # Process task
            result = await processor.process_task(task)

            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            task.actual_duration_seconds = task.completed_at - task.started_at

            # Update stats
            self.stats.completed_tasks += 1

            print(
                f"✅ {worker_id} completed: {task.task_id} ({task.actual_duration_seconds:.1f}s)"
            )

        except Exception as e:
            # Handle failure
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error_message = str(e)
            task.retry_count += 1

            self.stats.failed_tasks += 1

            print(f"❌ {worker_id} failed: {task.task_id} - {e}")

            # Retry logic
            if task.retry_count < task.max_retries and task.priority in [
                TaskPriority.IMMEDIATE,
                TaskPriority.NORMAL,
            ]:
                await asyncio.sleep(2**task.retry_count)  # Exponential backoff
                task.status = TaskStatus.RETRYING
                queue_name = self.queue_names[task.priority]
                await self.redis.lpush(queue_name, task.task_id)

        finally:
            # Update task storage
            await self.redis.set(
                f"metis:task:{task.task_id}", json.dumps(task.to_dict()), ex=3600
            )

            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def stream_task_progress(
        self, task_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time progress updates for a task"""
        while True:
            task_status = await self.get_task_status(task_id)
            if not task_status:
                break

            yield {
                "task_id": task_id,
                "status": task_status["status"],
                "progress": task_status["progress_percentage"],
                "current_step": task_status["current_step"],
                "completed_steps": task_status["completed_steps"],
                "total_steps": task_status["total_steps"],
                "estimated_completion": task_status.get("estimated_completion"),
            }

            # Stop streaming if task is complete
            if task_status["status"] in ["completed", "failed", "cancelled"]:
                break

            await asyncio.sleep(0.5)  # Update every 500ms


def get_async_processing_engine(max_concurrent_tasks: int = 5) -> AsyncProcessingEngine:
    """Factory function to get configured async processing engine"""
    return AsyncProcessingEngine(max_concurrent_tasks=max_concurrent_tasks)


# Example usage and testing
if __name__ == "__main__":

    async def test_async_processing():
        """Test the async processing engine"""
        print("🔄 Testing METIS Async Processing Engine")

        # Create and start engine
        engine = get_async_processing_engine(max_concurrent_tasks=3)
        await engine.start()

        try:
            # Submit various tasks
            task1 = await engine.submit_task(
                TaskType.COGNITIVE_ANALYSIS,
                TaskPriority.NORMAL,
                "Strategic Analysis: Market Expansion",
                "Analyze market expansion opportunities using cognitive frameworks",
                {
                    "problem": "Should we expand to B2B market?",
                    "industry": "E-commerce",
                },
            )

            task2 = await engine.submit_task(
                TaskType.RESEARCH_SYNTHESIS,
                TaskPriority.BACKGROUND,
                "Research: Competitor Analysis",
                "Synthesize competitive intelligence research",
                {
                    "competitors": ["Company A", "Company B"],
                    "market": "Corporate gifting",
                },
            )

            print(f"Tasks submitted: {task1}, {task2}")

            # Monitor progress for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                stats = await engine.get_queue_stats()
                print(
                    f"  Active: {stats['active_tasks']}, Completed: {stats['total_processed']}"
                )

                # Check task statuses
                status1 = await engine.get_task_status(task1)
                status2 = await engine.get_task_status(task2)

                if status1:
                    print(
                        f"  Task1: {status1['status']} ({status1['progress_percentage']:.1f}%)"
                    )
                if status2:
                    print(
                        f"  Task2: {status2['status']} ({status2['progress_percentage']:.1f}%)"
                    )

                await asyncio.sleep(2)

            print("✅ Async processing test completed")

        finally:
            await engine.stop()

    # Run test
    asyncio.run(test_async_processing())
