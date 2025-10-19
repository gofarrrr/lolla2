"""
METIS Customer Onboarding System
D002: Enterprise-grade customer onboarding with self-service and white-glove options

Implements comprehensive customer onboarding workflow with tenant provisioning,
user setup, training modules, and success metrics tracking.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any

# Email imports with Python 3.13 compatibility
try:
    from email.mime.text import MIMEText as MimeText
    from email.mime.multipart import MIMEMultipart as MimeMultipart
    from email.mime.base import MIMEBase as MimeBase

    EMAIL_AVAILABLE = True
except ImportError:
    try:
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase

        # Create aliases
        MimeText = MIMEText
        MimeMultipart = MIMEMultipart
        MimeBase = MIMEBase
        EMAIL_AVAILABLE = True
    except ImportError:
        print("Warning: email modules not available, using fallback")
        EMAIL_AVAILABLE = False

        # Mock email classes
        class MockMimeText:
            def __init__(self, text, subtype="plain"):
                self.text = text
                self.subtype = subtype

        class MockMimeMultipart:
            def __init__(self, subtype="mixed"):
                self.subtype = subtype
                self.parts = []

            def attach(self, part):
                self.parts.append(part)

        MimeText = MockMimeText
        MimeMultipart = MockMimeMultipart
        MimeBase = None
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import os
import smtplib

# Email import with Python 3.13 compatibility
try:
    from email.mime.text import MIMEText as MimeText
except ImportError:
    from email.mime.text import MIMEText

    MimeText = MIMEText


try:
    import stripe

    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail

    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    sendgrid = None


class OnboardingType(str, Enum):
    """Types of customer onboarding"""

    SELF_SERVICE = "self_service"
    GUIDED = "guided"
    WHITE_GLOVE = "white_glove"
    ENTERPRISE = "enterprise"


class OnboardingStage(str, Enum):
    """Onboarding workflow stages"""

    REGISTRATION = "registration"
    TENANT_PROVISIONING = "tenant_provisioning"
    ACCOUNT_SETUP = "account_setup"
    USER_CONFIGURATION = "user_configuration"
    TRAINING = "training"
    FIRST_ENGAGEMENT = "first_engagement"
    SUCCESS_METRICS = "success_metrics"
    COMPLETED = "completed"


class OnboardingStatus(str, Enum):
    """Status of onboarding process"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubscriptionTier(str, Enum):
    """METIS subscription tiers"""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"


@dataclass
class OnboardingConfiguration:
    """Configuration for different onboarding types"""

    onboarding_type: OnboardingType = OnboardingType.SELF_SERVICE
    subscription_tier: SubscriptionTier = SubscriptionTier.PROFESSIONAL

    # Workflow customization
    skip_training: bool = False
    custom_training_modules: List[str] = field(default_factory=list)
    dedicated_success_manager: bool = False

    # Integration settings
    enable_sso: bool = False
    enable_api_access: bool = True
    enable_custom_frameworks: bool = False

    # Resource limits
    max_users: int = 10
    max_engagements_per_month: int = 100
    storage_limit_gb: int = 50

    # Features
    advanced_analytics: bool = True
    white_label: bool = False
    priority_support: bool = False

    # Customization
    custom_branding: Dict[str, str] = field(default_factory=dict)
    custom_domains: List[str] = field(default_factory=list)


@dataclass
class OnboardingStep:
    """Individual step in onboarding workflow"""

    step_id: str = ""
    stage: OnboardingStage = OnboardingStage.REGISTRATION
    title: str = ""
    description: str = ""

    # Status
    status: OnboardingStatus = OnboardingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Requirements
    required: bool = True
    estimated_duration_minutes: int = 15
    prerequisites: List[str] = field(default_factory=list)

    # Content
    instructions: str = ""
    resources: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)

    # Automation
    automated: bool = False
    automation_function: Optional[str] = None

    # Progress tracking
    progress_percentage: int = 0
    user_feedback: Optional[str] = None
    completion_notes: str = ""


@dataclass
class CustomerProfile:
    """Customer information for onboarding"""

    customer_id: str = field(default_factory=lambda: str(uuid4()))

    # Company information
    company_name: str = ""
    company_size: str = ""  # startup, small, medium, large, enterprise
    industry: str = ""
    use_case: str = ""

    # Primary contact
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    primary_contact_title: str = ""
    primary_contact_phone: str = ""

    # Billing information
    billing_email: str = ""
    billing_address: Dict[str, str] = field(default_factory=dict)
    payment_method_id: Optional[str] = None

    # Technical information
    technical_contact_email: str = ""
    existing_tools: List[str] = field(default_factory=list)
    integration_requirements: List[str] = field(default_factory=list)

    # Onboarding preferences
    preferred_onboarding_type: OnboardingType = OnboardingType.SELF_SERVICE
    preferred_timeline: str = "standard"  # fast, standard, extended
    training_preferences: List[str] = field(default_factory=list)

    # Success criteria
    success_metrics: List[str] = field(default_factory=list)
    business_objectives: List[str] = field(default_factory=list)

    # Created metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    referral_source: str = ""


@dataclass
class OnboardingWorkflow:
    """Complete onboarding workflow for a customer"""

    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    customer_id: str = ""

    # Configuration
    config: OnboardingConfiguration = field(default_factory=OnboardingConfiguration)

    # Status
    status: OnboardingStatus = OnboardingStatus.PENDING
    current_stage: OnboardingStage = OnboardingStage.REGISTRATION
    progress_percentage: int = 0

    # Timeline
    started_at: Optional[datetime] = None
    expected_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Steps
    steps: List[OnboardingStep] = field(default_factory=list)

    # Resources created
    tenant_id: Optional[str] = None
    admin_user_id: Optional[str] = None
    subscription_id: Optional[str] = None

    # Support
    assigned_success_manager: Optional[str] = None
    support_ticket_ids: List[str] = field(default_factory=list)

    # Metrics
    engagement_count: int = 0
    training_modules_completed: int = 0
    first_engagement_date: Optional[datetime] = None
    time_to_value_days: Optional[int] = None

    # Feedback
    customer_satisfaction_score: Optional[int] = None
    feedback_comments: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)


class EmailNotificationService:
    """Email notification service for onboarding"""

    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

        # SendGrid fallback
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY", "")

        self.logger = logging.getLogger(__name__)

    async def send_welcome_email(
        self, customer_email: str, customer_name: str, onboarding_link: str
    ) -> bool:
        """Send welcome email to new customer"""

        subject = "Welcome to METIS - Your Cognitive Intelligence Platform"

        body = f"""
        Dear {customer_name},
        
        Welcome to METIS! We're excited to help you unlock the power of AI-native consulting.
        
        Your onboarding journey starts here: {onboarding_link}
        
        What's next:
        1. Complete your account setup
        2. Provision your dedicated tenant
        3. Complete the interactive training modules
        4. Run your first cognitive engagement
        
        Our team is here to support you every step of the way.
        
        Best regards,
        The METIS Team
        """

        return await self._send_email(customer_email, subject, body)

    async def send_step_completion_notification(
        self,
        customer_email: str,
        customer_name: str,
        step_title: str,
        next_step_title: str,
    ) -> bool:
        """Send notification when onboarding step is completed"""

        subject = f"METIS Onboarding Progress: {step_title} Complete"

        body = f"""
        Hi {customer_name},
        
        Great progress! You've successfully completed: {step_title}
        
        Next up: {next_step_title}
        
        Continue your onboarding journey at: https://app.metis.ai/onboarding
        
        Best regards,
        The METIS Team
        """

        return await self._send_email(customer_email, subject, body)

    async def send_completion_email(
        self, customer_email: str, customer_name: str, dashboard_link: str
    ) -> bool:
        """Send completion email when onboarding is finished"""

        subject = "METIS Onboarding Complete - You're Ready to Go!"

        body = f"""
        Congratulations {customer_name}!
        
        You've successfully completed your METIS onboarding. Your cognitive intelligence platform is ready.
        
        Access your dashboard: {dashboard_link}
        
        Quick start tips:
        - Create your first engagement
        - Explore our mental models library
        - Try the hypothesis generation engine
        - Generate your first executive deliverable
        
        Need help? Contact your success manager or visit our help center.
        
        Welcome to the future of consulting!
        
        The METIS Team
        """

        return await self._send_email(customer_email, subject, body)

    async def _send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email using available service"""

        try:
            if SENDGRID_AVAILABLE and self.sendgrid_api_key:
                return await self._send_via_sendgrid(to_email, subject, body)
            else:
                return await self._send_via_smtp(to_email, subject, body)
        except Exception as e:
            self.logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    async def _send_via_sendgrid(self, to_email: str, subject: str, body: str) -> bool:
        """Send email via SendGrid"""

        try:
            sg = sendgrid.SendGridAPIClient(api_key=self.sendgrid_api_key)

            message = Mail(
                from_email="noreply@metis.ai",
                to_emails=to_email,
                subject=subject,
                plain_text_content=body,
            )

            response = sg.send(message)
            return response.status_code < 300

        except Exception as e:
            self.logger.error(f"SendGrid error: {str(e)}")
            return False

    async def _send_via_smtp(self, to_email: str, subject: str, body: str) -> bool:
        """Send email via SMTP"""

        try:
            msg = MimeMultipart()
            msg["From"] = self.smtp_username
            msg["To"] = to_email
            msg["Subject"] = subject

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)

            text = msg.as_string()
            server.sendmail(self.smtp_username, to_email, text)
            server.quit()

            return True

        except Exception as e:
            self.logger.error(f"SMTP error: {str(e)}")
            return False


class TenantProvisioningService:
    """Service for provisioning customer tenants"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def provision_tenant(
        self, customer_profile: CustomerProfile, config: OnboardingConfiguration
    ) -> Dict[str, Any]:
        """Provision new tenant for customer"""

        tenant_id = str(uuid4())

        try:
            # Create tenant configuration
            tenant_config = {
                "tenant_id": tenant_id,
                "name": customer_profile.company_name,
                "domain": customer_profile.company_name.lower().replace(" ", "-"),
                "subscription_tier": config.subscription_tier.value,
                "max_users": config.max_users,
                "max_engagements_per_month": config.max_engagements_per_month,
                "storage_limit_gb": config.storage_limit_gb,
                "features": {
                    "advanced_analytics": config.advanced_analytics,
                    "api_access": config.enable_api_access,
                    "custom_frameworks": config.enable_custom_frameworks,
                    "white_label": config.white_label,
                    "sso": config.enable_sso,
                },
                "branding": config.custom_branding,
                "custom_domains": config.custom_domains,
            }

            # Create database schema
            await self._create_tenant_schema(tenant_id)

            # Setup tenant infrastructure
            await self._setup_tenant_infrastructure(tenant_id, tenant_config)

            # Configure security settings
            await self._configure_tenant_security(tenant_id, config)

            # Initialize default data
            await self._initialize_tenant_data(tenant_id, customer_profile)

            self.logger.info(
                f"Tenant {tenant_id} provisioned successfully for {customer_profile.company_name}"
            )

            return {
                "tenant_id": tenant_id,
                "status": "provisioned",
                "config": tenant_config,
                "provisioned_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to provision tenant: {str(e)}")
            return {
                "tenant_id": tenant_id,
                "status": "failed",
                "error": str(e),
                "attempted_at": datetime.utcnow().isoformat(),
            }

    async def _create_tenant_schema(self, tenant_id: str):
        """Create database schema for tenant"""
        # Would create isolated database schema
        pass

    async def _setup_tenant_infrastructure(
        self, tenant_id: str, config: Dict[str, Any]
    ):
        """Setup infrastructure components for tenant"""
        # Would setup Kafka topics, Redis namespace, etc.
        pass

    async def _configure_tenant_security(
        self, tenant_id: str, config: OnboardingConfiguration
    ):
        """Configure security settings for tenant"""
        # Would setup encryption keys, access policies, etc.
        pass

    async def _initialize_tenant_data(self, tenant_id: str, profile: CustomerProfile):
        """Initialize default data for tenant"""
        # Would create default mental models, templates, etc.
        pass


class OnboardingWorkflowEngine:
    """Engine for managing customer onboarding workflows"""

    def __init__(self):
        self.workflows: Dict[str, OnboardingWorkflow] = {}
        self.customer_profiles: Dict[str, CustomerProfile] = {}

        self.email_service = EmailNotificationService()
        self.provisioning_service = TenantProvisioningService()

        self.logger = logging.getLogger(__name__)

    async def initiate_onboarding(
        self,
        customer_profile: CustomerProfile,
        config: Optional[OnboardingConfiguration] = None,
    ) -> OnboardingWorkflow:
        """Initiate new customer onboarding workflow"""

        if config is None:
            config = self._determine_onboarding_config(customer_profile)

        workflow = OnboardingWorkflow(
            customer_id=customer_profile.customer_id,
            config=config,
            status=OnboardingStatus.IN_PROGRESS,
            started_at=datetime.utcnow(),
        )

        # Generate workflow steps
        workflow.steps = await self._generate_workflow_steps(customer_profile, config)

        # Calculate expected completion
        total_duration = sum(step.estimated_duration_minutes for step in workflow.steps)
        workflow.expected_completion = datetime.utcnow() + timedelta(
            minutes=total_duration
        )

        # Store workflow and profile
        self.workflows[workflow.workflow_id] = workflow
        self.customer_profiles[customer_profile.customer_id] = customer_profile

        # Send welcome email
        await self.email_service.send_welcome_email(
            customer_profile.primary_contact_email,
            customer_profile.primary_contact_name,
            f"https://app.metis.ai/onboarding/{workflow.workflow_id}",
        )

        # Start first step
        await self._start_next_step(workflow.workflow_id)

        self.logger.info(
            f"Onboarding initiated for {customer_profile.company_name} - Workflow {workflow.workflow_id}"
        )

        return workflow

    def _determine_onboarding_config(
        self, profile: CustomerProfile
    ) -> OnboardingConfiguration:
        """Determine onboarding configuration based on customer profile"""

        config = OnboardingConfiguration()

        # Determine subscription tier
        if profile.company_size in ["enterprise", "large"]:
            config.subscription_tier = SubscriptionTier.ENTERPRISE
            config.onboarding_type = OnboardingType.WHITE_GLOVE
            config.dedicated_success_manager = True
            config.max_users = 100
            config.max_engagements_per_month = 1000
            config.storage_limit_gb = 500
        elif profile.company_size == "medium":
            config.subscription_tier = SubscriptionTier.PROFESSIONAL
            config.onboarding_type = OnboardingType.GUIDED
            config.max_users = 50
            config.max_engagements_per_month = 500
            config.storage_limit_gb = 200
        else:
            config.subscription_tier = SubscriptionTier.STARTER
            config.onboarding_type = OnboardingType.SELF_SERVICE
            config.max_users = 10
            config.max_engagements_per_month = 100
            config.storage_limit_gb = 50

        # Configure features based on industry
        if profile.industry in ["consulting", "finance", "technology"]:
            config.enable_custom_frameworks = True
            config.advanced_analytics = True

        # Configure integrations
        if "sso" in profile.integration_requirements:
            config.enable_sso = True

        return config

    async def _generate_workflow_steps(
        self, profile: CustomerProfile, config: OnboardingConfiguration
    ) -> List[OnboardingStep]:
        """Generate onboarding steps based on configuration"""

        steps = []

        # Registration step
        steps.append(
            OnboardingStep(
                step_id="registration",
                stage=OnboardingStage.REGISTRATION,
                title="Account Registration",
                description="Complete account registration and email verification",
                estimated_duration_minutes=10,
                automated=True,
                automation_function="complete_registration",
            )
        )

        # Tenant provisioning
        steps.append(
            OnboardingStep(
                step_id="tenant_provisioning",
                stage=OnboardingStage.TENANT_PROVISIONING,
                title="Tenant Provisioning",
                description="Provision dedicated tenant environment",
                estimated_duration_minutes=5,
                automated=True,
                automation_function="provision_tenant",
                prerequisites=["registration"],
            )
        )

        # Account setup
        steps.append(
            OnboardingStep(
                step_id="account_setup",
                stage=OnboardingStage.ACCOUNT_SETUP,
                title="Account Configuration",
                description="Configure account settings and preferences",
                estimated_duration_minutes=15,
                prerequisites=["tenant_provisioning"],
                instructions="Set up your company profile, preferences, and initial configurations",
            )
        )

        # User configuration
        steps.append(
            OnboardingStep(
                step_id="user_setup",
                stage=OnboardingStage.USER_CONFIGURATION,
                title="User Setup",
                description="Create admin user and invite team members",
                estimated_duration_minutes=20,
                prerequisites=["account_setup"],
                instructions="Create your admin account and invite your team members",
            )
        )

        # SSO setup (if enabled)
        if config.enable_sso:
            steps.append(
                OnboardingStep(
                    step_id="sso_setup",
                    stage=OnboardingStage.ACCOUNT_SETUP,
                    title="SSO Configuration",
                    description="Configure Single Sign-On integration",
                    estimated_duration_minutes=30,
                    prerequisites=["user_setup"],
                    instructions="Configure your SSO provider for seamless authentication",
                )
            )

        # Training modules
        if not config.skip_training:
            training_steps = self._generate_training_steps(config)
            steps.extend(training_steps)

        # First engagement
        steps.append(
            OnboardingStep(
                step_id="first_engagement",
                stage=OnboardingStage.FIRST_ENGAGEMENT,
                title="First Cognitive Engagement",
                description="Create and run your first cognitive analysis",
                estimated_duration_minutes=45,
                prerequisites=(
                    ["training_complete"]
                    if not config.skip_training
                    else ["user_setup"]
                ),
                instructions="Use METIS to run your first cognitive engagement and generate insights",
            )
        )

        # Success metrics
        steps.append(
            OnboardingStep(
                step_id="success_metrics",
                stage=OnboardingStage.SUCCESS_METRICS,
                title="Success Metrics Setup",
                description="Configure success tracking and KPIs",
                estimated_duration_minutes=15,
                prerequisites=["first_engagement"],
                instructions="Set up metrics to track your success with METIS",
            )
        )

        return steps

    def _generate_training_steps(
        self, config: OnboardingConfiguration
    ) -> List[OnboardingStep]:
        """Generate training module steps"""

        steps = []

        # Core training modules
        core_modules = [
            (
                "metis_overview",
                "METIS Platform Overview",
                "Learn about METIS capabilities and architecture",
            ),
            (
                "mental_models",
                "Mental Models Framework",
                "Understanding and applying cognitive frameworks",
            ),
            (
                "hypothesis_engine",
                "Hypothesis Generation",
                "Learn to generate and test hypotheses",
            ),
            (
                "pyramid_principle",
                "Pyramid Principle",
                "Master executive communication structure",
            ),
            (
                "transparency_ux",
                "Progressive Transparency",
                "Navigate and customize the transparency interface",
            ),
        ]

        for i, (module_id, title, description) in enumerate(core_modules):
            steps.append(
                OnboardingStep(
                    step_id=f"training_{module_id}",
                    stage=OnboardingStage.TRAINING,
                    title=f"Training: {title}",
                    description=description,
                    estimated_duration_minutes=20,
                    prerequisites=(
                        ["user_setup"]
                        if i == 0
                        else [f"training_{core_modules[i-1][0]}"]
                    ),
                    instructions=f"Complete the {title} interactive training module",
                )
            )

        # Custom training modules
        for module in config.custom_training_modules:
            steps.append(
                OnboardingStep(
                    step_id=f"training_custom_{module}",
                    stage=OnboardingStage.TRAINING,
                    title=f"Custom Training: {module}",
                    description=f"Industry-specific training for {module}",
                    estimated_duration_minutes=25,
                    prerequisites=[f"training_{core_modules[-1][0]}"],
                    instructions=f"Complete the custom {module} training module",
                )
            )

        # Training completion marker
        steps.append(
            OnboardingStep(
                step_id="training_complete",
                stage=OnboardingStage.TRAINING,
                title="Training Complete",
                description="All training modules completed",
                estimated_duration_minutes=5,
                automated=True,
                automation_function="mark_training_complete",
                prerequisites=[f"training_{core_modules[-1][0]}"],
            )
        )

        return steps

    async def complete_step(
        self,
        workflow_id: str,
        step_id: str,
        completion_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark step as completed and advance workflow"""

        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]

        # Find step
        step = None
        for s in workflow.steps:
            if s.step_id == step_id:
                step = s
                break

        if not step:
            return False

        # Mark step as completed
        step.status = OnboardingStatus.COMPLETED
        step.completed_at = datetime.utcnow()
        step.progress_percentage = 100

        if completion_data:
            step.completion_notes = json.dumps(completion_data)

        # Update workflow progress
        completed_steps = [
            s for s in workflow.steps if s.status == OnboardingStatus.COMPLETED
        ]
        workflow.progress_percentage = len(completed_steps) / len(workflow.steps) * 100

        # Send notification
        customer = self.customer_profiles[workflow.customer_id]

        # Find next step
        next_step = self._find_next_step(workflow)
        if next_step:
            await self.email_service.send_step_completion_notification(
                customer.primary_contact_email,
                customer.primary_contact_name,
                step.title,
                next_step.title,
            )

            # Start next step
            await self._start_next_step(workflow_id)
        else:
            # Workflow completed
            await self._complete_workflow(workflow_id)

        self.logger.info(f"Step {step_id} completed for workflow {workflow_id}")

        return True

    async def _start_next_step(self, workflow_id: str):
        """Start the next available step"""

        workflow = self.workflows[workflow_id]
        next_step = self._find_next_step(workflow)

        if not next_step:
            return

        next_step.status = OnboardingStatus.IN_PROGRESS
        next_step.started_at = datetime.utcnow()

        # Update current stage
        workflow.current_stage = next_step.stage

        # Execute automated steps
        if next_step.automated and next_step.automation_function:
            await self._execute_automated_step(workflow_id, next_step)

        self.logger.info(f"Started step {next_step.step_id} for workflow {workflow_id}")

    def _find_next_step(self, workflow: OnboardingWorkflow) -> Optional[OnboardingStep]:
        """Find next step that can be started"""

        for step in workflow.steps:
            if step.status != OnboardingStatus.PENDING:
                continue

            # Check prerequisites
            prerequisites_met = True
            for prereq_id in step.prerequisites:
                prereq_step = None
                for s in workflow.steps:
                    if s.step_id == prereq_id:
                        prereq_step = s
                        break

                if not prereq_step or prereq_step.status != OnboardingStatus.COMPLETED:
                    prerequisites_met = False
                    break

            if prerequisites_met:
                return step

        return None

    async def _execute_automated_step(self, workflow_id: str, step: OnboardingStep):
        """Execute automated step function"""

        workflow = self.workflows[workflow_id]
        customer = self.customer_profiles[workflow.customer_id]

        try:
            if step.automation_function == "complete_registration":
                # Registration automation
                await asyncio.sleep(1)  # Simulate processing

            elif step.automation_function == "provision_tenant":
                # Tenant provisioning
                result = await self.provisioning_service.provision_tenant(
                    customer, workflow.config
                )

                if result["status"] == "provisioned":
                    workflow.tenant_id = result["tenant_id"]
                else:
                    raise Exception(
                        f"Tenant provisioning failed: {result.get('error')}"
                    )

            elif step.automation_function == "mark_training_complete":
                # Mark training as complete
                workflow.training_modules_completed = len(
                    [
                        s
                        for s in workflow.steps
                        if s.stage == OnboardingStage.TRAINING
                        and s.status == OnboardingStatus.COMPLETED
                    ]
                )

            # Mark step as completed
            await self.complete_step(workflow_id, step.step_id)

        except Exception as e:
            step.status = OnboardingStatus.FAILED
            step.completion_notes = f"Automation failed: {str(e)}"
            self.logger.error(f"Automated step {step.step_id} failed: {str(e)}")

    async def _complete_workflow(self, workflow_id: str):
        """Complete onboarding workflow"""

        workflow = self.workflows[workflow_id]
        customer = self.customer_profiles[workflow.customer_id]

        workflow.status = OnboardingStatus.COMPLETED
        workflow.completed_at = datetime.utcnow()
        workflow.progress_percentage = 100

        # Calculate time to value
        if workflow.started_at:
            workflow.time_to_value_days = (datetime.utcnow() - workflow.started_at).days

        # Send completion email
        await self.email_service.send_completion_email(
            customer.primary_contact_email,
            customer.primary_contact_name,
            f"https://app.metis.ai/dashboard?tenant_id={workflow.tenant_id}",
        )

        # Schedule follow-up
        await self._schedule_success_follow_up(workflow_id)

        self.logger.info(f"Onboarding workflow {workflow_id} completed successfully")

    async def _schedule_success_follow_up(self, workflow_id: str):
        """Schedule follow-up for success tracking"""
        # Would schedule check-ins at 7, 30, 90 days
        pass

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of onboarding workflow"""

        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]
        customer = self.customer_profiles[workflow.customer_id]

        # Get current step
        current_step = None
        for step in workflow.steps:
            if step.status == OnboardingStatus.IN_PROGRESS:
                current_step = step
                break

        # Calculate estimated time remaining
        remaining_steps = [
            s for s in workflow.steps if s.status == OnboardingStatus.PENDING
        ]
        estimated_time_remaining = sum(
            s.estimated_duration_minutes for s in remaining_steps
        )

        return {
            "workflow_id": workflow_id,
            "customer": {
                "company_name": customer.company_name,
                "primary_contact": customer.primary_contact_name,
            },
            "status": workflow.status.value,
            "current_stage": workflow.current_stage.value,
            "progress_percentage": workflow.progress_percentage,
            "current_step": (
                {
                    "step_id": current_step.step_id,
                    "title": current_step.title,
                    "description": current_step.description,
                }
                if current_step
                else None
            ),
            "timeline": {
                "started_at": (
                    workflow.started_at.isoformat() if workflow.started_at else None
                ),
                "expected_completion": (
                    workflow.expected_completion.isoformat()
                    if workflow.expected_completion
                    else None
                ),
                "estimated_time_remaining_minutes": estimated_time_remaining,
            },
            "resources": {
                "tenant_id": workflow.tenant_id,
                "admin_user_id": workflow.admin_user_id,
            },
            "metrics": {
                "engagement_count": workflow.engagement_count,
                "training_modules_completed": workflow.training_modules_completed,
                "time_to_value_days": workflow.time_to_value_days,
            },
        }

    def get_onboarding_analytics(self) -> Dict[str, Any]:
        """Get analytics on onboarding performance"""

        total_workflows = len(self.workflows)
        if total_workflows == 0:
            return {"message": "No onboarding data available"}

        completed_workflows = [
            w for w in self.workflows.values() if w.status == OnboardingStatus.COMPLETED
        ]
        in_progress_workflows = [
            w
            for w in self.workflows.values()
            if w.status == OnboardingStatus.IN_PROGRESS
        ]
        failed_workflows = [
            w for w in self.workflows.values() if w.status == OnboardingStatus.FAILED
        ]

        # Calculate completion rate
        completion_rate = len(completed_workflows) / total_workflows * 100

        # Calculate average time to completion
        completion_times = [
            (w.completed_at - w.started_at).days
            for w in completed_workflows
            if w.completed_at and w.started_at
        ]
        avg_completion_time = (
            sum(completion_times) / len(completion_times) if completion_times else 0
        )

        # Onboarding type distribution
        type_distribution = {}
        for workflow in self.workflows.values():
            onboarding_type = workflow.config.onboarding_type.value
            type_distribution[onboarding_type] = (
                type_distribution.get(onboarding_type, 0) + 1
            )

        # Common failure points
        failure_points = {}
        for workflow in self.workflows.values():
            for step in workflow.steps:
                if step.status == OnboardingStatus.FAILED:
                    failure_points[step.step_id] = (
                        failure_points.get(step.step_id, 0) + 1
                    )

        return {
            "summary": {
                "total_workflows": total_workflows,
                "completed": len(completed_workflows),
                "in_progress": len(in_progress_workflows),
                "failed": len(failed_workflows),
                "completion_rate": completion_rate,
            },
            "performance": {
                "avg_completion_time_days": avg_completion_time,
                "avg_time_to_value_days": (
                    sum(
                        w.time_to_value_days
                        for w in completed_workflows
                        if w.time_to_value_days
                    )
                    / len(completed_workflows)
                    if completed_workflows
                    else 0
                ),
            },
            "distribution": {
                "onboarding_types": type_distribution,
                "subscription_tiers": {
                    tier.value: len(
                        [
                            w
                            for w in self.workflows.values()
                            if w.config.subscription_tier == tier
                        ]
                    )
                    for tier in SubscriptionTier
                },
            },
            "failure_analysis": {
                "common_failure_points": failure_points,
                "failure_rate": len(failed_workflows) / total_workflows * 100,
            },
            "generated_at": datetime.utcnow().isoformat(),
        }


# Global onboarding engine instance
_global_onboarding_engine: Optional[OnboardingWorkflowEngine] = None


def get_onboarding_engine() -> OnboardingWorkflowEngine:
    """Get global onboarding engine instance"""
    global _global_onboarding_engine

    if _global_onboarding_engine is None:
        _global_onboarding_engine = OnboardingWorkflowEngine()

    return _global_onboarding_engine


# Convenience functions for common onboarding operations
async def start_customer_onboarding(
    company_name: str,
    contact_email: str,
    contact_name: str,
    company_size: str = "medium",
    industry: str = "",
    onboarding_type: OnboardingType = OnboardingType.SELF_SERVICE,
) -> str:
    """Convenience function to start customer onboarding"""

    # Create customer profile
    profile = CustomerProfile(
        company_name=company_name,
        company_size=company_size,
        industry=industry,
        primary_contact_name=contact_name,
        primary_contact_email=contact_email,
        preferred_onboarding_type=onboarding_type,
    )

    # Start onboarding
    engine = get_onboarding_engine()
    workflow = await engine.initiate_onboarding(profile)

    return workflow.workflow_id


async def complete_onboarding_step(
    workflow_id: str, step_id: str, completion_data: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience function to complete onboarding step"""
    engine = get_onboarding_engine()
    return await engine.complete_step(workflow_id, step_id, completion_data)


def get_onboarding_status(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get onboarding status"""
    engine = get_onboarding_engine()
    return engine.get_workflow_status(workflow_id)
