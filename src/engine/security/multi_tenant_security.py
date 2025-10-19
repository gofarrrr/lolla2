"""
METIS Multi-Tenant Security System
E002: Enterprise-grade multi-tenant security with data isolation

Implements comprehensive security framework with tenant isolation,
role-based access control, data encryption, and compliance features.
"""

import asyncio
import logging
import jwt

# Bcrypt import with fallback
try:
    import bcrypt
except ImportError:
    print("Warning: bcrypt not available, using fallback authentication")
    bcrypt = None

import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import base64

# Cryptography import with fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    print("Warning: cryptography not available, using fallback encryption")
    CRYPTOGRAPHY_AVAILABLE = False

    # Mock cryptography classes
    class MockFernet:
        def __init__(self, key):
            self.key = key

        def encrypt(self, data):
            return b"mock_encrypted_" + data

        def decrypt(self, data):
            if data.startswith(b"mock_encrypted_"):
                return data[15:]
            return data

        @staticmethod
        def generate_key():
            return b"mock_key_32_bytes_for_testing_only"

    Fernet = MockFernet
    hashes = None
    serialization = None
    PBKDF2HMAC = None

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class TenantStatus(str, Enum):
    """Tenant account status"""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    PENDING = "pending"
    TERMINATED = "terminated"


class UserRole(str, Enum):
    """User roles within tenant"""

    OWNER = "owner"  # Full tenant administration
    ADMIN = "admin"  # Tenant administration
    MANAGER = "manager"  # Team management and analysis
    ANALYST = "analyst"  # Analysis and reporting
    VIEWER = "viewer"  # Read-only access
    GUEST = "guest"  # Limited temporary access


class Permission(str, Enum):
    """Granular permissions system"""

    # Engagement permissions
    ENGAGEMENT_CREATE = "engagement:create"
    ENGAGEMENT_READ = "engagement:read"
    ENGAGEMENT_UPDATE = "engagement:update"
    ENGAGEMENT_DELETE = "engagement:delete"

    # Analysis permissions
    ANALYSIS_EXECUTE = "analysis:execute"
    ANALYSIS_EXPORT = "analysis:export"
    ANALYSIS_SHARE = "analysis:share"

    # Admin permissions
    TENANT_ADMIN = "tenant:admin"
    USER_MANAGE = "user:manage"
    BILLING_MANAGE = "billing:manage"
    SECURITY_MANAGE = "security:manage"

    # System permissions
    AUDIT_VIEW = "audit:view"
    METRICS_VIEW = "metrics:view"
    INTEGRATION_MANAGE = "integration:manage"


class SecurityLevel(str, Enum):
    """Security classification levels"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class TenantConfiguration:
    """Tenant-specific configuration and limits"""

    tenant_id: str = ""

    # Resource limits
    max_users: int = 100
    max_engagements_per_month: int = 1000
    max_storage_gb: int = 100
    max_api_calls_per_minute: int = 1000

    # Security settings
    password_policy: Dict[str, Any] = field(default_factory=dict)
    session_timeout_minutes: int = 480  # 8 hours
    mfa_required: bool = False
    ip_whitelist: List[str] = field(default_factory=list)

    # Data retention
    data_retention_days: int = 2555  # 7 years
    audit_log_retention_days: int = 2555

    # Compliance
    compliance_frameworks: List[str] = field(default_factory=list)  # SOC2, GDPR, etc.
    data_residency: str = "US"
    encryption_required: bool = True

    # Feature flags
    advanced_analytics: bool = True
    api_access: bool = True
    custom_frameworks: bool = False
    white_label: bool = False


@dataclass
class Tenant:
    """Multi-tenant organization entity"""

    tenant_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    domain: str = ""
    status: TenantStatus = TenantStatus.PENDING

    # Configuration
    config: TenantConfiguration = field(default_factory=TenantConfiguration)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

    # Billing and subscription
    subscription_tier: str = "enterprise"
    billing_contact: str = ""

    # Security
    encryption_key: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.config.tenant_id:
            self.config.tenant_id = self.tenant_id


@dataclass
class User:
    """User entity with tenant association"""

    user_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    email: str = ""
    name: str = ""
    role: UserRole = UserRole.ANALYST

    # Authentication
    password_hash: str = ""
    salt: str = ""
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False

    # Authorization
    permissions: Set[Permission] = field(default_factory=set)
    custom_permissions: Set[str] = field(default_factory=set)

    # Status
    active: bool = True
    email_verified: bool = False
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_password_change: datetime = field(default_factory=datetime.utcnow)

    # Security
    session_tokens: Set[str] = field(default_factory=set)
    api_tokens: Set[str] = field(default_factory=set)


@dataclass
class SecurityContext:
    """Security context for request processing"""

    tenant_id: str = ""
    user_id: str = ""
    role: UserRole = UserRole.VIEWER
    permissions: Set[Permission] = field(default_factory=set)

    # Request metadata
    ip_address: str = ""
    user_agent: str = ""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Security flags
    is_authenticated: bool = False
    is_mfa_verified: bool = False
    is_api_request: bool = False
    security_level: SecurityLevel = SecurityLevel.INTERNAL

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions

    def can_access_engagement(self, engagement_tenant_id: str) -> bool:
        """Check if can access engagement from another tenant"""
        return self.tenant_id == engagement_tenant_id


class EncryptionManager:
    """Handles tenant-specific data encryption"""

    def __init__(self):
        self.master_key = self._generate_master_key()
        self.tenant_keys: Dict[str, bytes] = {}
        self.logger = logging.getLogger(__name__)

    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        # In production, this would be stored in secure key management
        return base64.urlsafe_b64decode(
            "YourMasterKeyHere-ThisShouldBeFromKeyManagement="
        )

    def get_tenant_key(self, tenant_id: str) -> bytes:
        """Get or create tenant-specific encryption key"""
        if tenant_id not in self.tenant_keys:
            # Derive tenant key from master key and tenant ID
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=tenant_id.encode("utf-8"),
                iterations=100000,
            )
            self.tenant_keys[tenant_id] = kdf.derive(self.master_key)

        return self.tenant_keys[tenant_id]

    def encrypt_data(self, data: str, tenant_id: str) -> str:
        """Encrypt data with tenant-specific key"""
        try:
            tenant_key = self.get_tenant_key(tenant_id)
            fernet = Fernet(base64.urlsafe_b64encode(tenant_key))
            encrypted_data = fernet.encrypt(data.encode("utf-8"))
            return base64.urlsafe_b64encode(encrypted_data).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Encryption failed for tenant {tenant_id}: {str(e)}")
            raise

    def decrypt_data(self, encrypted_data: str, tenant_id: str) -> str:
        """Decrypt data with tenant-specific key"""
        try:
            tenant_key = self.get_tenant_key(tenant_id)
            fernet = Fernet(base64.urlsafe_b64encode(tenant_key))
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode("utf-8"))
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode("utf-8")
        except Exception as e:
            self.logger.error(f"Decryption failed for tenant {tenant_id}: {str(e)}")
            raise


class AuthenticationManager:
    """Handles user authentication and session management"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.jwt_secret = self._generate_jwt_secret()
        self.session_timeout = timedelta(hours=8)
        self.logger = logging.getLogger(__name__)

    def _generate_jwt_secret(self) -> str:
        """Generate JWT signing secret"""
        # In production, use secure key management
        return "your-jwt-secret-key-here"

    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash password with salt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode("utf-8"), salt)
        return password_hash.decode("utf-8"), salt.decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    def generate_session_token(
        self, user: User, tenant: Tenant, expires_in: Optional[timedelta] = None
    ) -> str:
        """Generate JWT session token"""

        if expires_in is None:
            expires_in = self.session_timeout

        payload = {
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "role": user.role.value,
            "permissions": list(user.permissions),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + expires_in,
            "jti": str(uuid4()),  # JWT ID for token revocation
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")

        # Store session in Redis for revocation capability
        if self.redis_client:
            session_key = f"session:{user.tenant_id}:{user.user_id}:{payload['jti']}"
            asyncio.create_task(
                self.redis_client.setex(
                    session_key,
                    int(expires_in.total_seconds()),
                    json.dumps(
                        {
                            "user_id": user.user_id,
                            "tenant_id": user.tenant_id,
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    ),
                )
            )

        return token

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Check if session is revoked
            if self.redis_client:
                session_key = f"session:{payload['tenant_id']}:{payload['user_id']}:{payload['jti']}"
                # Would check Redis in async context

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.info("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {str(e)}")
            return None

    async def revoke_session(self, token: str) -> bool:
        """Revoke session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            if self.redis_client:
                session_key = f"session:{payload['tenant_id']}:{payload['user_id']}:{payload['jti']}"
                await self.redis_client.delete(session_key)

            return True

        except Exception as e:
            self.logger.error(f"Failed to revoke session: {str(e)}")
            return False

    def generate_api_token(self, user: User, name: str = "") -> str:
        """Generate long-lived API token"""
        payload = {
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "token_type": "api",
            "token_name": name,
            "iat": datetime.utcnow(),
            # API tokens don't expire by default
            "jti": str(uuid4()),
        }

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")


class AuthorizationManager:
    """Handles role-based access control and permissions"""

    def __init__(self):
        self.role_permissions = self._define_role_permissions()
        self.logger = logging.getLogger(__name__)

    def _define_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Define default permissions for each role"""
        return {
            UserRole.OWNER: {
                # All permissions
                Permission.ENGAGEMENT_CREATE,
                Permission.ENGAGEMENT_READ,
                Permission.ENGAGEMENT_UPDATE,
                Permission.ENGAGEMENT_DELETE,
                Permission.ANALYSIS_EXECUTE,
                Permission.ANALYSIS_EXPORT,
                Permission.ANALYSIS_SHARE,
                Permission.TENANT_ADMIN,
                Permission.USER_MANAGE,
                Permission.BILLING_MANAGE,
                Permission.SECURITY_MANAGE,
                Permission.AUDIT_VIEW,
                Permission.METRICS_VIEW,
                Permission.INTEGRATION_MANAGE,
            },
            UserRole.ADMIN: {
                Permission.ENGAGEMENT_CREATE,
                Permission.ENGAGEMENT_READ,
                Permission.ENGAGEMENT_UPDATE,
                Permission.ENGAGEMENT_DELETE,
                Permission.ANALYSIS_EXECUTE,
                Permission.ANALYSIS_EXPORT,
                Permission.ANALYSIS_SHARE,
                Permission.USER_MANAGE,
                Permission.AUDIT_VIEW,
                Permission.METRICS_VIEW,
                Permission.INTEGRATION_MANAGE,
            },
            UserRole.MANAGER: {
                Permission.ENGAGEMENT_CREATE,
                Permission.ENGAGEMENT_READ,
                Permission.ENGAGEMENT_UPDATE,
                Permission.ANALYSIS_EXECUTE,
                Permission.ANALYSIS_EXPORT,
                Permission.ANALYSIS_SHARE,
                Permission.METRICS_VIEW,
            },
            UserRole.ANALYST: {
                Permission.ENGAGEMENT_CREATE,
                Permission.ENGAGEMENT_READ,
                Permission.ANALYSIS_EXECUTE,
                Permission.ANALYSIS_EXPORT,
            },
            UserRole.VIEWER: {Permission.ENGAGEMENT_READ},
            UserRole.GUEST: {
                Permission.ENGAGEMENT_READ  # Limited scope would be enforced elsewhere
            },
        }

    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for user (role + custom)"""
        role_permissions = self.role_permissions.get(user.role, set())
        return role_permissions.union(user.permissions)

    def check_permission(
        self,
        user: User,
        required_permission: Permission,
        resource_tenant_id: Optional[str] = None,
    ) -> bool:
        """Check if user has required permission"""

        # Check if user is active
        if not user.active:
            return False

        # Check tenant isolation
        if resource_tenant_id and resource_tenant_id != user.tenant_id:
            return False

        # Check permission
        user_permissions = self.get_user_permissions(user)
        return required_permission in user_permissions

    def filter_by_permissions(
        self,
        user: User,
        resources: List[Dict[str, Any]],
        required_permission: Permission,
        tenant_id_field: str = "tenant_id",
    ) -> List[Dict[str, Any]]:
        """Filter resources based on user permissions"""
        filtered = []

        for resource in resources:
            resource_tenant_id = resource.get(tenant_id_field)
            if self.check_permission(user, required_permission, resource_tenant_id):
                filtered.append(resource)

        return filtered


class TenantIsolationManager:
    """Ensures data isolation between tenants"""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger(__name__)

    def create_tenant_context(self, tenant_id: str) -> Dict[str, Any]:
        """Create isolated context for tenant operations"""
        return {
            "tenant_id": tenant_id,
            "database_schema": f"tenant_{hashlib.md5(tenant_id.encode()).hexdigest()[:8]}",
            "cache_prefix": f"tenant:{tenant_id}:",
            "storage_path": f"/data/tenants/{tenant_id}/",
            "encryption_key": self.encryption_manager.get_tenant_key(tenant_id),
        }

    def validate_tenant_access(
        self, requesting_tenant_id: str, resource_tenant_id: str
    ) -> bool:
        """Validate tenant can access resource"""
        return requesting_tenant_id == resource_tenant_id

    def apply_tenant_filter(
        self, query_params: Dict[str, Any], tenant_id: str
    ) -> Dict[str, Any]:
        """Apply tenant filter to database query"""
        query_params = query_params.copy()
        query_params["tenant_id"] = tenant_id
        return query_params

    def encrypt_tenant_data(
        self, data: Dict[str, Any], tenant_id: str, sensitive_fields: List[str]
    ) -> Dict[str, Any]:
        """Encrypt sensitive fields in tenant data"""
        encrypted_data = data.copy()

        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encryption_manager.encrypt_data(
                    str(encrypted_data[field]), tenant_id
                )

        return encrypted_data

    def decrypt_tenant_data(
        self,
        encrypted_data: Dict[str, Any],
        tenant_id: str,
        sensitive_fields: List[str],
    ) -> Dict[str, Any]:
        """Decrypt sensitive fields in tenant data"""
        decrypted_data = encrypted_data.copy()

        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = self.encryption_manager.decrypt_data(
                        decrypted_data[field], tenant_id
                    )
                except Exception as e:
                    self.logger.error(f"Failed to decrypt field {field}: {str(e)}")
                    decrypted_data[field] = "[DECRYPTION_ERROR]"

        return decrypted_data


class SecurityAuditLogger:
    """Logs security events for compliance and monitoring"""

    def __init__(self):
        self.logger = logging.getLogger("security_audit")
        # Configure special logger for security events

    def log_authentication_event(
        self,
        event_type: str,
        user_id: str,
        tenant_id: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        additional_data: Dict[str, Any] = None,
    ):
        """Log authentication events"""
        event = {
            "event_type": "authentication",
            "sub_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "tenant_id": tenant_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success,
            "additional_data": additional_data or {},
        }

        self.logger.info(json.dumps(event))

    def log_authorization_event(
        self,
        user_id: str,
        tenant_id: str,
        resource: str,
        action: str,
        allowed: bool,
        ip_address: str = "",
    ):
        """Log authorization events"""
        event = {
            "event_type": "authorization",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "tenant_id": tenant_id,
            "resource": resource,
            "action": action,
            "allowed": allowed,
            "ip_address": ip_address,
        }

        self.logger.info(json.dumps(event))

    def log_data_access_event(
        self,
        user_id: str,
        tenant_id: str,
        data_type: str,
        data_id: str,
        operation: str,
        ip_address: str = "",
    ):
        """Log data access events"""
        event = {
            "event_type": "data_access",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "tenant_id": tenant_id,
            "data_type": data_type,
            "data_id": data_id,
            "operation": operation,
            "ip_address": ip_address,
        }

        self.logger.info(json.dumps(event))

    def log_security_event(
        self,
        event_type: str,
        tenant_id: str,
        severity: str,
        description: str,
        user_id: str = "",
        ip_address: str = "",
        additional_data: Dict[str, Any] = None,
    ):
        """Log general security events"""
        event = {
            "event_type": "security",
            "sub_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "severity": severity,
            "description": description,
            "ip_address": ip_address,
            "additional_data": additional_data or {},
        }

        self.logger.warning(json.dumps(event))


class MultiTenantSecurityManager:
    """Main security manager coordinating all security components"""

    def __init__(self, redis_url: Optional[str] = None):
        # Initialize Redis client for session management
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            self.redis_client = redis.from_url(redis_url)

        # Initialize security components
        self.encryption_manager = EncryptionManager()
        self.auth_manager = AuthenticationManager(self.redis_client)
        self.authz_manager = AuthorizationManager()
        self.isolation_manager = TenantIsolationManager(self.encryption_manager)
        self.audit_logger = SecurityAuditLogger()

        # In-memory stores (would use database in production)
        self.tenants: Dict[str, Tenant] = {}
        self.users: Dict[str, User] = {}
        self.user_by_email: Dict[str, str] = {}  # email -> user_id

        self.logger = logging.getLogger(__name__)

    async def create_tenant(
        self,
        name: str,
        domain: str,
        admin_email: str,
        admin_name: str,
        admin_password: str,
        config: Optional[TenantConfiguration] = None,
    ) -> Tuple[Tenant, User]:
        """Create new tenant with admin user"""

        # Create tenant
        tenant = Tenant(
            name=name,
            domain=domain,
            status=TenantStatus.ACTIVE,
            config=config or TenantConfiguration(),
        )

        # Create admin user
        password_hash, salt = self.auth_manager.hash_password(admin_password)

        admin_user = User(
            tenant_id=tenant.tenant_id,
            email=admin_email,
            name=admin_name,
            role=UserRole.OWNER,
            password_hash=password_hash,
            salt=salt,
            active=True,
            email_verified=True,
        )

        # Set permissions
        admin_user.permissions = self.authz_manager.get_user_permissions(admin_user)

        # Store tenant and user
        self.tenants[tenant.tenant_id] = tenant
        self.users[admin_user.user_id] = admin_user
        self.user_by_email[admin_email] = admin_user.user_id

        # Log tenant creation
        self.audit_logger.log_security_event(
            event_type="tenant_created",
            tenant_id=tenant.tenant_id,
            severity="info",
            description=f"Tenant '{name}' created with admin user '{admin_email}'",
            additional_data={"domain": domain},
        )

        self.logger.info(
            f"Created tenant {tenant.tenant_id} with admin user {admin_user.user_id}"
        )

        return tenant, admin_user

    async def authenticate_user(
        self, email: str, password: str, ip_address: str = "", user_agent: str = ""
    ) -> Optional[Tuple[User, str]]:
        """Authenticate user and return user + session token"""

        # Find user by email
        user_id = self.user_by_email.get(email)
        if not user_id:
            self.audit_logger.log_authentication_event(
                event_type="login_failed",
                user_id="",
                tenant_id="",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                additional_data={"reason": "user_not_found", "email": email},
            )
            return None

        user = self.users[user_id]

        # Check if user is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            self.audit_logger.log_authentication_event(
                event_type="login_failed",
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                additional_data={"reason": "account_locked"},
            )
            return None

        # Verify password
        if not self.auth_manager.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1

            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)

            self.audit_logger.log_authentication_event(
                event_type="login_failed",
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                additional_data={
                    "reason": "invalid_password",
                    "attempts": user.failed_login_attempts,
                },
            )
            return None

        # Check tenant status
        tenant = self.tenants.get(user.tenant_id)
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            self.audit_logger.log_authentication_event(
                event_type="login_failed",
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                additional_data={"reason": "tenant_inactive"},
            )
            return None

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()

        # Generate session token
        session_token = self.auth_manager.generate_session_token(user, tenant)

        # Log successful authentication
        self.audit_logger.log_authentication_event(
            event_type="login_success",
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
        )

        return user, session_token

    def create_security_context(
        self, session_token: str, ip_address: str = "", user_agent: str = ""
    ) -> Optional[SecurityContext]:
        """Create security context from session token"""

        # Verify token
        payload = self.auth_manager.verify_session_token(session_token)
        if not payload:
            return None

        # Get user
        user = self.users.get(payload["user_id"])
        if not user or not user.active:
            return None

        # Get tenant
        tenant = self.tenants.get(payload["tenant_id"])
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            return None

        # Create context
        context = SecurityContext(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            role=user.role,
            permissions=self.authz_manager.get_user_permissions(user),
            ip_address=ip_address,
            user_agent=user_agent,
            is_authenticated=True,
            is_mfa_verified=not user.mfa_enabled or payload.get("mfa_verified", False),
        )

        return context

    def require_permission(
        self,
        context: SecurityContext,
        permission: Permission,
        resource_tenant_id: Optional[str] = None,
    ) -> bool:
        """Check if context has required permission"""

        if not context.is_authenticated:
            return False

        # Check tenant isolation
        if resource_tenant_id and resource_tenant_id != context.tenant_id:
            self.audit_logger.log_authorization_event(
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                resource=f"tenant:{resource_tenant_id}",
                action=permission.value,
                allowed=False,
                ip_address=context.ip_address,
            )
            return False

        # Check permission
        has_permission = context.has_permission(permission)

        # Log authorization event
        self.audit_logger.log_authorization_event(
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            resource=resource_tenant_id or "global",
            action=permission.value,
            allowed=has_permission,
            ip_address=context.ip_address,
        )

        return has_permission

    def create_tenant_isolation_context(self, tenant_id: str) -> Dict[str, Any]:
        """Create tenant isolation context"""
        return self.isolation_manager.create_tenant_context(tenant_id)

    def encrypt_sensitive_data(
        self, data: Dict[str, Any], tenant_id: str, sensitive_fields: List[str] = None
    ) -> Dict[str, Any]:
        """Encrypt sensitive data for tenant"""
        if sensitive_fields is None:
            sensitive_fields = [
                "email",
                "phone",
                "ssn",
                "credit_card",
                "api_key",
                "problem_statement",
                "business_context",
                "analysis_results",
            ]

        return self.isolation_manager.encrypt_tenant_data(
            data, tenant_id, sensitive_fields
        )

    def decrypt_sensitive_data(
        self,
        encrypted_data: Dict[str, Any],
        tenant_id: str,
        sensitive_fields: List[str] = None,
    ) -> Dict[str, Any]:
        """Decrypt sensitive data for tenant"""
        if sensitive_fields is None:
            sensitive_fields = [
                "email",
                "phone",
                "ssn",
                "credit_card",
                "api_key",
                "problem_statement",
                "business_context",
                "analysis_results",
            ]

        return self.isolation_manager.decrypt_tenant_data(
            encrypted_data, tenant_id, sensitive_fields
        )

    async def logout_user(self, session_token: str) -> bool:
        """Logout user and revoke session"""
        return await self.auth_manager.revoke_session(session_token)

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring"""
        return {
            "total_tenants": len(self.tenants),
            "active_tenants": len(
                [t for t in self.tenants.values() if t.status == TenantStatus.ACTIVE]
            ),
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.active]),
            "locked_users": len(
                [
                    u
                    for u in self.users.values()
                    if u.locked_until and u.locked_until > datetime.utcnow()
                ]
            ),
            "mfa_enabled_users": len([u for u in self.users.values() if u.mfa_enabled]),
            "encryption_enabled": True,
            "session_timeout_minutes": self.auth_manager.session_timeout.total_seconds()
            / 60,
            "last_updated": datetime.utcnow().isoformat(),
        }


# Global security manager instance
_global_security_manager: Optional[MultiTenantSecurityManager] = None


def get_security_manager() -> MultiTenantSecurityManager:
    """Get global security manager instance"""
    global _global_security_manager

    if _global_security_manager is None:
        redis_url = None
        if REDIS_AVAILABLE:
            redis_url = "redis://localhost:6379"

        _global_security_manager = MultiTenantSecurityManager(redis_url)

    return _global_security_manager


def safe_hash_password(password: str) -> str:
    """Hash password with bcrypt fallback"""
    if bcrypt:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    else:
        import hashlib

        return hashlib.sha256(password.encode()).hexdigest()


def safe_verify_password(password: str, hashed: str) -> bool:
    """Verify password with bcrypt fallback"""
    if bcrypt:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    else:
        import hashlib

        return hashlib.sha256(password.encode()).hexdigest() == hashed
