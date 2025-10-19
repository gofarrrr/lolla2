"""
METIS Enterprise Authentication Foundation
F004: OAuth 2.0 + RBAC framework using Xpander.ai proven patterns

Implements enterprise-grade authentication and authorization for
multi-tenant SaaS deployment with SOC 2 compliance foundation.
"""

import asyncio
import jwt
import secrets
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

try:
    from passlib.context import CryptContext
    from passlib.hash import bcrypt

    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False


class UserRole(str, Enum):
    """Hierarchical user roles for RBAC"""

    SUPER_ADMIN = "super_admin"  # Platform administration
    ORG_ADMIN = "org_admin"  # Organization administration
    ENGAGEMENT_LEAD = "engagement_lead"  # Strategic engagement management
    ANALYST = "analyst"  # Analysis execution
    VIEWER = "viewer"  # Read-only access
    API_CLIENT = "api_client"  # Programmatic access


class Permission(str, Enum):
    """Granular permissions for enterprise features"""

    # Engagement Management
    CREATE_ENGAGEMENT = "create_engagement"
    READ_ENGAGEMENT = "read_engagement"
    UPDATE_ENGAGEMENT = "update_engagement"
    DELETE_ENGAGEMENT = "delete_engagement"

    # Cognitive Operations
    EXECUTE_ANALYSIS = "execute_analysis"
    SELECT_MODELS = "select_models"
    VIEW_REASONING = "view_reasoning"
    EXPORT_RESULTS = "export_results"

    # Model Management
    CREATE_MODEL = "create_model"
    UPDATE_MODEL = "update_model"
    VALIDATE_MODEL = "validate_model"
    RETIRE_MODEL = "retire_model"

    # Organization Management
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_BILLING = "manage_billing"

    # System Administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_METRICS = "view_metrics"
    CONFIGURE_INTEGRATIONS = "configure_integrations"


class SessionStatus(str, Enum):
    """User session status tracking"""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class Organization:
    """Multi-tenant organization entity"""

    org_id: UUID = field(default_factory=uuid4)
    name: str = ""
    domain: Optional[str] = None
    subscription_tier: str = "basic"  # basic, professional, enterprise
    sso_enabled: bool = False
    sso_provider: Optional[str] = None
    sso_config: Dict[str, Any] = field(default_factory=dict)
    compliance_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class User:
    """User entity with RBAC integration"""

    user_id: UUID = field(default_factory=uuid4)
    email: str = ""
    password_hash: Optional[str] = None
    first_name: str = ""
    last_name: str = ""
    organization_id: UUID = field(default_factory=uuid4)
    roles: List[UserRole] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    sso_provider_id: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session with security tracking"""

    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    access_token: str = ""
    refresh_token: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=8)
    )
    last_activity: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    security_context: Dict[str, Any] = field(default_factory=dict)


class MetisAuthenticationManager:
    """
    Enterprise authentication manager implementing OAuth 2.0 + RBAC
    Based on Xpander.ai proven security patterns
    """

    def __init__(
        self,
        jwt_secret_key: str,
        jwt_algorithm: str = "HS256",
        access_token_expire_hours: int = 8,
        refresh_token_expire_days: int = 30,
    ):
        self.jwt_secret = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire = timedelta(hours=access_token_expire_hours)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)

        # Password security
        if PASSLIB_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        else:
            self.pwd_context = None

        # In-memory stores (replace with database in production)
        self.organizations: Dict[UUID, Organization] = {}
        self.users: Dict[UUID, User] = {}
        self.sessions: Dict[UUID, Session] = {}
        self.user_email_index: Dict[str, UUID] = {}

        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30

        # Role-Permission mapping
        self.role_permissions = self._initialize_role_permissions()

        self.logger = logging.getLogger(__name__)

        # Initialize with default organization and admin user
        asyncio.create_task(self._initialize_default_data())

    def _initialize_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Initialize role-permission mappings"""
        return {
            UserRole.SUPER_ADMIN: list(Permission),  # All permissions
            UserRole.ORG_ADMIN: [
                Permission.CREATE_ENGAGEMENT,
                Permission.READ_ENGAGEMENT,
                Permission.UPDATE_ENGAGEMENT,
                Permission.DELETE_ENGAGEMENT,
                Permission.EXECUTE_ANALYSIS,
                Permission.SELECT_MODELS,
                Permission.VIEW_REASONING,
                Permission.EXPORT_RESULTS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_ROLES,
                Permission.VIEW_AUDIT_LOGS,
                Permission.MANAGE_BILLING,
                Permission.VIEW_METRICS,
                Permission.CONFIGURE_INTEGRATIONS,
            ],
            UserRole.ENGAGEMENT_LEAD: [
                Permission.CREATE_ENGAGEMENT,
                Permission.READ_ENGAGEMENT,
                Permission.UPDATE_ENGAGEMENT,
                Permission.EXECUTE_ANALYSIS,
                Permission.SELECT_MODELS,
                Permission.VIEW_REASONING,
                Permission.EXPORT_RESULTS,
                Permission.VIEW_AUDIT_LOGS,
            ],
            UserRole.ANALYST: [
                Permission.READ_ENGAGEMENT,
                Permission.EXECUTE_ANALYSIS,
                Permission.VIEW_REASONING,
                Permission.EXPORT_RESULTS,
            ],
            UserRole.VIEWER: [Permission.READ_ENGAGEMENT, Permission.VIEW_REASONING],
            UserRole.API_CLIENT: [
                Permission.CREATE_ENGAGEMENT,
                Permission.READ_ENGAGEMENT,
                Permission.EXECUTE_ANALYSIS,
                Permission.EXPORT_RESULTS,
            ],
        }

    async def _initialize_default_data(self):
        """Initialize default organization and admin user"""
        try:
            # Create default organization
            default_org = Organization(
                name="METIS Platform",
                domain="metis.ai",
                subscription_tier="enterprise",
                compliance_settings={
                    "soc2_compliance": True,
                    "gdpr_compliance": True,
                    "audit_retention_days": 2555,  # 7 years
                },
            )
            self.organizations[default_org.org_id] = default_org

            # Create admin user
            admin_user = User(
                email="admin@metis.ai",
                first_name="System",
                last_name="Administrator",
                organization_id=default_org.org_id,
                roles=[UserRole.SUPER_ADMIN],
            )

            # Set default password (should be changed on first login)
            admin_password = "MetisAdmin2024!"
            admin_user.password_hash = await self._hash_password(admin_password)

            self.users[admin_user.user_id] = admin_user
            self.user_email_index[admin_user.email] = admin_user.user_id

            self.logger.info("Initialized default organization and admin user")

        except Exception as e:
            self.logger.error(f"Failed to initialize default data: {e}")

    async def create_organization(
        self, name: str, domain: Optional[str] = None, subscription_tier: str = "basic"
    ) -> Organization:
        """Create new multi-tenant organization"""

        org = Organization(
            name=name,
            domain=domain,
            subscription_tier=subscription_tier,
            compliance_settings={
                "soc2_compliance": subscription_tier == "enterprise",
                "gdpr_compliance": True,
                "audit_retention_days": 365 if subscription_tier == "basic" else 2555,
            },
        )

        self.organizations[org.org_id] = org
        self.logger.info(f"Created organization: {name} ({org.org_id})")

        return org

    async def create_user(
        self,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        organization_id: UUID,
        roles: List[UserRole],
        created_by_user_id: Optional[UUID] = None,
    ) -> User:
        """Create new user with role-based permissions"""

        # Validate organization exists
        if organization_id not in self.organizations:
            raise ValueError(f"Organization {organization_id} not found")

        # Check email uniqueness
        if email in self.user_email_index:
            raise ValueError(f"User with email {email} already exists")

        # Validate creator permissions (if applicable)
        if created_by_user_id:
            creator = self.users.get(created_by_user_id)
            if not creator or not await self._check_permission(
                creator, Permission.MANAGE_USERS
            ):
                raise ValueError("Insufficient permissions to create user")

        # Create user
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            organization_id=organization_id,
            roles=roles,
        )

        # Hash password
        user.password_hash = await self._hash_password(password)

        # Assign role-based permissions
        user.permissions = await self._calculate_user_permissions(user.roles)

        # Store user
        self.users[user.user_id] = user
        self.user_email_index[email] = user.user_id

        self.logger.info(f"Created user: {email} with roles {[r.value for r in roles]}")

        return user

    async def authenticate_user(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[Session]:
        """Authenticate user and create session"""

        user_id = self.user_email_index.get(email)
        if not user_id:
            self.logger.warning(f"Authentication failed: user {email} not found")
            return None

        user = self.users[user_id]

        # Check account lockout
        if await self._is_account_locked(user):
            self.logger.warning(f"Authentication failed: account {email} is locked")
            return None

        # Verify password
        if not await self._verify_password(password, user.password_hash):
            await self._record_failed_login(user)
            self.logger.warning(f"Authentication failed: invalid password for {email}")
            return None

        # Reset failed login attempts on successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()

        # Create session
        session = await self._create_session(user, ip_address, user_agent)

        self.logger.info(f"User authenticated successfully: {email}")
        return session

    async def authenticate_sso(
        self,
        sso_token: str,
        sso_provider: str,
        organization_id: UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[Session]:
        """Authenticate user via SSO provider"""

        # Validate organization supports SSO
        org = self.organizations.get(organization_id)
        if not org or not org.sso_enabled or org.sso_provider != sso_provider:
            self.logger.warning(
                "SSO authentication failed: invalid organization or provider"
            )
            return None

        # Validate SSO token (simplified - would integrate with actual providers)
        user_info = await self._validate_sso_token(
            sso_token, sso_provider, org.sso_config
        )
        if not user_info:
            self.logger.warning("SSO authentication failed: invalid token")
            return None

        # Find or create user
        user_email = user_info.get("email")
        user_id = self.user_email_index.get(user_email)

        if user_id:
            user = self.users[user_id]
        else:
            # Auto-provision user from SSO
            user = await self._auto_provision_sso_user(
                user_info, organization_id, sso_provider
            )

        # Update SSO info
        user.sso_provider_id = user_info.get("sub")
        user.last_login = datetime.utcnow()

        # Create session
        session = await self._create_session(user, ip_address, user_agent)

        self.logger.info(f"SSO user authenticated successfully: {user_email}")
        return session

    async def _create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Create authenticated session with JWT tokens"""

        # Generate tokens
        access_token = await self._generate_access_token(user)
        refresh_token = await self._generate_refresh_token(user)

        # Create session
        session = Session(
            user_id=user.user_id,
            organization_id=user.organization_id,
            access_token=access_token,
            refresh_token=refresh_token,
            ip_address=ip_address,
            user_agent=user_agent,
            security_context={
                "login_method": "sso" if user.sso_provider_id else "password",
                "risk_score": await self._calculate_risk_score(
                    user, ip_address, user_agent
                ),
            },
        )

        self.sessions[session.session_id] = session

        return session

    async def validate_session(
        self, session_id: UUID, update_activity: bool = True
    ) -> Optional[Session]:
        """Validate active session"""

        session = self.sessions.get(session_id)
        if not session:
            return None

        # Check session status
        if session.status != SessionStatus.ACTIVE:
            return None

        # Check expiration
        if datetime.utcnow() > session.expires_at:
            session.status = SessionStatus.EXPIRED
            return None

        # Validate access token
        if not await self._validate_access_token(session.access_token):
            session.status = SessionStatus.REVOKED
            return None

        # Update activity timestamp
        if update_activity:
            session.last_activity = datetime.utcnow()

        return session

    async def refresh_session(self, refresh_token: str) -> Optional[Session]:
        """Refresh session using refresh token"""

        # Find session by refresh token
        session = None
        for s in self.sessions.values():
            if s.refresh_token == refresh_token and s.status == SessionStatus.ACTIVE:
                session = s
                break

        if not session:
            return None

        # Validate refresh token
        if not await self._validate_refresh_token(refresh_token):
            session.status = SessionStatus.REVOKED
            return None

        # Get user
        user = self.users.get(session.user_id)
        if not user or not user.is_active:
            session.status = SessionStatus.REVOKED
            return None

        # Generate new tokens
        session.access_token = await self._generate_access_token(user)
        session.refresh_token = await self._generate_refresh_token(user)
        session.expires_at = datetime.utcnow() + self.access_token_expire
        session.last_activity = datetime.utcnow()

        return session

    async def revoke_session(self, session_id: UUID) -> bool:
        """Revoke user session"""

        session = self.sessions.get(session_id)
        if session:
            session.status = SessionStatus.REVOKED
            self.logger.info(f"Session revoked: {session_id}")
            return True

        return False

    async def check_permission(self, session_id: UUID, permission: Permission) -> bool:
        """Check if session has specific permission"""

        session = await self.validate_session(session_id)
        if not session:
            return False

        user = self.users.get(session.user_id)
        if not user:
            return False

        return await self._check_permission(user, permission)

    async def _check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions

    async def _calculate_user_permissions(
        self, roles: List[UserRole]
    ) -> List[Permission]:
        """Calculate user permissions based on roles"""
        permissions = set()

        for role in roles:
            role_perms = self.role_permissions.get(role, [])
            permissions.update(role_perms)

        return list(permissions)

    async def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        if self.pwd_context:
            return self.pwd_context.hash(password)
        else:
            # Fallback for development without passlib
            salt = secrets.token_hex(16)
            return (
                hashlib.pbkdf2_hmac(
                    "sha256", password.encode(), salt.encode(), 100000
                ).hex()
                + salt
            )

    async def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        if self.pwd_context:
            return self.pwd_context.verify(password, password_hash)
        else:
            # Fallback verification
            if len(password_hash) < 32:
                return False
            salt = password_hash[-32:]
            expected_hash = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 100000
            ).hex()
            return expected_hash == password_hash[:-32]

    async def _is_account_locked(self, user: User) -> bool:
        """Check if account is locked due to failed login attempts"""
        if user.account_locked_until and datetime.utcnow() < user.account_locked_until:
            return True

        if user.failed_login_attempts >= self.max_login_attempts:
            # Lock account
            user.account_locked_until = datetime.utcnow() + timedelta(
                minutes=self.lockout_duration_minutes
            )
            return True

        return False

    async def _record_failed_login(self, user: User):
        """Record failed login attempt"""
        user.failed_login_attempts += 1

        if user.failed_login_attempts >= self.max_login_attempts:
            user.account_locked_until = datetime.utcnow() + timedelta(
                minutes=self.lockout_duration_minutes
            )
            self.logger.warning(
                f"Account locked due to failed login attempts: {user.email}"
            )

    async def _generate_access_token(self, user: User) -> str:
        """Generate JWT access token"""

        payload = {
            "sub": str(user.user_id),
            "email": user.email,
            "org_id": str(user.organization_id),
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.access_token_expire,
            "token_type": "access",
        }

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    async def _generate_refresh_token(self, user: User) -> str:
        """Generate JWT refresh token"""

        payload = {
            "sub": str(user.user_id),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.refresh_token_expire,
            "token_type": "refresh",
            "jti": secrets.token_urlsafe(32),  # JWT ID for token tracking
        }

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    async def _validate_access_token(self, token: str) -> bool:
        """Validate JWT access token"""
        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            return payload.get("token_type") == "access"
        except jwt.InvalidTokenError:
            return False

    async def _validate_refresh_token(self, token: str) -> bool:
        """Validate JWT refresh token"""
        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            return payload.get("token_type") == "refresh"
        except jwt.InvalidTokenError:
            return False

    async def _validate_sso_token(
        self, sso_token: str, sso_provider: str, sso_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Validate SSO token with provider (simplified implementation)"""

        # In production, this would validate with actual SSO providers
        # (Google, Microsoft, Okta, etc.)

        # Simplified validation for demonstration
        try:
            # Would call provider's token validation endpoint
            # For now, return mock user info
            return {
                "sub": "sso_user_123",
                "email": "user@company.com",
                "name": "SSO User",
                "given_name": "SSO",
                "family_name": "User",
            }
        except Exception:
            return None

    async def _auto_provision_sso_user(
        self, user_info: Dict[str, Any], organization_id: UUID, sso_provider: str
    ) -> User:
        """Auto-provision user from SSO authentication"""

        # Default role for SSO users
        default_roles = [UserRole.ANALYST]

        user = User(
            email=user_info["email"],
            first_name=user_info.get("given_name", ""),
            last_name=user_info.get("family_name", ""),
            organization_id=organization_id,
            roles=default_roles,
            sso_provider_id=user_info["sub"],
        )

        # No password for SSO users
        user.password_hash = None

        # Calculate permissions
        user.permissions = await self._calculate_user_permissions(user.roles)

        # Store user
        self.users[user.user_id] = user
        self.user_email_index[user.email] = user.user_id

        self.logger.info(f"Auto-provisioned SSO user: {user.email}")

        return user

    async def _calculate_risk_score(
        self, user: User, ip_address: Optional[str], user_agent: Optional[str]
    ) -> float:
        """Calculate session risk score for security monitoring"""

        risk_score = 0.0

        # Check for unusual login patterns
        if user.last_login:
            time_since_last = datetime.utcnow() - user.last_login
            if time_since_last.days > 30:
                risk_score += 0.2

        # Check failed login history
        if user.failed_login_attempts > 0:
            risk_score += user.failed_login_attempts * 0.1

        # IP-based risk (simplified)
        if ip_address:
            # Would check against threat intelligence feeds
            pass

        return min(risk_score, 1.0)

    async def get_user_by_session(self, session_id: UUID) -> Optional[User]:
        """Get user information from session"""

        session = await self.validate_session(session_id)
        if not session:
            return None

        return self.users.get(session.user_id)

    async def get_organization_by_session(
        self, session_id: UUID
    ) -> Optional[Organization]:
        """Get organization information from session"""

        session = await self.validate_session(session_id)
        if not session:
            return None

        return self.organizations.get(session.organization_id)

    async def list_active_sessions(self, user_id: UUID) -> List[Session]:
        """List active sessions for user"""

        active_sessions = []
        for session in self.sessions.values():
            if (
                session.user_id == user_id
                and session.status == SessionStatus.ACTIVE
                and datetime.utcnow() <= session.expires_at
            ):
                active_sessions.append(session)

        return active_sessions

    async def get_auth_health_status(self) -> Dict[str, Any]:
        """Get authentication system health status"""

        total_users = len(self.users)
        active_sessions = len(
            [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        )
        locked_accounts = len(
            [
                u
                for u in self.users.values()
                if u.account_locked_until and u.account_locked_until > datetime.utcnow()
            ]
        )

        return {
            "total_organizations": len(self.organizations),
            "total_users": total_users,
            "active_sessions": active_sessions,
            "locked_accounts": locked_accounts,
            "sso_enabled_orgs": len(
                [o for o in self.organizations.values() if o.sso_enabled]
            ),
            "system_status": "healthy",
        }


# Global authentication manager instance
_auth_manager_instance: Optional[MetisAuthenticationManager] = None


async def get_auth_manager() -> MetisAuthenticationManager:
    """Get or create global authentication manager"""
    global _auth_manager_instance

    if _auth_manager_instance is None:
        # In production, load from secure configuration
        jwt_secret = secrets.token_urlsafe(32)
        _auth_manager_instance = MetisAuthenticationManager(jwt_secret)

    return _auth_manager_instance


# Utility functions for common auth operations
async def authenticate_request(
    session_id: UUID, required_permission: Permission
) -> Optional[User]:
    """Authenticate request with permission check"""
    auth_manager = await get_auth_manager()

    if await auth_manager.check_permission(session_id, required_permission):
        return await auth_manager.get_user_by_session(session_id)

    return None


async def require_permission(session_id: UUID, permission: Permission) -> bool:
    """Check if session has required permission"""
    auth_manager = await get_auth_manager()
    return await auth_manager.check_permission(session_id, permission)
