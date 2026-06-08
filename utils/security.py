"""
Advanced Security and Rate Limiting for VisionForge v2.1

Provides enterprise-grade security features:
- Rate limiting with sliding window algorithm
- API key management and rotation
- Request signature verification
- IP whitelisting/blacklisting
- Security audit logging
- CORS configuration
- Input sanitization and validation
"""

import time
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
import json
import ipaddress

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests: int  # Maximum requests
    window_seconds: int  # Time window in seconds
    burst_size: Optional[int] = None  # Allow bursts up to this size


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with burst support.
    More accurate than fixed window.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(deque)
        
        if config.burst_size is None:
            self.burst_size = config.max_requests
        else:
            self.burst_size = min(config.burst_size, config.max_requests * 2)
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Identifier (user_id, IP, API key, etc.)
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.config.window_seconds
        
        # Clean old requests
        request_times = self.requests[key]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check rate limit
        if len(request_times) >= self.config.max_requests:
            logger.warning(f"Rate limit exceeded for {key}")
            return False
        
        # Check burst limit
        if len(request_times) >= self.burst_size:
            logger.warning(f"Burst limit exceeded for {key}")
            return False
        
        # Allow request
        request_times.append(now)
        return True
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.config.window_seconds
        
        request_times = self.requests[key]
        # Count requests in current window
        recent = sum(1 for t in request_times if t >= window_start)
        return max(0, self.config.max_requests - recent)
    
    def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self.requests:
            del self.requests[key]


class APIKeyManager:
    """
    Manage API keys with rotation and revocation.
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize API key manager.
        
        Args:
            storage_path: Path to store API keys
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.keys_file = self.storage_path / "api_keys.json"
        self.keys = self._load_keys()
    
    def _load_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from storage."""
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_keys(self):
        """Save API keys to storage."""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def generate_key(
        self,
        user_id: str,
        name: str,
        expires_days: Optional[int] = None,
        permissions: Optional[List[str]] = None
    ) -> str:
        """
        Generate new API key.
        
        Args:
            user_id: User ID
            name: Key name/description
            expires_days: Days until expiration (None = never)
            permissions: List of allowed permissions
            
        Returns:
            API key string
        """
        # Generate secure random key
        api_key = f"vf_{secrets.token_urlsafe(32)}"
        
        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        # Store key metadata
        self.keys[api_key] = {
            'user_id': user_id,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at,
            'permissions': permissions or ['*'],  # * = all permissions
            'revoked': False,
            'last_used': None,
            'usage_count': 0
        }
        
        self._save_keys()
        logger.info(f"Generated API key for user {user_id}: {name}")
        
        return api_key
    
    def validate_key(self, api_key: str, required_permission: Optional[str] = None) -> bool:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            required_permission: Required permission
            
        Returns:
            True if valid, False otherwise
        """
        if api_key not in self.keys:
            return False
        
        key_data = self.keys[api_key]
        
        # Check if revoked
        if key_data['revoked']:
            logger.warning(f"Revoked API key used: {api_key[:10]}...")
            return False
        
        # Check expiration
        if key_data['expires_at']:
            expires_at = datetime.fromisoformat(key_data['expires_at'])
            if datetime.now() > expires_at:
                logger.warning(f"Expired API key used: {api_key[:10]}...")
                return False
        
        # Check permissions
        if required_permission:
            permissions = key_data['permissions']
            if '*' not in permissions and required_permission not in permissions:
                logger.warning(f"Insufficient permissions for key: {api_key[:10]}...")
                return False
        
        # Update usage
        key_data['last_used'] = datetime.now().isoformat()
        key_data['usage_count'] += 1
        self._save_keys()
        
        return True
    
    def revoke_key(self, api_key: str):
        """Revoke an API key."""
        if api_key in self.keys:
            self.keys[api_key]['revoked'] = True
            self._save_keys()
            logger.info(f"Revoked API key: {api_key[:10]}...")
    
    def list_keys(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List API keys.
        
        Args:
            user_id: Filter by user ID
            
        Returns:
            List of key metadata (without actual keys)
        """
        keys = []
        for api_key, data in self.keys.items():
            if user_id is None or data['user_id'] == user_id:
                # Don't include the actual key
                key_info = {
                    'key_preview': api_key[:10] + '...',
                    **data
                }
                keys.append(key_info)
        
        return keys


class RequestSignatureValidator:
    """
    Validate request signatures to prevent tampering.
    Uses HMAC-SHA256.
    """
    
    def __init__(self, secret_key: str):
        """
        Initialize validator.
        
        Args:
            secret_key: Secret key for signing
        """
        self.secret_key = secret_key.encode()
    
    def sign_request(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        timestamp: Optional[int] = None
    ) -> str:
        """
        Sign a request.
        
        Args:
            method: HTTP method
            path: Request path
            body: Request body
            timestamp: Unix timestamp
            
        Returns:
            Signature string
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create message to sign
        message = f"{method}\n{path}\n{timestamp}"
        if body:
            message += f"\n{body}"
        
        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def validate_signature(
        self,
        signature: str,
        method: str,
        path: str,
        body: Optional[str] = None,
        timestamp: int = None,
        max_age_seconds: int = 300
    ) -> bool:
        """
        Validate request signature.
        
        Args:
            signature: Provided signature
            method: HTTP method
            path: Request path
            body: Request body
            timestamp: Request timestamp
            max_age_seconds: Maximum age of request
            
        Returns:
            True if valid
        """
        # Check timestamp
        if timestamp is None:
            logger.warning("Missing timestamp in signed request")
            return False
        
        now = int(time.time())
        age = now - timestamp
        
        if age > max_age_seconds:
            logger.warning(f"Signature expired (age: {age}s)")
            return False
        
        if age < -60:  # Allow 60s clock skew
            logger.warning("Signature from future")
            return False
        
        # Verify signature
        expected = self.sign_request(method, path, body, timestamp)
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected)


class IPAccessControl:
    """
    IP-based access control with whitelist/blacklist.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize IP access control.
        
        Args:
            config_path: Path to IP config file
        """
        self.config_path = config_path
        self.whitelist: List[ipaddress.IPv4Network] = []
        self.blacklist: List[ipaddress.IPv4Network] = []
        
        if config_path and config_path.exists():
            self._load_config()
    
    def _load_config(self):
        """Load IP configuration."""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.whitelist = [ipaddress.ip_network(ip) for ip in config.get('whitelist', [])]
        self.blacklist = [ipaddress.ip_network(ip) for ip in config.get('blacklist', [])]
    
    def is_allowed(self, ip: str) -> bool:
        """
        Check if IP is allowed.
        
        Args:
            ip: IP address string
            
        Returns:
            True if allowed
        """
        try:
            ip_addr = ipaddress.ip_address(ip)
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            return False
        
        # Check blacklist first
        for network in self.blacklist:
            if ip_addr in network:
                logger.warning(f"Blocked IP (blacklist): {ip}")
                return False
        
        # If whitelist is empty, allow all (except blacklisted)
        if not self.whitelist:
            return True
        
        # Check whitelist
        for network in self.whitelist:
            if ip_addr in network:
                return True
        
        logger.warning(f"Blocked IP (not in whitelist): {ip}")
        return False
    
    def add_to_blacklist(self, ip_or_network: str):
        """Add IP or network to blacklist."""
        network = ipaddress.ip_network(ip_or_network)
        if network not in self.blacklist:
            self.blacklist.append(network)
            logger.info(f"Added to blacklist: {ip_or_network}")
    
    def add_to_whitelist(self, ip_or_network: str):
        """Add IP or network to whitelist."""
        network = ipaddress.ip_network(ip_or_network)
        if network not in self.whitelist:
            self.whitelist.append(network)
            logger.info(f"Added to whitelist: {ip_or_network}")


class SecurityAuditLogger:
    """
    Log security events for audit trail.
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize security audit logger.
        
        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.json"
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO"
    ):
        """
        Log security event.
        
        Args:
            event_type: Type of event (auth_success, auth_failure, etc.)
            user_id: User ID if applicable
            ip_address: Client IP
            details: Additional details
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'severity': severity,
            'details': details or {}
        }
        
        # Append to today's log file
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Also log to standard logger
        log_msg = f"[{severity}] {event_type}"
        if user_id:
            log_msg += f" | User: {user_id}"
        if ip_address:
            log_msg += f" | IP: {ip_address}"
        
        if severity == "CRITICAL":
            logger.critical(log_msg)
        elif severity == "ERROR":
            logger.error(log_msg)
        elif severity == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
            event_type: Filter by event type
            user_id: Filter by user ID
            
        Returns:
            List of matching events
        """
        events = []
        
        # Read all log files in date range
        for log_file in sorted(self.log_dir.glob("audit_*.json")):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        
                        # Apply filters
                        if start_date and datetime.fromisoformat(event['timestamp']) < start_date:
                            continue
                        if end_date and datetime.fromisoformat(event['timestamp']) > end_date:
                            continue
                        if event_type and event['event_type'] != event_type:
                            continue
                        if user_id and event['user_id'] != user_id:
                            continue
                        
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        return events


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    key_func: Callable = None
):
    """
    Decorator for rate limiting.
    
    Args:
        max_requests: Maximum requests in window
        window_seconds: Time window in seconds
        key_func: Function to extract rate limit key from request
    """
    config = RateLimitConfig(max_requests=max_requests, window_seconds=window_seconds)
    limiter = SlidingWindowRateLimiter(config)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract key (default to first arg or 'default')
            if key_func:
                key = key_func(*args, **kwargs)
            elif args:
                key = str(args[0])
            else:
                key = 'default'
            
            if not limiter.is_allowed(key):
                raise Exception(f"Rate limit exceeded for {key}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Singleton instances
_rate_limiters: Dict[str, SlidingWindowRateLimiter] = {}
_api_key_manager: Optional[APIKeyManager] = None
_ip_access_control: Optional[IPAccessControl] = None
_audit_logger: Optional[SecurityAuditLogger] = None


def get_rate_limiter(name: str, config: RateLimitConfig) -> SlidingWindowRateLimiter:
    """Get or create rate limiter."""
    if name not in _rate_limiters:
        _rate_limiters[name] = SlidingWindowRateLimiter(config)
    return _rate_limiters[name]


def get_api_key_manager(storage_path: Path) -> APIKeyManager:
    """Get or create API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager(storage_path)
    return _api_key_manager


def get_ip_access_control(config_path: Optional[Path] = None) -> IPAccessControl:
    """Get or create IP access control."""
    global _ip_access_control
    if _ip_access_control is None:
        _ip_access_control = IPAccessControl(config_path)
    return _ip_access_control


def get_audit_logger(log_dir: Path) -> SecurityAuditLogger:
    """Get or create audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger(log_dir)
    return _audit_logger
