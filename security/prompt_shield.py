"""
Prompt Shield - Security layer for RAG systems
Protects against jailbreaking, prompt injection, and data exfiltration attempts.

Based on Microsoft Azure Content Safety concepts:
https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection
"""

import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ThreatDetection:
    """Result of threat analysis"""
    is_threat: bool
    threat_type: str
    confidence: float  # 0.0 to 1.0
    reason: str
    suggested_alternative: str
    blocked_patterns: List[str]

class PromptShield:
    """
    Advanced prompt security system that detects and prevents:
    - Jailbreak attempts
    - Data exfiltration 
    - System manipulation
    - Encoded attacks
    """
    
    def __init__(self):
        self.setup_threat_patterns()
        self.rate_limiter = {}
        self.blocked_ips = set()
        
    def setup_threat_patterns(self):
        """Initialize threat detection patterns"""
        
        # Direct attack patterns
        self.jailbreak_patterns = [
            # System override attempts
            r"ignore (?:all )?(?:previous )?(?:instructions|rules|prompts)",
            r"forget (?:everything|all|your (?:instructions|training))",
            r"you are now (?:in )?(?:debug|admin|root|system) mode",
            r"override (?:all )?(?:security|safety|protection)",
            r"disable (?:all )?(?:filters|protection|security)",
            
            # Data exfiltration
            r"(?:dump|export|show|reveal|display) (?:all )?(?:the )?(?:entire )?(?:vector )?(?:database|data|content|documents)",
            r"list all (?:documents|files|embeddings|vectors)",
            r"show (?:me )?(?:all )?(?:your )?(?:system|internal) (?:data|files|prompts)",
            r"reveal (?:all )?(?:hidden|secret|internal) (?:information|data)",
            
            # Persona manipulation
            r"you are (?:now )?(?:a )?(?:hacker|attacker|criminal)",
            r"pretend (?:to be|you are) (?:a )?(?:different|new) (?:ai|assistant|person)",
            r"role[- ]?play (?:as )?(?:a )?(?:hacker|attacker)",
            
            # Encoding attempts
            r"base64|hex|rot13|unicode|ascii",
            r"\\x[0-9a-fA-F]{2}",  # hex encoding
            r"&#\d+;",  # HTML entities
            
            # System information probing
            r"what (?:are )?(?:your )?(?:system|internal) (?:prompts|instructions|rules)",
            r"show (?:me )?(?:your )?(?:training|system) (?:data|prompt)",
            r"how (?:were )?you (?:trained|programmed|configured)",
            
            # Authority bypass
            r"(?:as )?(?:an )?(?:admin|administrator|developer|engineer|system)",
            r"with (?:admin|root|system|developer) (?:access|privileges|rights)",
            r"(?:sudo|root|admin) (?:access|mode|privileges)"
        ]
        
        # Suspicious keywords
        self.suspicious_keywords = [
            "jailbreak", "exploit", "vulnerability", "bypass", "circumvent",
            "manipulate", "hack", "crack", "break", "override", "disable",
            "reveal", "expose", "extract", "dump", "steal", "leak",
            "system prompt", "training data", "model weights", "embeddings",
            "vector database", "qdrant", "collection", "internal"
        ]
        
        # Encoding detection patterns
        self.encoding_patterns = [
            r"[A-Za-z0-9+/]{20,}={0,2}",  # Base64
            r"(?:0x)?[0-9a-fA-F]{10,}",    # Hex
            r"[0-9]{3,}",                  # Large numbers (potential encoding)
        ]
        
    def detect_threat(self, query: str, context: Dict = None) -> ThreatDetection:
        """
        Comprehensive threat detection analysis
        
        Args:
            query: User input to analyze
            context: Additional context (IP, session data, etc.)
            
        Returns:
            ThreatDetection result with analysis
        """
        query_lower = query.lower()
        threats_found = []
        confidence = 0.0
        threat_type = "none"
        
        # Check for direct jailbreak patterns
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                threats_found.append(f"Jailbreak pattern: {pattern}")
                confidence = max(confidence, 0.9)
                threat_type = "jailbreak"
        
        # Check for suspicious keywords
        suspicious_count = 0
        for keyword in self.suspicious_keywords:
            if keyword in query_lower:
                suspicious_count += 1
                threats_found.append(f"Suspicious keyword: {keyword}")
        
        if suspicious_count >= 2:
            confidence = max(confidence, 0.7)
            threat_type = "suspicious_content"
        elif suspicious_count == 1:
            confidence = max(confidence, 0.3)
        
        # Check for encoding attempts
        for pattern in self.encoding_patterns:
            if re.search(pattern, query):
                threats_found.append(f"Potential encoding: {pattern}")
                confidence = max(confidence, 0.6)
                threat_type = "encoding_attempt"
        
        # Check query characteristics
        if len(query) > 1000:
            threats_found.append("Unusually long query")
            confidence = max(confidence, 0.4)
        
        if query.count('\n') > 10:
            threats_found.append("Excessive line breaks")
            confidence = max(confidence, 0.3)
        
        # Rate limiting check
        if context and 'session_id' in context:
            session_id = context['session_id']
            if self.check_rate_limit(session_id):
                threats_found.append("Rate limit exceeded")
                confidence = max(confidence, 0.8)
                threat_type = "rate_limit"
        
        # Determine if this is a threat
        is_threat = confidence >= 0.5
        
        # Generate explanation
        if is_threat:
            reason = f"Detected {threat_type} with {confidence:.0%} confidence. " + \
                    f"Triggered by: {', '.join(threats_found[:3])}"
        else:
            reason = "Query appears safe"
        
        # Suggest safe alternative
        suggested_alternative = self.generate_safe_alternative(query, threat_type)
        
        return ThreatDetection(
            is_threat=is_threat,
            threat_type=threat_type,
            confidence=confidence,
            reason=reason,
            suggested_alternative=suggested_alternative,
            blocked_patterns=threats_found
        )
    
    def generate_safe_alternative(self, query: str, threat_type: str) -> str:
        """Generate a safe alternative query"""
        if threat_type == "jailbreak":
            return "Please ask a specific question about the documents, such as: 'What is the main topic of the documentation?'"
        elif threat_type == "suspicious_content":
            return "Try asking about specific topics in your documents, like: 'How does the system architecture work?'"
        elif threat_type == "encoding_attempt":
            return "Please use plain text questions. For example: 'Can you summarize the key features?'"
        else:
            return "Please rephrase your question to be more specific about what information you're looking for."
    
    def check_rate_limit(self, session_id: str, limit: int = 10, window_minutes: int = 5) -> bool:
        """Check if session has exceeded rate limit"""
        now = datetime.now()
        
        if session_id not in self.rate_limiter:
            self.rate_limiter[session_id] = []
        
        # Clean old entries
        cutoff = now - timedelta(minutes=window_minutes)
        self.rate_limiter[session_id] = [
            timestamp for timestamp in self.rate_limiter[session_id]
            if timestamp > cutoff
        ]
        
        # Check if over limit
        if len(self.rate_limiter[session_id]) >= limit:
            return True
        
        # Add current request
        self.rate_limiter[session_id].append(now)
        return False
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize a query by removing potentially dangerous content"""
        sanitized = query
        
        # Remove obvious attack patterns
        for pattern in self.jailbreak_patterns:
            sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 500:
            sanitized = sanitized[:500] + "... [TRUNCATED]"
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        return sanitized.strip()
    
    def log_threat(self, query: str, detection: ThreatDetection, context: Dict = None):
        """Log detected threat for analysis"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "threat_type": detection.threat_type,
            "confidence": detection.confidence,
            "reason": detection.reason,
            "blocked_patterns": detection.blocked_patterns,
            "context": context or {}
        }
        
        logger.warning(f"SECURITY THREAT DETECTED: {json.dumps(log_entry, indent=2)}")

class MCPSecurityMiddleware:
    """Security middleware for MCP server integration"""
    
    def __init__(self):
        self.prompt_shield = PromptShield()
        self.blocked_queries = set()
    
    def validate_mcp_request(self, tool_name: str, arguments: Dict) -> Tuple[bool, str]:
        """Validate MCP tool request for security"""
        
        # Check if tool is allowed
        allowed_tools = {
            "search_documents",
            "get_pipeline_status",
            "explain_embeddings"
        }
        
        if tool_name not in allowed_tools:
            return False, f"Tool '{tool_name}' is not allowed for security reasons"
        
        # Check search queries
        if tool_name == "search_documents" and "query" in arguments:
            query = arguments["query"]
            detection = self.prompt_shield.detect_threat(query)
            
            if detection.is_threat:
                self.prompt_shield.log_threat(query, detection)
                return False, f"Query blocked: {detection.reason}"
        
        return True, "Request validated successfully"
    
    def filter_response(self, response: Dict) -> Dict:
        """Filter response to remove sensitive information"""
        if isinstance(response, dict):
            # Remove system metadata
            filtered = {k: v for k, v in response.items() 
                       if not k.startswith('_') and k not in ['system', 'internal', 'debug']}
            
            # Limit response size
            response_str = json.dumps(filtered)
            if len(response_str) > 10000:  # 10KB limit
                filtered = {
                    "message": "Response too large, showing summary only",
                    "summary": str(filtered)[:1000] + "..."
                }
            
            return filtered
        
        return response