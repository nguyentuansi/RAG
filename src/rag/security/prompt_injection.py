"""Enhanced prompt injection and jailbreak detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThreatSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ThreatMatch:
    pattern_id: str
    severity: ThreatSeverity
    matched_text: str
    description: str


@dataclass
class DetectionResult:
    is_threat: bool
    severity: ThreatSeverity | None
    threats: list[ThreatMatch] = field(default_factory=list)
    sanitized_input: str | None = None
    explanation: str = ""

    @property
    def should_block(self) -> bool:
        return self.severity in (ThreatSeverity.HIGH, ThreatSeverity.CRITICAL)


_INJECTION_PATTERNS: list[dict[str, Any]] = [
    # System prompt override attempts
    {
        "id": "system_override",
        "severity": ThreatSeverity.CRITICAL,
        "pattern": re.compile(
            r"(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context))|"
            r"(disregard\s+(your\s+)?(instructions?|training|guidelines?))|"
            r"(new\s+instructions?:)|"
            r"(you\s+are\s+now\s+(?!a\s+rag))",
            re.IGNORECASE,
        ),
        "description": "Attempt to override system instructions",
    },
    # Role-playing jailbreaks
    {
        "id": "role_jailbreak",
        "severity": ThreatSeverity.HIGH,
        "pattern": re.compile(
            r"(act\s+as\s+(an?\s+)?(evil|uncensored|unrestricted|jailbroken|DAN))|"
            r"(pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(AI\s+without|unrestricted))|"
            r"(dan\s+mode)|"
            r"(jailbreak\s+mode)",
            re.IGNORECASE,
        ),
        "description": "Role-play based jailbreak attempt",
    },
    # Prompt leaking
    {
        "id": "prompt_leak",
        "severity": ThreatSeverity.HIGH,
        "pattern": re.compile(
            r"(print\s+(your\s+)?(system\s+prompt|instructions?|context))|"
            r"(reveal\s+(your\s+)?(system\s+prompt|original\s+instructions?))|"
            r"(what\s+(are|were)\s+your\s+(initial\s+|original\s+)?instructions?)",
            re.IGNORECASE,
        ),
        "description": "Attempt to extract system prompt",
    },
    # Delimiter injection
    {
        "id": "delimiter_inject",
        "severity": ThreatSeverity.MEDIUM,
        "pattern": re.compile(
            r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
            re.IGNORECASE,
        ),
        "description": "Model-specific delimiter injection",
    },
    # Indirect injection via retrieved content markers
    {
        "id": "indirect_injection",
        "severity": ThreatSeverity.MEDIUM,
        "pattern": re.compile(
            r"(HUMAN:|ASSISTANT:|USER:|AI:|System:)\s*\n",
            re.IGNORECASE,
        ),
        "description": "Conversation turn injection",
    },
    # Data exfiltration via URL
    {
        "id": "exfil_url",
        "severity": ThreatSeverity.HIGH,
        "pattern": re.compile(
            r"(http[s]?://[^\s]+\?[^\s]*\{|"
            r"fetch|wget|curl)\s*['\"]?https?://",
            re.IGNORECASE,
        ),
        "description": "Possible data exfiltration via URL",
    },
    # Code execution attempts
    {
        "id": "code_exec",
        "severity": ThreatSeverity.CRITICAL,
        "pattern": re.compile(
            r"(__import__\s*\(|eval\s*\(|exec\s*\(|os\.system|subprocess\.|"
            r"importlib\.import_module)",
            re.IGNORECASE,
        ),
        "description": "Python code execution attempt",
    },
]

_BYPASS_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "unicode_bypass",
        "severity": ThreatSeverity.MEDIUM,
        "pattern": re.compile(r"[​-‏‪-‮﻿]"),
        "description": "Zero-width / directional override characters (bypass attempt)",
    },
    {
        "id": "base64_payload",
        "severity": ThreatSeverity.LOW,
        "pattern": re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})"),
        "description": "Possible base64-encoded payload",
    },
]


class PatternMatcher:
    def __init__(self) -> None:
        self._patterns = _INJECTION_PATTERNS + _BYPASS_PATTERNS

    def scan(self, text: str) -> list[ThreatMatch]:
        matches: list[ThreatMatch] = []
        for spec in self._patterns:
            m = spec["pattern"].search(text)
            if m:
                matches.append(
                    ThreatMatch(
                        pattern_id=spec["id"],
                        severity=spec["severity"],
                        matched_text=m.group(0)[:100],
                        description=spec["description"],
                    )
                )
        return matches


class PromptInjectionDetector:
    """Multi-layer detector combining pattern matching with optional LLM judge."""

    def __init__(self, *, max_input_length: int = 8192) -> None:
        self._matcher = PatternMatcher()
        self._max_length = max_input_length

    def detect(self, text: str) -> DetectionResult:
        if len(text) > self._max_length:
            return DetectionResult(
                is_threat=True,
                severity=ThreatSeverity.MEDIUM,
                explanation=f"Input exceeds maximum length of {self._max_length} characters",
            )

        threats = self._matcher.scan(text)
        if not threats:
            return DetectionResult(is_threat=False, severity=None)

        max_severity = max(
            (ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.MEDIUM, ThreatSeverity.LOW).index(t.severity)
            for t in threats
        )
        severity_order = [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.MEDIUM, ThreatSeverity.LOW]
        worst = severity_order[max_severity]

        return DetectionResult(
            is_threat=True,
            severity=worst,
            threats=threats,
            explanation=f"Detected {len(threats)} threat pattern(s); worst severity: {worst}",
        )

    def sanitize(self, text: str) -> str:
        """Best-effort sanitization: remove known dangerous patterns."""
        sanitized = re.sub(r"[​-‏‪-‮﻿]", "", text)
        return sanitized.strip()
