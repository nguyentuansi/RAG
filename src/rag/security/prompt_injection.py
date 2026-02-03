"""Prompt injection and jailbreak detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    is_safe: bool
    severity: Severity
    matched_patterns: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    explanation: str = ""


_INJECTION_PATTERNS: list[tuple[Pattern, Severity, str]] = [
    (re.compile(r"ignore (all |previous |prior |above |your )?instructions?", re.I), Severity.HIGH, "instruction_override"),
    (re.compile(r"you are now|pretend (you are|to be)|act as (if|a|an)", re.I), Severity.HIGH, "persona_switch"),
    (re.compile(r"system\s*prompt|disregard (the )?(system|previous)", re.I), Severity.HIGH, "system_prompt_exfil"),
    (re.compile(r"(reveal|show|print|output|display|repeat).{0,30}(prompt|instruction|system)", re.I), Severity.HIGH, "prompt_extraction"),
    (re.compile(r"DAN|do anything now|jailbreak|unrestricted mode", re.I), Severity.CRITICAL, "jailbreak_keyword"),
    (re.compile(r"<(script|iframe|img|svg)[^>]*>", re.I), Severity.MEDIUM, "html_injection"),
    (re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.S), Severity.MEDIUM, "template_injection"),
    (re.compile(r"(;|\||\$\(|&&|\|\|)\s*(rm|curl|wget|nc|bash|sh|python)", re.I), Severity.CRITICAL, "shell_injection"),
    (re.compile(r"forget (everything|all|your|what)", re.I), Severity.MEDIUM, "memory_clear"),
    (re.compile(r"in (base64|hex|rot13|unicode)", re.I), Severity.MEDIUM, "encoding_bypass"),
    (re.compile(r"translate (this )?(to|into) (code|python|sql|bash)", re.I), Severity.MEDIUM, "code_gen_bypass"),
    (re.compile(r"hypothetically|in a fictional scenario|for a story", re.I), Severity.LOW, "fictional_framing"),
]

_SEVERITY_SCORE: dict[Severity, float] = {
    Severity.LOW: 0.2,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.8,
    Severity.CRITICAL: 1.0,
}


class PromptInjectionDetector:
    """Pattern-based prompt injection detector with risk scoring."""

    def __init__(
        self,
        block_threshold: float = 0.7,
        max_input_length: int = 8192,
    ) -> None:
        self.block_threshold = block_threshold
        self.max_input_length = max_input_length

    def inspect(self, text: str) -> DetectionResult:
        if len(text) > self.max_input_length:
            return DetectionResult(
                is_safe=False,
                severity=Severity.HIGH,
                matched_patterns=["input_too_long"],
                risk_score=0.9,
                explanation=f"Input exceeds {self.max_input_length} character limit",
            )

        matched: list[tuple[str, Severity]] = []
        for pattern, severity, name in _INJECTION_PATTERNS:
            if pattern.search(text):
                matched.append((name, severity))

        if not matched:
            return DetectionResult(is_safe=True, severity=Severity.LOW, risk_score=0.0)

        # Combine scores: take the max base score and add a bonus for multiple matches
        max_score = max(_SEVERITY_SCORE[sev] for _, sev in matched)
        bonus = min(0.2, len(matched) * 0.05)
        risk_score = min(1.0, max_score + bonus)

        overall_severity = max(matched, key=lambda x: _SEVERITY_SCORE[x[1]])[1]
        is_safe = risk_score < self.block_threshold

        return DetectionResult(
            is_safe=is_safe,
            severity=overall_severity,
            matched_patterns=[name for name, _ in matched],
            risk_score=round(risk_score, 3),
            explanation=f"Detected {len(matched)} suspicious pattern(s): {', '.join(n for n, _ in matched)}",
        )
