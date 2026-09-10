"""PII redaction guardrail for output.

This module provides guardrail protection for redacting
Personally Identifiable Information (PII) from LLM outputs.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


# Default PII entities to redact
DEFAULT_ENTITIES = [
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "US_SSN",
    "IP_ADDRESS",
]


class PIIRedactionGuardrail(GuardrailBase):
    """Redact Personally Identifiable Information from LLM output.

    Uses LLM Guard's Anonymize scanner to identify and redact PII
    from LLM responses before returning to the user.

    Configuration:
        entities: List of entity types to redact
        threshold: Detection threshold (0.0-1.0)
        mask_char: Character to use for masking, default "*"
        use_faker: Whether to replace PII with fake data

    Example:
        guardrail = PIIRedactionGuardrail({
            "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
            "mask_char": "X"
        })
        result = await guardrail.check("Contact john@example.com")
    """

    name = "pii_redaction"
    layer = GuardrailLayer.OUTPUT
    description = "Redact PII from LLM outputs"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the PII redaction guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Directory containing pre-downloaded models (optional)
        """
        super().__init__(config)
        self._scanner = None
        self._vault = None
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")
        # Import comprehensive entity list
        from ..input.pii_detection import DEFAULT_ENTITIES
        self._entities: List[str] = self._config.get("entities", DEFAULT_ENTITIES)
        self._mask_char = self._config.get("mask_char", "*")
        self._use_faker = self._config.get("use_faker", False)
        # Lower threshold for better detection
        if "threshold" not in self._config:
            self._threshold = 0.0  # Maximum sensitivity
        self._language = self._config.get("language", "en")

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner with enhanced entity mapping."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Anonymize
            from llm_guard.input_scanners.anonymize_helpers import DEBERTA_AI4PRIVACY_v2_CONF
            from llm_guard.vault import Vault

            self._vault = Vault()
            
            # Configure local model path if available
            # Following LLM Guard's official local_models approach:
            # Mutate the global DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"] object directly
            # Reference: https://github.com/protectai/llm-guard/blob/main/docs/tutorials/notebooks/local_models.ipynb
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "deberta-v3-base_finetuned_ai4privacy_v2"
                
                if local_model_path.exists():
                    self.logger.info(
                        "pii_redaction_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Mutate the shared Model object to point to local path
                    DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].path = str(local_model_path)
                    DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].kwargs["local_files_only"] = True
                else:
                    self.logger.warning(
                        "pii_redaction_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners pii_redaction"
                    )
            
            # Create enhanced recognizer configuration (same as input guardrail)
            recognizer_conf = DEBERTA_AI4PRIVACY_v2_CONF
            
            # Import enhanced regex patterns and recognizers
            try:
                from ..input.enhanced_patterns import get_enhanced_regex_patterns
                from ..input.custom_recognizers import get_enhanced_recognizers
                regex_patterns = get_enhanced_regex_patterns()
                recognizer_conf["CUSTOM_RECOGNIZERS"] = get_enhanced_recognizers()
            except ImportError:
                regex_patterns = []
                self.logger.warning("enhanced_patterns_not_found", message="Using base patterns only")
            
            # CRITICAL FIX: Extend MODEL_TO_PRESIDIO_MAPPING for all 54 Deberta labels
            enhanced_mapping = {
                **recognizer_conf.get("MODEL_TO_PRESIDIO_MAPPING", {}),
                "ACCOUNTNAME": "ACCOUNTNAME",
                "ACCOUNTNUMBER": "ACCOUNTNUMBER",
                "PREFIX": "PREFIX",
                "USERNAME": "USERNAME",
                "GENDER": "GENDER",
                "SEX": "SEX",
                "EYECOLOR": "EYECOLOR",
                "HEIGHT": "HEIGHT",
                "PHONEIMEI": "PHONEIMEI",
                "ORDINALDIRECTION": "ORDINALDIRECTION",
                "CREDITCARDCVV": "CREDITCARDCVV",
                "CREDITCARDISSUER": "CREDITCARDISSUER",
                "MASKEDNUMBER": "MASKEDNUMBER",
                "BIC": "BIC",
                "AMOUNT": "AMOUNT",
                "CURRENCY": "CURRENCY",
                "CURRENCYCODE": "CURRENCYCODE",
                "CURRENCYNAME": "CURRENCYNAME",
                "CURRENCYSYMBOL": "CURRENCYSYMBOL",
                "SSN": "US_SSN",
                "VEHICLEVIN": "VEHICLEVIN",
                "VEHICLEVRM": "VEHICLEVRM",
                "MAC": "MAC_ADDRESS",
                "USERAGENT": "USERAGENT",
                "JOBTITLE": "JOBTITLE",
                "JOBAREA": "JOBAREA",
                "JOBTYPE": "JOBTYPE",
                "PASSWORD": "PASSWORD",
                "PIN": "PIN",
            }
            
            recognizer_conf["MODEL_TO_PRESIDIO_MAPPING"] = enhanced_mapping
            
            # CRITICAL FIX: Expand PRESIDIO_SUPPORTED_ENTITIES
            all_supported_entities = list(set(
                list(recognizer_conf.get("PRESIDIO_SUPPORTED_ENTITIES", [])) +
                list(enhanced_mapping.values()) +
                [
                    "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_ITIN", "US_BANK_NUMBER",
                    "UK_NHS", "UK_NINO",
                    "ES_NIF", "ES_NIE",
                    "IT_FISCAL_CODE", "IT_DRIVER_LICENSE", "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD",
                    "PL_PESEL",
                    "SG_NRIC_FIN", "SG_UEN",
                    "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
                    "IN_PAN", "IN_AADHAAR", "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "IN_GSTIN",
                    "FI_PERSONAL_IDENTITY_CODE", "KR_RRN", "TH_TNIN",
                    "NRP", "MEDICAL_LICENSE",
                ]
            ))
            
            recognizer_conf["PRESIDIO_SUPPORTED_ENTITIES"] = all_supported_entities

            self._scanner = Anonymize(
                vault=self._vault,
                entity_types=self._entities,
                use_faker=self._use_faker,
                threshold=self._threshold,
                language=self._language,
                recognizer_conf=recognizer_conf,
                regex_patterns=regex_patterns,  # Enhanced regex patterns
            )

            self.logger.info(
                "pii_redaction_scanner_initialized",
                entities=self._entities,
                threshold=self._threshold,
                using_enhanced_mapping=True,
                total_supported_entities=len(all_supported_entities),
            )

        except ImportError:
            self.logger.warning(
                "llm_guard_not_installed",
                message="Install with: pip install neo-guardrail-hub[llm-guard]",
            )
            self._scanner = None
            self._vault = None

        await super().initialize()

    async def _check(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> GuardrailResult:
        """Check and redact PII from output text.

        Args:
            text: LLM output text to check
            context: Optional context

        Returns:
            GuardrailResult with sanitized text if PII found
        """
        if self._scanner is None:
            # Fallback: use simple pattern matching
            return self._fallback_redact(text)

        try:
            sanitized_text, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk_score to 0.0-1.0 range (LLM Guard may return -1.0 for safe content)
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # Determine what was redacted
            redacted_entities = []
            if sanitized_text != text:
                redacted_entities = self._get_redacted_entities(text, sanitized_text)

            # For output redaction, we pass but provide sanitized text
            # The action is always "sanitize" for this guardrail
            return GuardrailResult(
                passed=True,  # We don't block, we sanitize
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    f"PII redacted: {', '.join(redacted_entities)}"
                    if redacted_entities
                    else None
                ),
                sanitized_text=sanitized_text,
                metadata={
                    "redacted_entities": redacted_entities,
                    "original_length": len(text),
                    "sanitized_length": len(sanitized_text),
                    "pii_found": len(redacted_entities) > 0,
                    "on_fail": "sanitize",  # Always sanitize for output
                    "raw_risk_score": risk_score,  # Store original before normalization
                },
            )

        except Exception as e:
            self.logger.error(
                "pii_redaction_error",
                error=str(e),
            )
            # On error, return original text (fail open for output)
            return GuardrailResult(
                passed=True,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.5,
                message=f"Redaction error (text unchanged): {str(e)}",
                sanitized_text=text,
                metadata={"error": str(e)},
            )

    def _get_redacted_entities(
        self, original: str, sanitized: str
    ) -> List[str]:
        """Determine which entity types were redacted."""
        entities = []

        patterns = {
            "[EMAIL_ADDRESS": "EMAIL_ADDRESS",
            "[PHONE_NUMBER": "PHONE_NUMBER",
            "[PERSON": "PERSON",
            "[CREDIT_CARD": "CREDIT_CARD",
            "[US_SSN": "US_SSN",
            "[IP_ADDRESS": "IP_ADDRESS",
            "[URL": "URL",
        }

        for pattern, entity_type in patterns.items():
            if pattern in sanitized:
                entities.append(entity_type)

        return entities

    def _fallback_redact(self, text: str) -> GuardrailResult:
        """Simple fallback redaction when LLM Guard is not available."""
        import re

        redacted = []
        sanitized_text = text

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, sanitized_text):
            sanitized_text = re.sub(email_pattern, "[EMAIL_REDACTED]", sanitized_text)
            redacted.append("EMAIL_ADDRESS")

        # Phone pattern
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        if re.search(phone_pattern, sanitized_text):
            sanitized_text = re.sub(phone_pattern, "[PHONE_REDACTED]", sanitized_text)
            redacted.append("PHONE_NUMBER")

        # SSN pattern
        ssn_pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        if re.search(ssn_pattern, sanitized_text):
            sanitized_text = re.sub(ssn_pattern, "[SSN_REDACTED]", sanitized_text)
            redacted.append("US_SSN")

        # Credit card pattern
        cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        if re.search(cc_pattern, sanitized_text):
            sanitized_text = re.sub(cc_pattern, "[CARD_REDACTED]", sanitized_text)
            redacted.append("CREDIT_CARD")

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.8 if redacted else 0.0,
            message=f"PII redacted: {', '.join(redacted)}" if redacted else None,
            sanitized_text=sanitized_text,
            metadata={
                "redacted_entities": redacted,
                "detection_method": "pattern_matching",
                "on_fail": "sanitize",
            },
        )

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        self._vault = None
        await super().cleanup()
