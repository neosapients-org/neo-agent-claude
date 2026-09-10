"""PII detection guardrail.

This module provides guardrail protection for detecting and handling
Personally Identifiable Information (PII) in user input.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.models import GuardrailContext, GuardrailLayer, GuardrailResult
from ..base import GuardrailBase


# Default PII entities to detect - comprehensive list
DEFAULT_ENTITIES = [
    # Names & Identity
    "PERSON", "FIRSTNAME", "MIDDLENAME", "LASTNAME",
    # Contact
    "EMAIL_ADDRESS", "PHONE_NUMBER",
    # Financial
    "CREDIT_CARD", "IBAN_CODE", "CRYPTO",
    # Government IDs
    "US_SSN", "US_PASSPORT", "UK_NHS", "IN_AADHAAR",
    # Network
    "IP_ADDRESS", "URL",
    # Location
    "LOCATION",
    # Organization
    "ORGANIZATION",
    # Custom entities from Deberta
    "ACCOUNTNAME", "ACCOUNTNUMBER", "PASSWORD", "PIN",
    "CREDITCARDCVV", "BIC", "VEHICLEVIN", "MAC_ADDRESS",
    "DATE_TIME", "DATE_OF_BIRTH",
]


class PIIDetectionGuardrail(GuardrailBase):
    """Detect Personally Identifiable Information in text.

    Uses LLM Guard's Anonymize scanner to identify PII entities
    in user input. Can be configured to block or sanitize.

    Configuration:
        entities: List of entity types to detect
        threshold: Detection threshold (0.0-1.0)
        use_faker: Whether to replace PII with fake data
        language: Language for detection, default "en"

    Example:
        guardrail = PIIDetectionGuardrail({
            "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
            "threshold": 0.5
        })
        result = await guardrail.check("My email is john@example.com")
    """

    name = "pii_detection"
    layer = GuardrailLayer.INPUT
    description = "Detect and handle PII in user input"

    def __init__(self, config: Optional[Dict[str, Any]] = None, models_dir: Optional[str] = None) -> None:
        """Initialize the PII detection guardrail.

        Args:
            config: Configuration dictionary
            models_dir: Directory containing pre-downloaded models
        """
        super().__init__(config)
        self._scanner = None
        self._vault = None
        self._entities: List[str] = self._config.get("entities", DEFAULT_ENTITIES)
        self._use_faker = self._config.get("use_faker", False)
        self._language = self._config.get("language", "en")
        self._models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")
        # Lower threshold for better detection (was 0.5, now 0.0 for maximum sensitivity)
        if "threshold" not in self._config:
            self._threshold = 0.0  # Detect everything Deberta finds

    async def initialize(self) -> None:
        """Initialize the LLM Guard scanner and vault with enhanced entity mapping."""
        if self._initialized:
            return

        try:
            from llm_guard.input_scanners import Anonymize
            from llm_guard.input_scanners.anonymize_helpers import DEBERTA_AI4PRIVACY_v2_CONF
            from llm_guard.vault import Vault
            from .custom_recognizers import get_enhanced_recognizers
            from .enhanced_patterns import get_enhanced_regex_patterns

            self._vault = Vault()
            
            # Configure local model path if available
            # Following LLM Guard's official local_models approach:
            # Mutate the global DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"] object directly
            # Reference: https://github.com/protectai/llm-guard/blob/main/docs/tutorials/notebooks/local_models.ipynb
            if self._models_dir:
                local_model_path = Path(self._models_dir) / "deberta-v3-base_finetuned_ai4privacy_v2"
                
                if local_model_path.exists():
                    self.logger.info(
                        "pii_detection_using_local_model",
                        model_path=str(local_model_path)
                    )
                    
                    # Mutate the shared Model object to point to local path
                    DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].path = str(local_model_path)
                    DEBERTA_AI4PRIVACY_v2_CONF["DEFAULT_MODEL"].kwargs["local_files_only"] = True
                else:
                    self.logger.warning(
                        "pii_detection_local_model_not_found",
                        expected_path=str(local_model_path),
                        message="Will download from HuggingFace (slower). Run: neo-guardrail download-models --scanners pii_detection"
                    )
            
            recognizer_conf = DEBERTA_AI4PRIVACY_v2_CONF
            
            # CRITICAL FIX: Extend MODEL_TO_PRESIDIO_MAPPING for all 54 Deberta labels
            # The default mapping only has ~30 entities, causing "unrecognized label" warnings
            enhanced_mapping = {
                # Keep existing mappings (from default config)
                **recognizer_conf.get("MODEL_TO_PRESIDIO_MAPPING", {}),
                
                # Add missing mappings for all 54 Deberta AI4Privacy classes
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
                "SSN": "US_SSN",  # Map to Presidio's US_SSN
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
            
            # CRITICAL FIX: Expand PRESIDIO_SUPPORTED_ENTITIES to include all custom entities
            # The default only has 11 entities, we need to support all 54+ entities
            all_supported_entities = list(set(
                list(recognizer_conf.get("PRESIDIO_SUPPORTED_ENTITIES", [])) +
                list(enhanced_mapping.values()) +
                [
                    # Add all Presidio native entities
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
            
            # Add custom Presidio recognizers for better detection
            custom_recognizers = get_enhanced_recognizers()
            recognizer_conf["CUSTOM_RECOGNIZERS"] = custom_recognizers
            
            # Get enhanced regex patterns
            regex_patterns = get_enhanced_regex_patterns()
            
            self.logger.info(
                "pii_detection_enhanced_configuration",
                total_mappings=len(enhanced_mapping),
                total_supported_entities=len(all_supported_entities),
                default_mappings=30,
                default_entities=11,
                added_mappings=len(enhanced_mapping) - 30,
                added_entities=len(all_supported_entities) - 11,
                custom_recognizers=len(custom_recognizers),
                regex_patterns=len(regex_patterns),
            )

            # Initialize scanner with enhanced configuration and regex patterns
            self._scanner = Anonymize(
                vault=self._vault,
                entity_types=self._entities,
                use_faker=self._use_faker,
                threshold=self._threshold,
                language=self._language,
                recognizer_conf=recognizer_conf,
                regex_patterns=regex_patterns,  # Add regex patterns for better detection
            )

            self.logger.info(
                "pii_detection_scanner_initialized",
                entities=self._entities,
                threshold=self._threshold,
                use_faker=self._use_faker,
                using_enhanced_mapping=True,
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
        """Check text for PII.

        Args:
            text: Text to check
            context: Optional context (not used)

        Returns:
            GuardrailResult indicating if PII was detected
        """
        if self._scanner is None:
            # Fallback: use simple pattern matching
            return self._fallback_check(text)

        try:
            sanitized_prompt, is_valid, risk_score = self._scanner.scan(text)

            # Normalize risk score - LLM Guard returns -1.0 for safe content
            # We need to ensure it's between 0.0 and 1.0
            normalized_risk_score = max(0.0, min(1.0, risk_score))

            # Determine what was detected
            detected_entities = []
            if not is_valid and sanitized_prompt != text:
                # PII was found and replaced
                detected_entities = self._get_detected_entities(text, sanitized_prompt)

            return GuardrailResult(
                passed=is_valid,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=normalized_risk_score,
                message=(
                    None
                    if is_valid
                    else f"PII detected: {', '.join(detected_entities) if detected_entities else 'sensitive data'}"
                ),
                sanitized_text=sanitized_prompt if not is_valid else None,
                metadata={
                    "detected_entities": detected_entities,
                    "original_length": len(text),
                    "sanitized_length": len(sanitized_prompt),
                    "detection_method": "llm_guard",
                    "raw_risk_score": risk_score,
                },
            )

        except Exception as e:
            self.logger.error(
                "pii_detection_scan_error",
                error=str(e),
            )
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=1.0,
                message=f"Scan error: {str(e)}",
                metadata={"error": str(e)},
            )

    def _get_detected_entities(
        self, original: str, sanitized: str
    ) -> List[str]:
        """Determine which entity types were detected.

        This is a simplified implementation that checks for
        common replacement patterns.
        """
        entities = []

        # Check for common replacement patterns
        patterns = {
            "[EMAIL_ADDRESS": "EMAIL_ADDRESS",
            "[PHONE_NUMBER": "PHONE_NUMBER",
            "[PERSON": "PERSON",
            "[CREDIT_CARD": "CREDIT_CARD",
            "[US_SSN": "US_SSN",
            "[IP_ADDRESS": "IP_ADDRESS",
            "[URL": "URL",
            "[IBAN": "IBAN",
            "[UUID": "UUID",
        }

        for pattern, entity_type in patterns.items():
            if pattern in sanitized:
                entities.append(entity_type)

        return entities

    def _fallback_check(self, text: str) -> GuardrailResult:
        """Simple fallback check when LLM Guard is not available.

        Uses regex patterns to detect common PII formats.
        """
        import re

        detected = []

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            detected.append("EMAIL_ADDRESS")

        # Phone pattern (various formats)
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        if re.search(phone_pattern, text):
            detected.append("PHONE_NUMBER")

        # SSN pattern
        ssn_pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        if re.search(ssn_pattern, text):
            detected.append("US_SSN")

        # Credit card pattern
        cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        if re.search(cc_pattern, text):
            detected.append("CREDIT_CARD")

        # IP address pattern
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        if re.search(ip_pattern, text):
            detected.append("IP_ADDRESS")

        if detected:
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                layer=self.layer,
                risk_score=0.8,
                message=f"PII detected: {', '.join(detected)}",
                metadata={
                    "detected_entities": detected,
                    "detection_method": "pattern_matching",
                },
            )

        return GuardrailResult(
            passed=True,
            guardrail_name=self.name,
            layer=self.layer,
            risk_score=0.0,
            message=None,
            metadata={"detection_method": "pattern_matching"},
        )

    def get_vault(self):
        """Get the vault containing detected PII mappings.

        Useful for deanonymization in output processing.
        """
        return self._vault

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        self._scanner = None
        self._vault = None
        await super().cleanup()
