"""
Model Registry for LLM Guard Scanners

This module maintains a registry of all HuggingFace models used by LLM Guard scanners,
enabling pre-downloading and local loading to reduce runtime latency.

Key concepts:
- Each scanner has a ModelInfo that maps to its HuggingFace repository
- Models can be downloaded using git clone (with git lfs)
- Scanners are configured to load from local paths using LLM Guard's Model constants

Reference: https://protectai.github.io/llm-guard/tutorials/notebooks/local_models/
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelInfo:
    """Information about an LLM Guard model.
    
    Attributes:
        scanner_type: The type of scanner (e.g., "prompt_injection")
        hf_repo: HuggingFace repository ID (e.g., "protectai/deberta-v3-base-prompt-injection-v2")
        model_constant_import: Python import path to LLM Guard's model constant
                              (e.g., "llm_guard.input_scanners.prompt_injection.V2_MODEL")
        local_dir_name: Directory name when downloaded locally
        requires_model: False for rule-based scanners that don't need models
        onnx_available: Whether an ONNX version is available (faster inference)
    """
    scanner_type: str
    hf_repo: str
    model_constant_import: str
    local_dir_name: str
    requires_model: bool = True
    onnx_available: bool = False


# Registry of all LLM Guard models
LLM_GUARD_MODELS: Dict[str, ModelInfo] = {
    # =========================================================================
    # INPUT SCANNERS
    # =========================================================================
    
    "prompt_injection": ModelInfo(
        scanner_type="prompt_injection",
        hf_repo="protectai/deberta-v3-base-prompt-injection-v2",
        model_constant_import="llm_guard.input_scanners.prompt_injection.V2_MODEL",
        local_dir_name="deberta-v3-base-prompt-injection-v2",
        onnx_available=True
    ),
    
    "pii_detection": ModelInfo(
        scanner_type="pii_detection",
        hf_repo="Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
        model_constant_import="llm_guard.input_scanners.anonymize_helpers.DEBERTA_AI4PRIVACY_v2_CONF",
        local_dir_name="deberta-v3-base_finetuned_ai4privacy_v2",
        onnx_available=True
    ),
    
    "ban_code": ModelInfo(
        scanner_type="ban_code",
        hf_repo="vishnun/codenlbert-sm",  # Using default MODEL_SM
        model_constant_import="llm_guard.input_scanners.ban_code.MODEL_SM",
        local_dir_name="codenlbert-sm",
        onnx_available=True
    ),
    
    "ban_competitors": ModelInfo(
        scanner_type="ban_competitors",
        hf_repo="guishe/nuner-v1_orgs",
        model_constant_import="llm_guard.input_scanners.ban_competitors.MODEL_V1",
        local_dir_name="nuner-v1_orgs",
        onnx_available=True
    ),
    
    "ban_substrings": ModelInfo(
        scanner_type="ban_substrings",
        hf_repo="",  # Rule-based, no model needed
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "ban_topics": ModelInfo(
        scanner_type="ban_topics",
        hf_repo="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",  # Using MODEL_DEBERTA_BASE_V2
        model_constant_import="llm_guard.input_scanners.ban_topics.MODEL_DEBERTA_BASE_V2",
        local_dir_name="deberta-v3-base-zeroshot-v2.0",
        onnx_available=True
    ),
    
    "code_detection": ModelInfo(
        scanner_type="code_detection",
        hf_repo="philomath-1209/programming-language-identification",
        model_constant_import="llm_guard.input_scanners.code.DEFAULT_MODEL",
        local_dir_name="programming-language-identification",
        onnx_available=True
    ),
    
    "gibberish": ModelInfo(
        scanner_type="gibberish",
        hf_repo="madhurjindal/autonlp-Gibberish-Detector-492513457",
        model_constant_import="llm_guard.input_scanners.gibberish.DEFAULT_MODEL",
        local_dir_name="autonlp-Gibberish-Detector-492513457",
        onnx_available=True
    ),
    
    "language": ModelInfo(
        scanner_type="language",
        hf_repo="papluca/xlm-roberta-base-language-detection",
        model_constant_import="llm_guard.input_scanners.language.DEFAULT_MODEL",
        local_dir_name="xlm-roberta-base-language-detection",
        onnx_available=True
    ),
    
    "toxicity": ModelInfo(
        scanner_type="toxicity",
        hf_repo="unitary/unbiased-toxic-roberta",
        model_constant_import="llm_guard.input_scanners.toxicity.DEFAULT_MODEL",
        local_dir_name="unbiased-toxic-roberta",
        onnx_available=True
    ),
    
    "toxicity_input": ModelInfo(
        scanner_type="toxicity_input",
        hf_repo="unitary/unbiased-toxic-roberta",
        model_constant_import="llm_guard.input_scanners.toxicity.DEFAULT_MODEL",
        local_dir_name="unbiased-toxic-roberta",
        onnx_available=True
    ),
    
    # Rule-based scanners (no models required)
    "regex": ModelInfo(
        scanner_type="regex",
        hf_repo="",  # Rule-based
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "secret_detection": ModelInfo(
        scanner_type="secret_detection",
        hf_repo="",  # Regex + entropy heuristics
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "sentiment": ModelInfo(
        scanner_type="sentiment",
        hf_repo="",  # NLTK Vader (no HuggingFace model)
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "input_length": ModelInfo(
        scanner_type="input_length",
        hf_repo="",  # Rule-based token counting
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    # =========================================================================
    # OUTPUT SCANNERS
    # =========================================================================
    
    "ban_competitors_output": ModelInfo(
        scanner_type="ban_competitors_output",
        hf_repo="guishe/nuner-v1_orgs",  # Same as input scanner
        model_constant_import="llm_guard.input_scanners.ban_competitors.MODEL_V1",
        local_dir_name="nuner-v1_orgs",
        onnx_available=True
    ),
    
    "ban_topics_output": ModelInfo(
        scanner_type="ban_topics_output",
        hf_repo="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",  # Same as input scanner
        model_constant_import="llm_guard.input_scanners.ban_topics.MODEL_DEBERTA_BASE_V2",
        local_dir_name="deberta-v3-base-zeroshot-v2.0",
        onnx_available=True
    ),
    
    "bias_output": ModelInfo(
        scanner_type="bias_output",
        hf_repo="valurank/distilroberta-bias",  # Output-only model
        model_constant_import="llm_guard.output_scanners.bias.DEFAULT_MODEL",
        local_dir_name="distilroberta-bias",
        onnx_available=True
    ),
    
    "code_detection_output": ModelInfo(
        scanner_type="code_detection_output",
        hf_repo="philomath-1209/programming-language-identification",  # Same as input scanner
        model_constant_import="llm_guard.input_scanners.code.DEFAULT_MODEL",
        local_dir_name="programming-language-identification",
        onnx_available=True
    ),
    
    "factual_consistency_output": ModelInfo(
        scanner_type="factual_consistency_output",
        hf_repo="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",  # Uses same model as ban_topics
        model_constant_import="llm_guard.input_scanners.ban_topics.MODEL_DEBERTA_BASE_V2",
        local_dir_name="deberta-v3-base-zeroshot-v2.0",
        onnx_available=True
    ),
    
    "gibberish_output": ModelInfo(
        scanner_type="gibberish_output",
        hf_repo="madhurjindal/autonlp-Gibberish-Detector-492513457",  # Same as input scanner
        model_constant_import="llm_guard.input_scanners.gibberish.DEFAULT_MODEL",
        local_dir_name="autonlp-Gibberish-Detector-492513457",
        onnx_available=True
    ),
    
    "language_output": ModelInfo(
        scanner_type="language_output",
        hf_repo="papluca/xlm-roberta-base-language-detection",  # Same as input scanner
        model_constant_import="llm_guard.input_scanners.language.DEFAULT_MODEL",
        local_dir_name="xlm-roberta-base-language-detection",
        onnx_available=True
    ),
    
    "relevance_output": ModelInfo(
        scanner_type="relevance_output",
        hf_repo="BAAI/bge-base-en-v1.5",  # Output-only model
        model_constant_import="llm_guard.output_scanners.relevance.MODEL_EN_BGE_BASE",
        local_dir_name="bge-base-en-v1.5",
        onnx_available=True
    ),
    
    "toxicity_output": ModelInfo(
        scanner_type="toxicity_output",
        hf_repo="unitary/unbiased-toxic-roberta",  # Same as input scanner
        model_constant_import="llm_guard.input_scanners.toxicity.DEFAULT_MODEL",
        local_dir_name="unbiased-toxic-roberta",
        onnx_available=True
    ),
    
    "pii_redaction": ModelInfo(
        scanner_type="pii_redaction",
        hf_repo="Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
        model_constant_import="llm_guard.input_scanners.anonymize.DEFAULT_ENTITY_RECOGNIZER_MODEL",
        local_dir_name="deberta-v3-base_finetuned_ai4privacy_v2",
        onnx_available=True
    ),
    
    # Rule-based output scanners (no models required)
    "deanonymize_output": ModelInfo(
        scanner_type="deanonymize_output",
        hf_repo="",  # Rule-based
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "json_output": ModelInfo(
        scanner_type="json_output",
        hf_repo="",  # Rule-based JSON validation
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "malicious_urls_output": ModelInfo(
        scanner_type="malicious_urls_output",
        hf_repo="",  # Rule-based URL scanning
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "no_refusal_output": ModelInfo(
        scanner_type="no_refusal_output",
        hf_repo="",  # Rule-based pattern matching
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "output_length": ModelInfo(
        scanner_type="output_length",
        hf_repo="",  # Rule-based token counting
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "reading_time_output": ModelInfo(
        scanner_type="reading_time_output",
        hf_repo="",  # Rule-based calculation
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "regex_output": ModelInfo(
        scanner_type="regex_output",
        hf_repo="",  # Rule-based
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "sensitive_info_output": ModelInfo(
        scanner_type="sensitive_info_output",
        hf_repo="",  # Rule-based regex patterns
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
    
    "url_reachability_output": ModelInfo(
        scanner_type="url_reachability_output",
        hf_repo="",  # Rule-based HTTP checks
        model_constant_import="",
        local_dir_name="",
        requires_model=False
    ),
}


def get_model_info(scanner_type: str) -> Optional[ModelInfo]:
    """Get model information for a scanner type.
    
    Args:
        scanner_type: The scanner type (e.g., "prompt_injection")
        
    Returns:
        ModelInfo if found, None otherwise
    """
    return LLM_GUARD_MODELS.get(scanner_type)


def list_all_models() -> list[ModelInfo]:
    """Get list of all registered models.
    
    Returns:
        List of all ModelInfo objects
    """
    return list(LLM_GUARD_MODELS.values())


def list_models_requiring_download() -> list[ModelInfo]:
    """Get list of models that need to be downloaded.
    
    Returns:
        List of ModelInfo objects that require model downloads
    """
    return [info for info in LLM_GUARD_MODELS.values() if info.requires_model]
