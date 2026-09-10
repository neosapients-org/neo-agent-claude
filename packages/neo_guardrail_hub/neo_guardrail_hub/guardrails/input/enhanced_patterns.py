"""
Enhanced Regex Patterns for PII Detection - Correct LLM Guard Format
Uses LLM Guard's regex pattern format: {name, expressions, context, score, languages}
"""

from typing import List, Dict, Any


def get_enhanced_regex_patterns() -> List[Dict[str, Any]]:
    """
    Create custom regex patterns for entities missed by Deberta.
    Format: {name, expressions, context, score, languages}
    
    Returns:
        List of regex pattern configurations
    """
    patterns = []
    
    # US SSN
    patterns.append({
        "name": "US_SSN",
        "expressions": [r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"],
        "context": ["ssn", "social security"],
        "score": 0.9,
        "languages": ["en"],
    })
    
    # Credit Card
    patterns.append({
        "name": "CREDIT_CARD",
        "expressions": [r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"],
        "context": ["card", "credit", "visa"],
        "score": 0.85,
        "languages": ["en"],
    })
    
    # CVV
    patterns.append({
        "name": "CREDITCARDCVV", 
        "expressions": [r"(?i)(?:cvv|cvc)[\s:]+(\d{3,4})"],
        "context": ["cvv", "cvc"],
        "score": 0.8,
        "languages": ["en"],
    })
    
    # Account Number
    patterns.append({
        "name": "ACCOUNTNUMBER",
        "expressions": [r"(?i)(?:account|acct)[\s#:]+(\d{8,17})"],
        "context": ["account", "acct"],
        "score": 0.7,
        "languages": ["en"],
    })
    
    # BIC
    patterns.append({
        "name": "BIC",
        "expressions": [r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"],
        "context": ["bic", "swift"],
        "score": 0.8,
        "languages": ["en"],
    })
    
    # UK NHS
    patterns.append({
        "name": "UK_NHS",
        "expressions": [r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b"],
        "context": ["nhs", "health"],
        "score": 0.85,
        "languages": ["en"],
    })
    
    # Spanish IDs
    patterns.append({
        "name": "ES_NIF",
        "expressions": [r"\b\d{8}[A-Z]\b"],
        "context": ["nif"],
        "score": 0.8,
        "languages": ["en", "es"],
    })
    
    patterns.append({
        "name": "ES_NIE",
        "expressions": [r"\b[XYZ]\d{7}[A-Z]\b"],
        "context": ["nie"],
        "score": 0.8,
        "languages": ["en", "es"],
    })
    
    # Italian IDs
    patterns.append({
        "name": "IT_FISCAL_CODE",
        "expressions": [r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"],
        "context": ["fiscal", "codice"],
        "score": 0.9,
        "languages": ["en", "it"],
    })
    
    # Polish PESEL
    patterns.append({
        "name": "PL_PESEL",
        "expressions": [r"\b\d{11}\b"],
        "context": ["pesel"],
        "score": 0.6,
        "languages": ["en", "pl"],
    })
    
    # Finnish ID
    patterns.append({
        "name": "FI_PERSONAL_IDENTITY_CODE",
        "expressions": [r"\b\d{6}[\-+A]\d{3}[0-9A-Z]\b"],
        "context": ["finnish", "finland"],
        "score": 0.85,
        "languages": ["en", "fi"],
    })
    
    # Indian Aadhaar
    patterns.append({
        "name": "IN_AADHAAR",
        "expressions": [r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"],
        "context": ["aadhaar", "aadhar"],
        "score": 0.85,
        "languages": ["en"],
    })
    
    # Indian GSTIN
    patterns.append({
        "name": "IN_GSTIN",
        "expressions": [r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b"],
        "context": ["gstin", "gst"],
        "score": 0.9,
        "languages": ["en"],
    })
    
    # Korean RRN
    patterns.append({
        "name": "KR_RRN",
        "expressions": [r"\b\d{6}-\d{7}\b"],
        "context": ["rrn", "korean"],
        "score": 0.9,
        "languages": ["en", "ko"],
    })
    
    # Thai ID
    patterns.append({
        "name": "TH_TNIN",
        "expressions": [r"\b\d{1}-\d{4}-\d{5}-\d{2}-\d\b"],
        "context": ["thai", "thailand"],
        "score": 0.8,
        "languages": ["en", "th"],
    })
    
    # Singapore UEN
    patterns.append({
        "name": "SG_UEN",
        "expressions": [r"\b\d{9}[A-Z]\b", r"\b\d{10}[A-Z]\b"],
        "context": ["uen", "singapore"],
        "score": 0.7,
        "languages": ["en"],
    })
    
    # Australian IDs
    patterns.append({
        "name": "AU_ABN",
        "expressions": [r"\b\d{2}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\b"],
        "context": ["abn", "australian"],
        "score": 0.85,
        "languages": ["en"],
    })
    
    patterns.append({
        "name": "AU_ACN",
        "expressions": [r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\b"],
        "context": ["acn"],
        "score": 0.8,
        "languages": ["en"],
    })
    
    patterns.append({
        "name": "AU_MEDICARE",
        "expressions": [r"\b\d{4}[\s\-]?\d{5}[\s\-]?\d\b"],
        "context": ["medicare"],
        "score": 0.85,
        "languages": ["en"],
    })
    
    # Security
    patterns.append({
        "name": "PIN",
        "expressions": [r"(?i)(?:pin|pin\s+code)[\s:]+(\d{4,6})"],
        "context": ["pin"],
        "score": 0.7,
        "languages": ["en"],
    })
    
    patterns.append({
        "name": "PASSWORD",
        "expressions": [r"(?i)(?:password|passwd|pwd)[\s:]+([^\s]{8,})"],
        "context": ["password", "passwd"],
        "score": 0.6,
        "languages": ["en"],
    })
    
    # Vehicle
    patterns.append({
        "name": "VEHICLEVIN",
        "expressions": [r"\b[A-HJ-NPR-Z0-9]{17}\b"],
        "context": ["vin", "vehicle"],
        "score": 0.8,
        "languages": ["en"],
    })
    
    patterns.append({
        "name": "VEHICLEVRM",
        "expressions": [r"\b[A-Z]{2}\d{2}[\s]?[A-Z]{3}\b"],
        "context": ["registration", "plate"],
        "score": 0.7,
        "languages": ["en"],
    })
    
    # Date of Birth
    patterns.append({
        "name": "DATE_OF_BIRTH",
        "expressions": [
            r"(?i)(?:born|birth|dob)[\s:]+\d{1,2}[-/]\d{1,2}[-/]\d{4}",
            r"(?i)(?:born|birth|dob)[\s:]+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        ],
        "context": ["born", "birth", "dob"],
        "score": 0.7,
        "languages": ["en"],
    })
    
    # IMEI
    patterns.append({
        "name": "PHONEIMEI",
        "expressions": [r"\b\d{15}\b"],
        "context": ["imei", "phone"],
        "score": 0.6,
        "languages": ["en"],
    })
    
    # Medical License
    patterns.append({
        "name": "MEDICAL_LICENSE",
        "expressions": [r"\bM[A-Z]\d{6,8}\b"],
        "context": ["medical", "license"],
        "score": 0.7,
        "languages": ["en"],
    })
    
    # Gender
    patterns.append({
        "name": "GENDER",
        "expressions": [r"(?i)(?:gender|sex)[\s:]+(\bmale\b|\bfemale\b|\bm\b|\bf\b)"],
        "context": ["gender", "sex"],
        "score": 0.6,
        "languages": ["en"],
    })
    
    # Eye Color
    patterns.append({
        "name": "EYECOLOR",
        "expressions": [r"(?i)(?:eye\s+color)[\s:]+(\w+)"],
        "context": ["eye color"],
        "score": 0.6,
        "languages": ["en"],
    })
    
    return patterns
