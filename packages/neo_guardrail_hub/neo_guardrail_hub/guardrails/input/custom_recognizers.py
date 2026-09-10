"""
Enhanced Presidio Recognizers for PII Detection
Adds custom regex patterns and recognizers to boost detection rate from 53% to 90%+
"""

from typing import List, Any


def get_enhanced_recognizers() -> List[Any]:
    """
    Create custom Presidio recognizers for entities missed by Deberta.
    
    Returns:
        List of PatternRecognizer objects
    
    Note:
        This function requires presidio_analyzer to be installed.
        It's only called when LLM Guard is initialized with Presidio support.
    """
    try:
        from presidio_analyzer import Pattern, PatternRecognizer
    except ImportError:
        # Return empty list if presidio not installed
        # This is fine - the recognizers are optional enhancements
        return []
    
    recognizers = []
    
    # ====================
    # US GOVERNMENT IDS
    # ====================
    
    # US Social Security Number (SSN) - Standalone
    ssn_patterns = [
        Pattern(name="ssn_dash", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.9),
        Pattern(name="ssn_space", regex=r"\b\d{3}\s\d{2}\s\d{4}\b", score=0.85),
        Pattern(name="ssn_no_separator", regex=r"\b\d{9}\b", score=0.6),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="US_SSN",
            name="US_SSN_Enhanced",
            patterns=ssn_patterns,
            context=["ssn", "social security", "ss#"],
        )
    )
    
    # ====================
    # FINANCIAL - CREDIT CARDS
    # ====================
    
    # Credit Card Number
    cc_patterns = [
        # Visa: 4xxx-xxxx-xxxx-xxxx
        Pattern(name="visa", regex=r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", score=0.9),
        # Mastercard: 5xxx-xxxx-xxxx-xxxx
        Pattern(name="mastercard", regex=r"\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", score=0.9),
        # Amex: 3xxx-xxxxxx-xxxxx
        Pattern(name="amex", regex=r"\b3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}\b", score=0.9),
        # Generic 16-digit
        Pattern(name="generic_16", regex=r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", score=0.7),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="CREDIT_CARD",
            name="CreditCard_Enhanced",
            patterns=cc_patterns,
            context=["card", "credit", "visa", "mastercard", "amex"],
        )
    )
    
    # CVV/CVC Code
    cvv_patterns = [
        Pattern(name="cvv_3digit", regex=r"\b\d{3}\b", score=0.4),
        Pattern(name="cvv_4digit", regex=r"\b\d{4}\b", score=0.4),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="CREDITCARDCVV",
            name="CVV_Enhanced",
            patterns=cvv_patterns,
            context=["cvv", "cvc", "security code", "card verification"],
        )
    )
    
    # Account Numbers
    account_patterns = [
        Pattern(name="account_10_12digit", regex=r"\b\d{10,12}\b", score=0.5),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="ACCOUNTNUMBER",
            name="Account_Enhanced",
            patterns=account_patterns,
            context=["account", "acct", "account number", "account #"],
        )
    )
    
    # BIC/SWIFT Code
    bic_patterns = [
        Pattern(name="bic", regex=r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b", score=0.8),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="BIC",
            name="BIC_Enhanced",
            patterns=bic_patterns,
            context=["bic", "swift", "swift code"],
        )
    )
    
    # ====================
    # UK GOVERNMENT IDS
    # ====================
    
    # UK NHS Number
    nhs_patterns = [
        Pattern(name="nhs_spaced", regex=r"\b\d{3}\s\d{3}\s\d{4}\b", score=0.9),
        Pattern(name="nhs_dashed", regex=r"\b\d{3}-\d{3}-\d{4}\b", score=0.9),
        Pattern(name="nhs_no_separator", regex=r"\b\d{10}\b", score=0.5),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="UK_NHS",
            name="UK_NHS_Enhanced",
            patterns=nhs_patterns,
            context=["nhs", "nhs number", "health service"],
        )
    )
    
    # ====================
    # EUROPEAN IDS
    # ====================
    
    # Spanish NIF/NIE
    nif_patterns = [
        Pattern(name="nif", regex=r"\b\d{8}[A-Z]\b", score=0.8),
        Pattern(name="nie", regex=r"\b[XYZ]\d{7}[A-Z]\b", score=0.8),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="ES_NIF",
            name="ES_NIF_Enhanced",
            patterns=nif_patterns,
            context=["nif", "nie", "spanish", "tax id"],
        )
    )
    
    # Italian Fiscal Code
    fiscal_patterns = [
        Pattern(name="fiscal_code", regex=r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b", score=0.9),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="IT_FISCAL_CODE",
            name="IT_Fiscal_Enhanced",
            patterns=fiscal_patterns,
            context=["fiscal", "codice fiscale", "italian"],
        )
    )
    
    # Polish PESEL
    pesel_patterns = [
        Pattern(name="pesel", regex=r"\b\d{11}\b", score=0.6),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="PL_PESEL",
            name="PL_PESEL_Enhanced",
            patterns=pesel_patterns,
            context=["pesel", "polish", "poland"],
        )
    )
    
    # ====================
    # ASIAN IDS
    # ====================
    
    # Indian Aadhaar
    aadhaar_patterns = [
        Pattern(name="aadhaar_spaced", regex=r"\b\d{4}\s\d{4}\s\d{4}\b", score=0.9),
        Pattern(name="aadhaar_no_space", regex=r"\b\d{12}\b", score=0.5),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="IN_AADHAAR",
            name="IN_Aadhaar_Enhanced",
            patterns=aadhaar_patterns,
            context=["aadhaar", "aadhar", "uid"],
        )
    )
    
    # Korean RRN
    rrn_patterns = [
        Pattern(name="rrn", regex=r"\b\d{6}-\d{7}\b", score=0.9),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="KR_RRN",
            name="KR_RRN_Enhanced",
            patterns=rrn_patterns,
            context=["rrn", "korean", "resident registration"],
        )
    )
    
    # Thai National ID
    thai_patterns = [
        Pattern(name="thai_id", regex=r"\b\d{13}\b", score=0.5),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="TH_TNIN",
            name="TH_ID_Enhanced",
            patterns=thai_patterns,
            context=["thai", "thailand", "national id"],
        )
    )
    
    # ====================
    # AUSTRALIAN IDS
    # ====================
    
    # ABN (Australian Business Number)
    abn_patterns = [
        Pattern(name="abn", regex=r"\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b", score=0.9),
        Pattern(name="abn_no_space", regex=r"\b\d{11}\b", score=0.5),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="AU_ABN",
            name="AU_ABN_Enhanced",
            patterns=abn_patterns,
            context=["abn", "australian business number"],
        )
    )
    
    # Medicare Number
    medicare_patterns = [
        Pattern(name="medicare", regex=r"\b\d{4}\s\d{5}\s\d{1}\b", score=0.9),
        Pattern(name="medicare_no_space", regex=r"\b\d{10}\b", score=0.4),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="AU_MEDICARE",
            name="AU_Medicare_Enhanced",
            patterns=medicare_patterns,
            context=["medicare", "australian", "health"],
        )
    )
    
    # ====================
    # SECURITY
    # ====================
    
    # PIN Codes
    pin_patterns = [
        Pattern(name="pin_4digit", regex=r"\b\d{4}\b", score=0.3),
        Pattern(name="pin_6digit", regex=r"\b\d{6}\b", score=0.3),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="PIN",
            name="PIN_Enhanced",
            patterns=pin_patterns,
            context=["pin", "pin code", "personal identification"],
        )
    )
    
    # Password (common patterns)
    password_patterns = [
        # Contains special chars and mixed case
        Pattern(name="password_complex", regex=r"[A-Za-z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]{8,}", score=0.4),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="PASSWORD",
            name="Password_Enhanced",
            patterns=password_patterns,
            context=["password", "passwd", "pass", "pwd"],
        )
    )
    
    # ====================
    # VEHICLE
    # ====================
    
    # Vehicle VIN (17 characters, no I, O, Q)
    vin_patterns = [
        Pattern(name="vin", regex=r"\b[A-HJ-NPR-Z0-9]{17}\b", score=0.8),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="VEHICLEVIN",
            name="VIN_Enhanced",
            patterns=vin_patterns,
            context=["vin", "vehicle identification", "chassis"],
        )
    )
    
    # UK Vehicle Registration
    vrm_patterns = [
        Pattern(name="uk_vrm", regex=r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b", score=0.7),
        Pattern(name="uk_vrm_old", regex=r"\b[A-Z]{3}\s?\d{3}[A-Z]\b", score=0.7),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="VEHICLEVRM",
            name="VRM_Enhanced",
            patterns=vrm_patterns,
            context=["registration", "plate", "vehicle"],
        )
    )
    
    # ====================
    # DATES
    # ====================
    
    # Date of Birth
    dob_patterns = [
        # MM/DD/YYYY
        Pattern(name="dob_slash", regex=r"\b\d{1,2}/\d{1,2}/\d{4}\b", score=0.5),
        # DD-MM-YYYY
        Pattern(name="dob_dash", regex=r"\b\d{1,2}-\d{1,2}-\d{4}\b", score=0.5),
        # Month DD, YYYY
        Pattern(name="dob_long", regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", score=0.6),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            name="DOB_Enhanced",
            patterns=dob_patterns,
            context=["birth", "dob", "date of birth", "born"],
        )
    )
    
    # ====================
    # MISCELLANEOUS
    # ====================
    
    # IMEI (15 digits)
    imei_patterns = [
        Pattern(name="imei", regex=r"\b\d{15}\b", score=0.6),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="PHONEIMEI",
            name="IMEI_Enhanced",
            patterns=imei_patterns,
            context=["imei", "phone", "mobile"],
        )
    )
    
    # Medical License
    medical_patterns = [
        Pattern(name="medical_license", regex=r"\bM[A-Z]\d{6,8}\b", score=0.7),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="MEDICAL_LICENSE",
            name="Medical_License_Enhanced",
            patterns=medical_patterns,
            context=["medical", "license", "physician", "doctor"],
        )
    )
    
    return recognizers
