from typing import Dict, List

CCY_TO_COUNTRY: Dict[str, str] = {
    'EUR': 'EUR',   # Eurozone
    'GBP': 'GBR',   # United Kingdom
    'JPY': 'JPN',   # Japan
    'CHF': 'CHE',   # Switzerland
    'AUD': 'AUS',   # Australia
    'NZD': 'NZL',   # New Zealand
    'CAD': 'CAN',   # Canada
    'NOK': 'NOR',   # Norway
    'SEK': 'SWE',   # Sweden
    'TRY': 'TUR',   # Turkey
    'ZAR': 'ZAF',   # South Africa
    'BRL': 'BRA',   # Brazil
    'INR': 'IND',   # India
    'MXN': 'MEX',   # Mexico
    # 'IDR': 'IDN',   # Indonesia
    'PHP': 'PHL',   # Philippines
    'THB': 'THA',   # Thailand
    'PLN': 'POL',   # Poland
    'HUF': 'HUN',   # Hungary
    'CLP': 'CHL',   # Chile
    'COP': 'COL',   # Colombia
    'PEN': 'PER',   # Peru
    # 'EGP': 'EGY',   # Egypt
    # 'NGN': 'NGA',   # Nigeria
    }

COUNTRY_TO_CCY = {v: k for k, v in CCY_TO_COUNTRY.items()}

CCYS: List[str] = list(CCY_TO_COUNTRY.keys())
COUNTRIES: List[str] = list(CCY_TO_COUNTRY.values())