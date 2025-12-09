# ============================================================================================================
# PROJECT      : STEA – Science, Technology & Energy Analysis
# PIPELINE     : MSTI – Main Science and Technology Indicators
# MODULE       : pipelines/utils/msti_variables_mapping.py
# PURPOSE      : Dictionaries and mappings for MSTI variables and geographical entities
# DESCRIPTION  :
#   - Country to continent mapping for geographical aggregation.
#   - Descriptive dictionaries for OECD MSTI indicators.
#   - Thematic organization of variables based on OECD logic:
#       Financial Inputs, Human Inputs, Results, Macroeconomy, Sectoral.
# ============================================================================================================

# ------------------------------------------------------------------------------------------------------------
# 1. Country → Continent Mapping
# ------------------------------------------------------------------------------------------------------------
COUNTRY_TO_CONTINENT = {
    "Afrique du Sud": "Africa",
    "Allemagne": "Europe", "Autriche": "Europe", "Belgique": "Europe", "Bulgarie": "Europe", "Croatie": "Europe",
    "Danemark": "Europe", "Espagne": "Europe", "Estonie": "Europe", "Finlande": "Europe", "France": "Europe",
    "Grèce": "Europe", "Hongrie": "Europe", "Irlande": "Europe", "Islande": "Europe", "Italie": "Europe",
    "Lettonie": "Europe", "Lituanie": "Europe", "Luxembourg": "Europe", "OCDE": "OECD", "Pays-Bas": "Europe",
    "Pologne": "Europe", "Portugal": "Europe", "Roumanie": "Europe", "Slovénie": "Europe", "Suisse": "Europe",
    "Suède": "Europe", "Tchéquie": "Europe", "Argentine": "America", "Australie": "Oceania", "Canada": "America",
    "Chili": "America", "Chine (République populaire de)": "Asia", "Colombie": "America", "Corée": "Asia",
    "Costa Rica": "America", "Israël": "Asia", "Japon": "Asia", "Mexique": "America", "Norvège": "Europe",
    "Nouvelle-Zélande": "Oceania", "Royaume-Uni": "Europe", "Russie": "Asia", "République slovaque": "Europe",
    "Singapour": "Asia", "Taipei chinois": "Asia", "Türkiye": "Asia", "États-Unis": "America",
    "Union européenne (27 pays à partir du 01/02/2020)": "EU",
}   # geographical mapping from OECD reference zones to continental aggregates


# ------------------------------------------------------------------------------------------------------------
# 2. OECD MSTI Variable Descriptions
# ------------------------------------------------------------------------------------------------------------
VARIABLE_DESCRIPTIONS = {
    # --- R&D Inputs: Gross Domestic Expenditure on R&D (GERD, BERD, HERD, GOVERD) ---
    "DIRD_%pib": "Gross domestic expenditure on R&D (% of GDP)",
    "DIRD_$": "Gross domestic expenditure on R&D (USD, PPP)",
    "DIRD_$pers": "Gross domestic expenditure on R&D per capita (USD, PPP)",
    "DIRDciv_%pib": "Civil R&D (% of GDP)",
    "DIRDgov_%pib": "Government-funded R&D (% of GDP)",
    "DIRDgov_%dird": "Government share of total R&D (% of GERD)",
    "DIRDgov_$": "Government-funded R&D (USD, PPP)",
    "DIRDentr_%pib": "Business-funded R&D (% of GDP)",
    "DIRDentr_%dird": "Business share of total R&D (% of GERD)",
    "DIRDentr_$": "Business-funded R&D (USD, PPP)",
    "DIRDrm_%pib": "Foreign-funded R&D (% of GDP)",
    "DIRDrm_%dird": "Rest-of-world share of total R&D (% of GERD)",
    "DIRDrm_$": "Foreign-funded R&D (USD, PPP)",
    "DIRDensup_%pib": "Higher education and NPIs (% of GDP)",
    "DIRDensup_%dird": "Higher education and NPIs share of total R&D (% of GERD)",
    "DIRDensup_$": "Higher education and NPIs (USD, PPP)",
    "DIRDISBL_%dird": "Non-profit institutions R&D (% of GERD)",
    "Rechfond_%pib": "Basic research expenditures (% of GDP)",
    "RDcher_etp": "Researchers (FTE)",
    "RDcher_1kemp": "Researchers per 1,000 employees",
    "RDpers_etp": "R&D personnel (FTE)",
    "RDpers_1kemp": "R&D personnel per 1,000 employees",
    "RDcherfme_%cher": "Female researchers (% of total)",
    "RDcherfme_nb": "Female researchers (headcount)",

    # --- BERD: Business Enterprise R&D ---
    "DIRDE_%pib": "Business enterprise R&D (% of GDP)",
    "DIRDE_%dird": "Business share of total R&D (% of GERD)",
    "DIRDEgov_%dirde": "Government share of BERD (% of BERD)",
    "DIRDEentr_%dirde": "Self-financed business R&D (% of BERD)",
    "DIRDEensup_%dirde": "Higher education share (% of BERD)",
    "DIRDErm_%dirde": "Foreign-financed R&D (% of BERD)",
    "DIRDEmanuf_%dirde": "Manufacturing R&D share (% of BERD)",
    "DIRDEserv_%dirde": "Services R&D share (% of BERD)",
    "DIRDEtic_%dirde": "ICT R&D share (% of BERD)",
    "DIRDEpharma_%dirde": "Pharmaceutical R&D share (% of BERD)",
    "DIRDEaero_%dirde": "Aerospace R&D share (% of BERD)",
    "DIRDEelec_%dirde": "Electronics R&D share (% of BERD)",
    "DIRDE_$": "Business enterprise R&D (USD, PPP)",
    "DIRDEentr_$": "Self-financed business R&D (USD, PPP)",
    "DIRDEcher_%cher": "Researchers in business (% of total)",
    "DIRDEcher_etp": "Researchers in business (FTE)",
    "DIRDEcher_nb": "Researchers in business (headcount)",
    "DIRDEcherfme_%cher": "Female researchers in business (% of total)",
    "DIRDEcherfme_nb": "Female researchers in business (headcount)",
    "DIRDEpers_%pers": "Business R&D personnel (% of total)",
    "DIRDEper_etp": "Business R&D personnel (FTE)",

    # --- HERD: Higher Education R&D ---
    "DIRDES_%pib": "Higher education R&D (% of GDP)",
    "DIRDES_%dird": "Higher education share of total R&D (% of GERD)",
    "DIRDES_$": "Higher education R&D (USD, PPP)",
    "DIRDEScher_%cher": "Researchers in higher education (% of total)",
    "DIRDEScher_nb": "Researchers in higher education (headcount)",
    "DIRDEScher_etp": "Researchers in higher education (FTE)",
    "DIRDEScherfme_%cher": "Female researchers in higher education (% of total)",
    "DIRDEScherfme_nb": "Female researchers in higher education (headcount)",
    "DIRDESentr_%dirdes": "Business share in higher education R&D (% of HERD)",
    "DIRDESpers_%pers": "Higher education R&D personnel (% of total)",
    "DIRDESper_etp": "Higher education R&D personnel (FTE)",

    # --- GOVERD: Government R&D ---
    "DIRDET_%pib": "Government R&D (% of GDP)",
    "DIRDET_%dird": "Government R&D share of total R&D (% of GERD)",
    "DIRDET_$": "Government R&D (USD, PPP)",
    "DIRDETcher_%cher": "Government researchers (% of total)",
    "DIRDETcher_nb": "Government researchers (headcount)",
    "DIRDETcher_etp": "Government researchers (FTE)",
    "DIRDETcherfme_%cher": "Female government researchers (% of total)",
    "DIRDETcherfme_nb": "Female government researchers (headcount)",
    "DIRDETentr_%dirdet": "Private share in government R&D (% of GOVERD)",
    "DIRDETpers_%pers": "Government R&D personnel (% of total)",
    "DIRDETper_etp": "Government R&D personnel (FTE)",

    # --- GBARD: Government Budget Appropriations or Outlays for R&D ---
    "CBPRD_$": "Public budget appropriations for R&D (USD, PPP)",
    "CBPRDciv_%cbprd": "Civil credits (% of total GBARD)",
    "CBPRDdef_%cbprd": "Defense credits (% of total GBARD)",
    "CBPRDdeveco_%cbprdciv": "Economic development R&D (% of civil GBARD)",
    "CBPRDenssoc_%cbprdciv": "Education and social R&D (% of civil GBARD)",
    "CBPRDspace_%cbprdciv": "Space R&D (% of civil GBARD)",
    "CBPRDsantenv_%cbprdciv": "Health & environment R&D (% of civil GBARD)",
    "CBPRDrechnonor_%cbprdciv": "Non-oriented research (% of civil GBARD)",
    "CBPRDuniv_%cbprdciv": "University R&D (% of civil GBARD)",

    # --- Results: Patents, Innovation, Exports ---
    "Brevtech_nb": "Technological patents (count)",
    "Brevtriad_nb": "Triadic patents (count)",
    "Brevtriad_%brev": "Share of triadic patents (% total)",
    "Brevtic_nb": "ICT patents (count)",
    "Brevpct_nb": "Pharmaceutical patents (count)",
    "Exportaero_%export": "Aerospace exports (% total)",
    "Exportelec_%export": "Electronics exports (% total)",
    "Exportpharma_%export": "Pharmaceutical exports (% total)",

    # --- Macroeconomic and Structural Variables ---
    "Pop_nb": "Total population (thousands)",
    "Empltot_nb": "Total employment (thousands)",
    "PPA_$": "Purchasing power parity (USD)",
    "PIB_ind": "GDP price index (base 100)",
    "Cher_nb": "Researchers (headcount)",
}   # comprehensive human-readable descriptions for all MSTI indicator variables


# ------------------------------------------------------------------------------------------------------------
# 3. MSTI Variable Organization (OECD Analytical Blocks)
# ------------------------------------------------------------------------------------------------------------
VARIABLE_BLOCKS = {
    # --- R&D Inputs: Expenditure and Gross Domestic Expenditure on R&D (GERD) ---
    "INPUT_EXPENDITURE_R&D": [
        "DIRD_$", "DIRD_$pers", "DIRD_%pib", "DIRDciv_%pib", "Rechfond_%pib",
        "DIRDgov_$", "DIRDgov_%dird", "DIRDgov_%pib",
        "DIRDentr_$", "DIRDentr_%dird", "DIRDentr_%pib",
        "DIRDensup_$", "DIRDensup_%dird", "DIRDensup_%pib",
        "DIRDISBL_%dird",
        "DIRDrm_$", "DIRDrm_%dird", "DIRDrm_%pib",
        "DIRDE_$", "DIRDE_%dird", "DIRDE_%pib",
        "DIRDES_$", "DIRDES_%dird", "DIRDES_%pib",
        "DIRDET_$", "DIRDET_%dird", "DIRDET_%pib",
    ],

    # --- R&D Inputs: Government Budget (GBARD) ---
    "INPUT_GOV_BUDGET_R&D": [
        "CBPRD_$", "CBPRDciv_%cbprd", "CBPRDdef_%cbprd",
        "CBPRDuniv_%cbprdciv", "CBPRDenssoc_%cbprdciv", "CBPRDdeveco_%cbprdciv",
        "CBPRDrechnonor_%cbprdciv", "CBPRDsantenv_%cbprdciv", "CBPRDspace_%cbprdciv",
    ],

    # --- R&D Inputs: Human Resources (Personnel and Researchers) ---
    "INPUT_HUMAN_R&D": [
        "DIRDEScher_nb", "DIRDEScher_%cher", "DIRDEScher_etp",
        "DIRDEScherfme_nb", "DIRDEScherfme_%cher",
        "DIRDESper_etp", "DIRDESpers_%pers",
        "DIRDETcher_nb", "DIRDETcher_%cher", "DIRDETcher_etp",
        "DIRDETcherfme_nb", "DIRDETcherfme_%cher",
        "DIRDETper_etp", "DIRDETpers_%pers",
        "DIRDEcher_nb", "DIRDEcher_%cher", "DIRDEcher_etp",
        "DIRDEcherfme_nb", "DIRDEcherfme_%cher",
        "DIRDEper_etp", "DIRDEpers_%pers", "Cher_nb",
        "RDcher_etp", "RDcher_1kemp",
        "RDcherfme_nb", "RDcherfme_%cher",
        "RDpers_etp", "RDpers_1kemp",
    ],

    # --- R&D Inputs: Sectoral R&D Distribution ---
    "INPUT_SECTORAL_R&D": [
        "DIRDEmanuf_%dirde", "DIRDEaero_%dirde", "DIRDEpharma_%dirde",
        "DIRDEelec_%dirde", "DIRDEtic_%dirde", "DIRDEserv_%dirde",
        "DIRDErm_%dirde", "DIRDEensup_%dirde", "DIRDEgov_%dirde",
        "DIRDEentr_$", "DIRDEentr_%dirde",
        "DIRDESentr_%dirdes", "DIRDETentr_%dirdet",
    ],

    # --- Outputs: Innovation and Patents ---
    "OUTPUT_PATENTS": [
        "Brevtech_nb", "Brevtic_nb", "Brevpct_nb",
        "Brevtriad_nb", "Brevtriad_%brev",
    ],

    # --- Outputs: Trade and Exports ---
    "OUTPUT_TRADE_EXPORTS": [
        "Exportaero_%export", "Exportelec_%export", "Exportpharma_%export",
    ],

    # --- Macroeconomic: GDP and Price Index ---
    "MACRO_GDP": [
        "PIB_ind", "PPA_$",
    ],

    # --- Macroeconomic: Employment and Population ---
    "MACRO_EMPLOYMENT": [
        "Pop_nb", "Empltot_nb",
    ],
}   # thematic organization of MSTI variables following OECD analytical framework
