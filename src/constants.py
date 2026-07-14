"""Survey coding maps and column lists (2021 NCME&PR numeric codebook)."""

# Label dictionaries are used when raw values are strings; numeric codes are
# used as-is as ordinal ranks when the extract already stores integers.

EDUCATION_ORDER = {
    "Less than high school": 1,
    "High school graduate": 2,
    "Some college": 3,
    "Bachelor's degree": 4,
    "Graduate degree": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
}

INCOME_ORDER = {
    "Under $25,000": 1,
    "$25,000-$49,999": 2,
    "$50,000-$74,999": 3,
    "$75,000-$99,999": 4,
    "$100,000 or more": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
}

SELF_TREAT_ORDER = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Somewhat disagree": 3,
    "Neutral": 4,
    "Somewhat agree": 5,
    "Agree": 6,
    "Strongly agree": 7,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
}

MED_ITEMS = ["Med7", "Med8", "Med9"]
DTCA_ITEMS = ["DTCA_Info", "DTCA_Prescribe"]

# Info_Other may be absent in some extracts; preprocess skips missing columns.
INFO_SOURCES = [
    "Info_Google",
    "Info_App",
    "Info_Fam",
    "Info_MD",
    "Info_RPh",
    "Info_OtherProf",
    "Info_Web",
    "Info_SocMedia",
    "Info_Print",
    "Info_Other",
]
FREQ_ORDER = ["None", "One", "Two", "Three+"]

REQUIRED_COLUMNS = frozenset(
    {
        "respondent_id",
        "Education",
        "HouseIncome",
        "Self_Treat",
        "NumOTC",
        "NumHerbal",
        *MED_ITEMS,
        *DTCA_ITEMS,
    }
)

# Self_Treat intentionally excluded — modelled separately as self_treat_z.
HISB_COMPONENTS = [
    *MED_ITEMS,
    "dtca_info_bin",
    "dtca_prescribe_bin",
    "info_source_count",
]
