"""Survey coding maps and column lists (2021 NCME codebook)."""

EDUCATION_ORDER = {
    "Less than high school": 0,
    "High school graduate": 1,
    "Some college": 2,
    "Bachelor's degree": 3,
    "Graduate degree": 4,
}

INCOME_ORDER = {
    "Under $25,000": 0,
    "$25,000-$49,999": 1,
    "$50,000-$74,999": 2,
    "$75,000-$99,999": 3,
    "$100,000 or more": 4,
}

SELF_TREAT_ORDER = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

MED_ITEMS = ["Med7", "Med8", "Med9"]
DTCA_ITEMS = ["DTCA_Info", "DTCA_Prescribe"]
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

# Self_Treat intentionally excluded - modelled separately as Self_Treat_z.
HISB_COMPONENTS = [
    "Med7",
    "Med8",
    "Med9",
    "dtca_info_bin",
    "dtca_prescribe_bin",
    "info_source_count",
]

FREQ_ORDER = ["None", "One", "Two", "Three+"]

REQUIRED_COLUMNS = {
    "respondent_id",
    "Education",
    "HouseIncome",
    *MED_ITEMS,
    *DTCA_ITEMS,
    "Self_Treat",
    *INFO_SOURCES,
    "NumOTC",
    "NumHerbal",
}
