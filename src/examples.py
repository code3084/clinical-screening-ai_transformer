"""Two SYNTHETIC demo cases (invented for this prototype; not from the TrialGPT dataset)."""

EXAMPLES = {
    "Type 2 diabetes trial": dict(
        note=("62-year-old man with type 2 diabetes for 12 years, on metformin and insulin glargine. "
              "HbA1c was 8.4% last month. He has hypertension controlled on lisinopril. "
              "Two years ago he had a myocardial infarction treated with a coronary stent. "
              "He is a former smoker and quit 5 years ago. Most recent eGFR was 48 mL/min/1.73m2. "
              "No history of cancer. He denies current alcohol use."),
        criteria=("Inclusion Criteria:\n"
                  "- Age between 18 and 75 years\n"
                  "- Diagnosis of type 2 diabetes for at least 1 year\n"
                  "- HbA1c between 7.0% and 10.0%\n"
                  "- Body mass index of 30 kg/m2 or higher\n\n"
                  "Exclusion Criteria:\n"
                  "- Myocardial infarction within the past 6 months\n"
                  "- eGFR below 30 mL/min/1.73m2\n"
                  "- Pregnant or breastfeeding\n"
                  "- Current use of an insulin pump")),
    "Lung cancer trial": dict(
        note=("A 54-year-old woman presents with a persistent cough and 6 kg weight loss over 3 months. "
              "CT shows a 3.2 cm mass in the right upper lobe. "
              "Biopsy confirms non-small cell lung adenocarcinoma. "
              "EGFR mutation testing is pending. ECOG performance status is 1. "
              "She has received no prior chemotherapy. Brain MRI shows no metastases. "
              "She reports an allergy to penicillin."),
        criteria=("Inclusion Criteria:\n"
                  "- Histologically confirmed non-small cell lung cancer\n"
                  "- ECOG performance status of 0 or 1\n"
                  "- Documented EGFR activating mutation\n\n"
                  "Exclusion Criteria:\n"
                  "- Prior systemic chemotherapy for advanced disease\n"
                  "- Symptomatic brain metastases\n"
                  "- Known hypersensitivity to the study drug")),
}
