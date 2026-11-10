# clinical-screening-ai_transformer

**Transformer-based clinical trial eligibility screening with evidence-grounded criterion assessment.**

clinical-screening-ai_transformer is a research prototype for matching patient descriptions to individual clinical-trial inclusion and exclusion criteria. It is designed to predict criterion-level eligibility categories, identify supporting patient-record sentences, and make inference behavior measurable through experiments with KV caching, tiled attention, and speculative decoding.

> **Important:** This project is an educational/research prototype, not a medical device and not a clinical enrollment decision tool. Its outputs must not be used to make real patient-care or trial-enrollment decisions. All results require qualified human review.

## Project status

🚧 **Early development — implementation and evaluation are in progress.**

This README describes the intended project structure and workflow. A component should only be described as implemented after its code, tests, and experimental evidence are available.

## Objectives

- Fine-tune an open-weight, locally executable autoregressive Transformer on the prescribed TrialGPT Criterion Annotations dataset.
- Classify each inclusion or exclusion criterion while preserving the dataset's label categories, including insufficient-information and not-applicable outcomes.
- Link predictions to sentence-level evidence from the supplied patient description.
- Implement and validate cache-aware autoregressive decoding.
- Implement reference scaled dot-product attention and a tiled online-softmax variant.
- Implement greedy speculative decoding with a compatible draft/target model pair.
- Provide a browser-accessible interface for screening, inspecting evidence, comparing inference modes, and exporting a report.
- Evaluate the system using a fixed patient-grouped split and reproducible experiments.

## dataset and evaluation protocol

The required dataset is [TrialGPT Criterion Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations). The [NCBI TrialGPT repository](https://github.com/ncbi-nlp/TrialGPT) may be consulted as a reference; this project must contain its own implementation and verification.

following is the planned patient-grouped split:

1. Sort the unique patient IDs.
2. Shuffle them using `numpy.random.default_rng(42)`.
3. Assign the first 37 patients to training, the next 8 to validation, and the remaining 8 to testing.
4. Keep all annotations for a patient in exactly one split.
5. Save the split manifest and checks for patient leakage.
6. Do not use the held-out test set for model, prompt, or threshold tuning.

Do not use previous model-generated predictions or explanations from the dataset as input features. The split manifest should record patient IDs and counts as required by the assignment. Do not publish patient records or other sensitive data in the repository.

## Planned system components

| Component | Purpose |
|---|---|
| Data preparation | Load annotations, validate records, create the fixed patient-grouped split |
| Model adaptation | Fine-tune the selected open-weight Transformer |
| Eligibility prediction | Predict criterion-level labels for inclusion and exclusion criteria |
| Evidence pipeline | Return cited sentence IDs and corresponding source text |
| KV-cached decoding | Reuse and update key/value states during autoregressive generation |
| Tiled attention | Compare reference attention with a numerically validated online-softmax implementation |
| Speculative decoding | Draft candidate tokens and verify them with the target model |
| Web application | Accept patient records and trial criteria; display results and export reports |
| Evaluation | Calculate classification, evidence-retrieval, correctness, latency, throughput, and memory metrics |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-github-username>/clinical-screening-ai_transformer.git
cd clinical-screening-ai_transformer
```

### 2. Create an environment

Python version and dependencies will be pinned after the model and hardware setup are selected.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install dependencies once `requirements.txt` has been created:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Obtain the dataset

Download the required dataset from the [Hugging Face dataset page](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) and place local data files under `data/raw/`.

Keep downloaded data and patient-level records out of Git unless the dataset's terms explicitly permit redistribution and the assignment requires it. Never commit identifiable or confidential patient information.

### 4. Prepare data and splits

The data-preparation scripts will create the prescribed 37/8/8 patient-grouped split using seed `42`, save a split manifest, and verify that no patient appears in more than one split.

> Add the actual command here once the data-preparation entry point has been implemented.

### 5. Train and evaluate

Training, validation, and final test commands will be documented here after their entry points and configuration files are in place. The test set must remain untouched until final evaluation.

### 6. Run the application

The final application will provide a standalone browser-accessible workflow. Startup instructions and the exact command will be added once the application framework and entry point are implemented.

## Data, model, and artifact notes

- Dataset: [TrialGPT Criterion Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)
- Reference implementation: [NCBI TrialGPT](https://github.com/ncbi-nlp/TrialGPT)
- Base and draft model details: **To be selected and documented.**
- Fine-tuned weights / adapters: **To be published separately, with download instructions.**
- Final evaluation results: **Pending.**
- Demonstration video: **To be added.**

## Responsible use and limitations

- This system is for coursework and research only.
- It does not provide medical advice, diagnose conditions, or determine actual clinical-trial eligibility.
- Missing information must not be treated automatically as proof of eligibility or ineligibility.
- Predictions should be traceable to the patient-record sentences supplied as input.
- Do not upload real patient records or protected health information to public demos, issue trackers, or third-party services.
- Outputs may be incomplete, incorrect, or unsupported and must be reviewed by qualified people.

## Acknowledgements

This project uses the TrialGPT Criterion Annotations dataset and may use open-source model architectures and libraries. Specific papers, software, pretrained models, and AI-assisted contributions will be acknowledged in the technical report and project documentation.

