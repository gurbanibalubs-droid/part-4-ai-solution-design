# Part 4: AI Solution Design for a Business Problem

## Overview

This repository contains the solution for **Part 4** of the assignment. As an AI Business Analyst, a complete AI solution has been designed for a real-world business problem in the **Healthcare** domain.

---

## Selected Domain & Problem

**Domain:** Healthcare  
**Problem:** Manual chest X-ray review by radiologists is slow (35–45 hrs resolution time), error-prone (~8% error rate), and cannot scale with demand.  
**Proposed Solution:** CNN-based chest X-ray triage system using ResNet-50 (transfer learning) to classify scans as Normal, Pneumonia, or Urgent — automatically prioritizing the radiologist's queue.

---

## Repository Structure

```
part-4-ai-solution-design/
│
├── README.md
├── solution_report.md          ← Full 8-task design report
└── diagrams/
    └── solution_architecture.html  ← Interactive solution architecture diagram
```

---

## Report Summary

| Task | Content |
|---|---|
| Task 1 | Domain: Healthcare |
| Task 2 | Problem: Manual X-ray triage — slow, inconsistent, unscalable |
| Task 3 | AI Task: Image Classification |
| Task 4 | Data: 50,000+ labelled X-rays, patient metadata, radiologist labels |
| Task 5 | Model: ResNet-50 transfer learning → Softmax(3 classes) |
| Task 6 | Evaluation: Recall >95% (Urgent), AUC >0.95, resolution time <20hrs |
| Task 7 | Risks: Bias, false negatives, privacy, over-reliance — all mitigated |
| Task 8 | One-page solution summary with full impact & risk plan |

---

## Reference Files Used

- `ai_usecase_reference_catalog.csv` — Domain and model selection reference
- `business_kpi_sample.csv` — Baseline KPIs for measuring business impact

---

## Dataset Source

[Google Drive – Part 4 Reference Files](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)

> Reference CSVs are not uploaded to this repository per assignment instructions.
