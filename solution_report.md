# AI Solution Design Report
## Automated Medical Image Triage Using CNN

**Domain:** Healthcare  
**AI Task Type:** Image Classification  
**Prepared by:** AI Business Analyst  
**Dataset References:** `ai_usecase_reference_catalog.csv`, `business_kpi_sample.csv`

---

## Task 1: Business Domain

**Selected Domain: Healthcare**

Healthcare was selected because it offers high-impact, life-critical use cases where AI can directly improve patient outcomes. Medical imaging is one of the most data-rich and well-researched areas of AI application, making it ideal for a CNN-based solution. The potential to reduce diagnostic delays and assist overloaded radiologists makes this domain both technically feasible and socially meaningful.

---

## Task 2: Business Problem Definition

### What Problem Is Being Solved?
Hospitals and diagnostic centres receive hundreds to thousands of medical scans (X-rays, CT scans) every day. Radiologists must manually review each scan to detect conditions such as pneumonia, tuberculosis, or COVID-19. This manual review process is slow, inconsistent, and bottlenecked — particularly in high-volume or under-resourced settings.

**The proposed solution:** An AI-powered **chest X-ray triage system** that automatically classifies incoming scans into priority categories — Normal, Pneumonia, or Urgent — so that radiologists can focus on the most critical cases first.

### Who Are the Users and Stakeholders?

| Stakeholder | Role |
|---|---|
| Radiologists | Primary users — receive AI-assisted triage suggestions |
| Hospital Administrators | Monitor efficiency gains and throughput |
| Patients | Benefit from faster diagnosis and treatment |
| IT / ML Engineering Teams | Deploy and maintain the system |
| Regulatory Bodies | Ensure compliance with medical AI standards |

### Current Manual Process
1. Scans are uploaded to the hospital's PACS (Picture Archiving System)
2. A radiologist manually opens and reviews each scan in the queue
3. Written reports are generated and sent to the referring physician
4. Urgent cases are identified only after manual review — delays are common

### Limitations of the Current Process

- **Speed:** A radiologist can review 50–80 scans per day; AI can process thousands per hour
- **Consistency:** Human review quality varies with fatigue, workload, and experience
- **Prioritization:** Without triage, critical cases may wait behind routine scans
- **Scalability:** Adding more radiologists is expensive and slow
- **Data from KPI sample:** Average resolution time is 35–45 hours; error rate averages 7–8%; satisfaction scores average 6.5/10 — all indicating room for significant improvement

---

## Task 3: AI Task Type

**Selected Task Type: Image Classification**

### Why Image Classification?
- Each chest X-ray must be assigned exactly one diagnostic category (Normal / Pneumonia / Urgent)
- No spatial localization or pixel-level segmentation is required at the triage stage
- Deep learning image classifiers (CNNs) have achieved radiologist-level accuracy on chest X-ray datasets such as CheXNet and NIH Chest X-ray14
- This is the most mature and well-validated AI task in medical imaging

---

## Task 4: Data Requirement Plan

### Type of Data Required

| Data Type | Description | Format |
|---|---|---|
| Chest X-ray images | Frontal-view (PA or AP) digital X-rays | DICOM / JPEG / PNG |
| Diagnostic labels | Ground truth: Normal / Pneumonia / Urgent | Structured (CSV) |
| Patient metadata | Age, sex, referring physician, scan date | Structured (CSV / EHR) |
| Radiologist reports | Text reports for reference validation | Unstructured (PDF / text) |

### Structured vs Unstructured
- **Primary input:** Unstructured (images)
- **Labels and metadata:** Structured

### Input Features
- Raw pixel values of chest X-ray images (resized to 224×224×3)
- Optionally: patient age, sex as auxiliary inputs to the model

### Target Variable / Labels
- `diagnosis`: `0 = Normal`, `1 = Pneumonia`, `2 = Urgent`
- Labels must be verified by at least 2 board-certified radiologists per image

### Data Collection Method
- Partner with 3–5 hospitals to access de-identified X-ray archives
- Use publicly available datasets: NIH ChestX-ray14, CheXpert, MIMIC-CXR
- Minimum recommended dataset size: **50,000 labelled images**

### Data Quality Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Label disagreement between radiologists | Model learns noisy labels | Use majority vote from 3 annotators |
| Class imbalance (Normal >> Pneumonia) | Model biased toward Normal | Oversampling, class weights |
| Image quality variation (equipment, positioning) | Inconsistent features | Standardize preprocessing pipeline |
| Missing or incomplete metadata | Reduced auxiliary signal | Train image-only model as primary |
| Patient privacy (HIPAA/GDPR) | Legal and ethical risk | De-identify all data before use |

---

## Task 5: Model Recommendation

### Recommended Model: Transfer Learning with ResNet-50 (CNN)

| Option | Architecture | Rationale |
|---|---|---|
| ✅ **Primary** | ResNet-50 (pretrained on ImageNet, fine-tuned) | State-of-the-art on medical imaging; fast convergence; proven in CheXNet |
| Baseline | Simple CNN (4–6 conv layers) | Lower accuracy but interpretable and lightweight |
| Advanced | Vision Transformer (ViT) | Higher accuracy at scale but needs more data and compute |

### Why ResNet-50?
- **Transfer learning advantage:** ResNet-50 pretrained on ImageNet has already learned general visual features (edges, textures, shapes). Fine-tuning on X-rays adapts these features to medical patterns with far less labelled data
- **Residual connections** prevent vanishing gradients and allow very deep networks to train stably
- **Proven benchmark:** The CheXNet paper (Stanford, 2017) used a 121-layer DenseNet (similar family) to exceed radiologist-level pneumonia detection
- **Deployment-ready:** Model can be exported as a REST API and integrated with existing PACS systems

### Architecture Overview
```
Input X-ray (224×224×3)
  → Preprocessing (normalize, augment)
  → ResNet-50 backbone (pretrained, fine-tuned)
  → Global Average Pooling
  → Dense(256, ReLU) + Dropout(0.5)
  → Dense(3, Softmax)
  → Output: [P(Normal), P(Pneumonia), P(Urgent)]
```

---

## Task 6: Evaluation Plan

### Technical Metrics

| Metric | Target | Why |
|---|---|---|
| Accuracy | > 90% | Overall correctness |
| Recall (Sensitivity) | > 95% for Urgent class | Missing urgent cases is catastrophic |
| Precision | > 88% | Minimize unnecessary escalations |
| F1-Score | > 91% | Balance of precision and recall |
| AUC-ROC | > 0.95 | Ranking ability across thresholds |

> **Priority:** Recall on the Urgent class is the most critical metric. A false negative (missing an urgent case) is far more harmful than a false positive.

### Business Metrics (from KPI reference data)

| KPI | Current (Baseline) | AI Target |
|---|---|---|
| Average resolution time | 35–45 hours | < 20 hours |
| Manual processing hours/month | 500+ hours | < 200 hours |
| Error rate | 7–8% | < 3% |
| Customer satisfaction score | 6.4–6.9 / 10 | > 8.5 / 10 |

### Possible Failure Cases
- Model performs poorly on X-rays from equipment not seen in training (distribution shift)
- Paediatric or elderly patients have different scan characteristics — model may be less reliable
- Rare diseases or co-occurring conditions may be misclassified
- System downtime or integration failure with PACS

### Human Review and Validation Process
- All AI predictions in the Urgent category must be reviewed by a radiologist before acting
- A random 10% sample of Normal predictions are reviewed weekly to catch systematic errors
- Model performance is monitored monthly; retraining triggered if accuracy drops > 2%
- Radiologists can override AI decisions at any time with documented reasoning

---

## Task 7: Responsible AI Considerations

### 1. Bias in Data
Training data from specific hospitals may not represent all demographic groups equally. X-rays from different equipment, populations (age, sex, ethnicity), or geographies may produce biased predictions.
**Mitigation:** Audit model performance separately across demographic subgroups. Include diverse data sources. Report disaggregated metrics.

### 2. Incorrect Predictions (False Negatives)
A false negative — classifying an urgent case as Normal — could delay life-saving treatment.
**Mitigation:** Tune the decision threshold to prioritize recall on the Urgent class. Implement mandatory radiologist review for borderline confidence scores (< 80%).

### 3. Privacy Concerns
Medical images are highly sensitive personal data governed by HIPAA (US) and GDPR (EU). Unauthorized access or data leakage is a serious legal and ethical risk.
**Mitigation:** De-identify all training data. Encrypt data at rest and in transit. Restrict model API access to authenticated hospital systems only.

### 4. Over-Reliance on AI
Radiologists may begin rubber-stamping AI decisions without independent judgment, especially under time pressure.
**Mitigation:** Train radiologists that AI is a decision-support tool, not a replacement. Log all cases where radiologist agreed vs. overrode AI for accountability.

### 5. Impact on Users (Patients)
Patients may not know their diagnosis was AI-assisted. Incorrect AI triage could cause harm — either delayed urgent care or unnecessary anxiety from false positives.
**Mitigation:** Require informed consent disclosure that AI is used in the diagnostic workflow. Ensure patients can request human-only review.

### 6. Need for Human Oversight
Medical AI must never operate fully autonomously in clinical decisions.
**Mitigation:** The system is designed as **AI-assisted triage**, not autonomous diagnosis. All final diagnostic reports are signed by a licensed radiologist. A clinical governance board reviews AI performance quarterly.

---

## Task 8: Final Solution Summary

| Section | Detail |
|---|---|
| **Problem** | Radiologists manually review hundreds of chest X-rays daily — process is slow (35–45 hrs resolution), error-prone (7–8% error rate), and cannot scale with demand |
| **Proposed AI Solution** | CNN-based chest X-ray triage system using ResNet-50 (transfer learning) to classify scans as Normal, Pneumonia, or Urgent — prioritizing the radiologist's queue automatically |
| **Required Data** | 50,000+ labelled chest X-ray images (NIH, CheXpert, hospital archives); patient metadata; radiologist-verified labels |
| **Model Recommendation** | ResNet-50 pretrained on ImageNet, fine-tuned on chest X-rays; deployed as REST API integrated with hospital PACS |
| **Expected Business Impact** | Resolution time reduced from 45 hrs → < 20 hrs; processing hours cut by 60%; error rate from 8% → < 3%; patient satisfaction from 6.5 → 8.5+/10 |
| **Risks & Mitigation** | Demographic bias → disaggregated audits; false negatives → high-recall threshold + radiologist review; privacy → de-identification + encryption; over-reliance → mandatory human sign-off on all reports |

> **Conclusion:** This AI solution does not replace radiologists — it empowers them. By automating triage, radiologists can focus their expertise on the cases that need it most, improving outcomes for patients and efficiency for hospitals.

---

*Report prepared using reference data from `ai_usecase_reference_catalog.csv` and `business_kpi_sample.csv`*
