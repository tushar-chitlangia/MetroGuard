# MetroGuard

### Machine Learning-Based Anomaly Detection and Early-Warning Support for Metro Train Air-Compressor Maintenance

MetroGuard is a machine-learning prototype for analysing historical sensor data from a metro train air-compressor system. It investigates whether unusual patterns in compressor telemetry can be identified and presented in a way that may support maintenance investigation.

The project uses the **MetroPT-3 dataset** from the UCI Machine Learning Repository. Its workflow includes time-series preparation, rolling-feature engineering, an Isolation Forest anomaly detector, a statistical comparison baseline, chronological evaluation, and a Streamlit dashboard for exploring the saved results.

> **Project scope:** MetroGuard is an experimental historical anomaly-monitoring prototype. The results do not establish validated early failure prediction, calibrated failure probabilities, or operational safety.

---

## 1. Problem and objectives

Metro train air-compressor equipment produces sensor readings that vary with operating conditions. This project investigates whether an unsupervised machine-learning approach can identify sensor patterns that differ from normal training-period behaviour.

The main objectives are to:

- Prepare timestamped compressor sensor readings for time-series analysis.
- Engineer features that summarise recent sensor behaviour.
- Train an Isolation Forest model using normal observations from the training period.
- Compare its results with a statistical sensor z-score baseline.
- Evaluate both methods on later chronological periods using thresholds selected on validation data.
- Present model outputs, sensor trends, and alert activity through an interactive dashboard.

The prototype analyses historical data. It does not connect to live train telemetry or provide confirmed equipment-fault diagnoses.

## 2. Dataset

**Source:** [UCI Machine Learning Repository — MetroPT-3 Dataset (ID 791)](https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset)

The dataset contains timestamped measurements from a metro train air-production/compressor system.

| Dataset property | Value |
|---|---:|
| Raw records | 1,516,948 |
| Sensor signals | 15 |
| Recorded period | 1 February 2020 – 1 September 2020 |
| Rows after one-minute resampling | 306,960 |
| Model-ready rows | 243,277 |
| Model features | 75 |

The 15 sensor signals used are `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`, `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level` and `Caudal_impulses`.

The original dataset is not stored in this repository. The notebook retrieves it from UCI when run in an environment with internet access. Refer to the UCI dataset page for the source data and its terms of use.

### Incident reference and warning-window assumption

The experiment uses four reported incident periods to construct an evaluation reference. For each documented incident, the **120 minutes immediately before its start time** are marked as an assumed warning window. The incident period itself is excluded from model-ready observations.

This 120-minute window is a project-defined assumption, not verified minute-by-minute ground truth. Therefore, the evaluation measures agreement with the assumed reference labels; it does not establish whether the model detected confirmed pre-failure signals.

## 3. Data preparation and feature engineering

The notebook performs the following steps:

1. Parses timestamps and sorts records chronologically.
2. Retains the 15 expected sensor columns and converts their values to numeric format.
3. Resamples sensor readings into one-minute intervals using the mean.
4. Identifies completely empty minutes and uses them to separate time-series segments.
5. Forward-fills short gaps of up to five minutes within each segment; completely empty minutes remain missing.
6. Calculates rolling means and rolling standard deviations for 5-minute and 15-minute windows for each sensor.
7. Uses `shift(1)` when calculating rolling features so that each feature uses preceding observations rather than current or future sensor values.
8. Excludes documented incident intervals and rows without complete feature values.

The final feature set contains 15 current sensor measurements and 60 rolling-statistic features, resulting in **75 features** for modelling.

## 4. Modelling approach

### Isolation Forest — primary model

Isolation Forest is an unsupervised anomaly-detection algorithm. It identifies observations that are easier to isolate from the rest of the data. In this project, it is fitted on normal observations from the training period.

Key implementation details:

- `StandardScaler` is fitted on normal training observations and applied to validation and test observations.
- Isolation Forest uses `n_estimators=150`, `contamination="auto"`, and `random_state=42`.
- The anomaly score is the negative of the model's decision function, so higher scores indicate more unusual observations.

### Statistical baseline

The baseline uses the raw sensor measurements:

1. Calculate each sensor's mean and standard deviation from normal training observations.
2. Calculate the absolute z-score for each sensor.
3. Use the maximum absolute z-score across the 15 sensors as the observation's anomaly score.

This provides a straightforward statistical reference for comparison with Isolation Forest.

## 5. Evaluation methodology

The data is divided chronologically rather than randomly.

| Partition | Period | Purpose |
|---|---|---|
| Training | Before 1 June 2020 | Fit scaling and anomaly detection using normal training rows |
| Validation | June 2020 | Select alert thresholds and assess validation behaviour |
| Held-out test | July 2020 | Evaluate the fixed thresholds on later observations |

The saved July test predictions contain **35,898 rows**.

### Threshold selection

Each method has its own anomaly-score scale and therefore its own alert threshold. Thresholds are selected using normal June validation scores, targeting approximately a 1% false-positive rate on those normal validation observations. The selected thresholds are then held fixed for the July test evaluation. July test data is not used to select the thresholds.

### Evaluation metrics

- **Precision:** proportion of predicted alert minutes that overlap the assumed warning labels.
- **Recall:** proportion of assumed warning minutes that are flagged.
- **F1-score:** harmonic mean of precision and recall.
- **PR-AUC (average precision):** summarises precision-recall ranking across score thresholds.
- **False positives per 1,000 normal minutes:** false-positive alert minutes scaled to 1,000 observations labelled normal under the project's assumed reference.

These metrics are calculated against the assumed warning labels and should not be interpreted as verified real-world fault-detection metrics.

## 6. Results

### Held-out July 2020 test results

| Metric | Isolation Forest | Statistical baseline |
|---|---:|---:|
| Fixed alert threshold | 0.1524 | 9.1646 |
| PR-AUC (average precision) | **0.1281** | 0.0125 |
| Alert minutes | **197** | 1,502 |
| Assumed warning minutes flagged | 0 of 120 | 0 of 120 |
| Precision | 0.0000 | 0.0000 |
| Recall | 0.0000 | 0.0000 |
| F1-score | 0.0000 | 0.0000 |
| False positives per 1,000 normal minutes | **5.5062** | 41.9811 |

The July event-level check concerns the documented incident beginning on **15 July 2020 at 14:30**. The 120 minutes before that start time form the assumed warning window for this test-period event check.

### June 2020 validation results

| Metric | Isolation Forest | Statistical baseline |
|---|---:|---:|
| PR-AUC (average precision) | 0.0029 | 0.0036 |
| Alert minutes | 329 | 97 |
| Precision / Recall / F1-score | 0 / 0 / 0 | 0 / 0 / 0 |
| False positives per 1,000 normal minutes | 9.9366 | 2.9296 |

The validation results are included to provide context. They show that Isolation Forest did not outperform the baseline on every measure or in every period.

### Interpreting the July results

The zero Precision, Recall, and F1-score values are the actual results at the selected alert thresholds. They are not missing values or a display error.

For the July test event, the 120 minutes before the documented incident are marked as the assumed positive warning window. At the fixed thresholds, Isolation Forest generated 197 alert minutes and the statistical baseline generated 1,502 alert minutes, but neither method flagged any of the 120 assumed warning minutes. Under these labels, neither method produced a true-positive warning minute.

PR-AUC can still be non-zero because it evaluates the ranking of continuous anomaly scores across a range of thresholds. Precision, recall, and F1-score in the table are calculated at one fixed threshold selected using June validation data. A model can therefore have a non-zero PR-AUC while having zero recall at its selected operating threshold. In this test, Isolation Forest ranked the assumed warning observations better overall than the baseline, but its selected threshold did not flag any of them.

### Findings

On the July test period, Isolation Forest achieved a higher PR-AUC than the statistical baseline (0.1281 versus 0.0125) and generated substantially fewer alert minutes (197 versus 1,502). It also had a lower false-positive rate under the assumed reference labels. However, neither method detected any of the 120 assumed warning minutes at its selected threshold. The June validation results also show that Isolation Forest was not better on every measure.

**The results suggest that Isolation Forest provides better score ranking and a lower alert burden than the selected statistical baseline on the July test period. They do not demonstrate successful early-warning detection or validated failure prediction.**

## 7. Streamlit dashboard

The `dashboard/` directory contains the Streamlit application and its saved output artifacts. The dashboard reads these saved outputs; it does not retrain the model.

The dashboard includes:

- **Overview:** key evaluation metrics and anomaly-score timeline.
- **Sensor trends:** inspection of selected sensor measurements over time.
- **Alert timeline:** inspection of alert times and grouped alert episodes.
- **Model comparison:** comparison of Isolation Forest and the statistical baseline.

The dashboard supports exploration of the June validation and July test periods and presents the warning-label assumption alongside the results.

## 8. Run the dashboard locally

### Requirements

- Python 3.10 or later
- `pip`
- The project files, including the saved output files in `dashboard/artifacts/`

### Install dependencies

From the project root, run:

```bash
pip install -r dashboard/requirements.txt
```

### Launch the application

From the project root, run:

```bash
streamlit run dashboard/app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that address in a browser.

A virtual environment is recommended to keep project dependencies isolated from other Python projects. To create and activate one:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies and launch the dashboard using the commands above. The dashboard uses the saved CSV and JSON outputs; it does not need to retrain the model to display the results.

## 9. Reproducing the analysis

The notebook documents the data preparation, feature engineering, chronological split, model training, threshold selection, evaluation, and artifact-generation workflow.

To reproduce the analysis:

1. Obtain the MetroPT-3 dataset from the [UCI dataset page](https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset).
2. Open `notebooks/MetroGuard_Final.ipynb` in Google Colab or Jupyter.
3. Run the notebook cells in order in an environment with the packages used by the notebook and internet access where required.
4. Review the generated metrics, prediction files, event evaluation, and saved model artifact.

The notebook retrieves the source dataset from UCI when it is not already available in its configured runtime location. The saved artifacts in the repository allow the dashboard to be viewed without rerunning model training.

## 10. Limitations and responsible use

- The two-hour warning horizon is an experimental assumption, not verified minute-level ground truth.
- The July test-period event check contains one documented incident. This is insufficient to establish generalisation across incidents, trains, or operating conditions.
- Both models have zero precision, recall, and F1-score at their selected July thresholds under the assumed warning labels.
- Anomaly scores are not calibrated probabilities of equipment failure.
- The prototype uses historical data and is not connected to live train telemetry.
- Sensor anomalies may reflect operating changes, maintenance actions, missing data, or other conditions; alerts require human interpretation.
- The prototype has not been independently validated for operational use and must not be used as a safety system or as the sole basis for maintenance decisions.

## 11. Future work

1. Review the assumed warning horizon and incident interpretation with domain experts, and use better-labelled events where available.
2. Evaluate additional documented incidents and later time periods.
3. Further investigate missing-data handling and verify that time-dependent features use only information available at prediction time.
4. Compare additional anomaly-detection approaches using the same chronological evaluation design and alert-burden constraints.
5. Add event-level measures such as lead time, missed incidents, and false-alarm episodes.
6. Introduce a human review workflow for classifying alerts.
7. Consider real-time integration only after independent validation, monitoring design, and safety review.

## 12. Technology

- **Python** — implementation language
- **Pandas and NumPy** — data preparation and numerical operations
- **scikit-learn** — feature scaling, Isolation Forest, and evaluation metrics
- **Joblib** — model-artifact persistence
- **Matplotlib** — notebook analysis plots
- **Streamlit** — interactive dashboard
- **Plotly** — interactive dashboard charts
- **Google Colab / Jupyter** — notebook execution environment

## References

1. UCI Machine Learning Repository, [MetroPT-3 Dataset (ID 791)](https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset).
2. F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*, pp. 413–422, 2008. [https://doi.org/10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
3. scikit-learn developers, [IsolationForest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
4. Streamlit, [Documentation](https://docs.streamlit.io/).
5. Plotly, [Python graphing library documentation](https://plotly.com/python/).

---

*MetroGuard is an experimental project based on the dataset, implementation, evaluation assumptions, and time periods described above. Its outputs are intended for analytical exploration and demonstration, not operational decision-making.*
