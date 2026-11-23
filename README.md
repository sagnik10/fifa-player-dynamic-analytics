<!-- SAGNIK10-STYLE-HEADER -->
<div align="center">
  <a href="https://github.com/sagnik10"><img src="https://img.shields.io/badge/Sagnik_Sen-Web%20%26%20Interactive-275cce?style=flat-square" alt="Category" /></a>
  <img src="https://img.shields.io/badge/Stack-Python-192a43?style=flat-square" alt="Stack" />
  <img src="https://img.shields.io/badge/License-Apache-2.0-54647a?style=flat-square" alt="License" />
</div>

<p align="center">
  <a href="https://sagnik10.github.io/SagnikSenPortfolio/">Portfolio</a> ·
  <a href="https://github.com/sagnik10">GitHub Profile</a> ·
  <a href="https://github.com/sagnik10/sagnik-sen-antigravity-plugin">Antigravity Plugin</a>
</p>

---
<!-- /SAGNIK10-STYLE-HEADER -->
# FIFA Player Dynamic Analytics

Dynamic exploratory analysis and automated PDF reporting for FIFA player performance and market value datasets.

## Overview

This project analyzes FIFA player data and generates a dataset-aware analytics report. The analyzer detects available numerical and categorical columns, selects a useful prediction target, creates charts, writes plain-language insights, and runs basic machine learning diagnostics.

The included dataset is:

- `fifa_player_performance_market_value.csv`

## Features

- Automatic local CSV/Excel discovery
- Dynamic PDF report generation
- Chart-level insight generation
- Correlation heatmaps
- Numeric distribution charts
- Category frequency charts
- Target relationship charts
- Category-vs-target comparisons
- Boxplots for group spread analysis
- Random Forest and Gradient Boosting model diagnostics
- Feature importance charts
- KMeans player clustering
- Isolation Forest anomaly detection
- Exported insight summary CSV

## Project Structure

```text
.
├── Analyzer.py
├── fifa_player_performance_market_value.csv
├── README.md
├── LICENSE
├── requirements.txt
└── Output/                 # generated after running the analyzer
```

## Setup

Install Python 3.10 or newer, then install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the analyzer from the project root:

```powershell
python Analyzer.py
```

On Windows, if the Python launcher is configured:

```powershell
py Analyzer.py
```

## Outputs

After a successful run, the script creates:

- `Output/FIFA_Dynamic_Player_Analytics_Report.pdf`
- `Output/dynamic_insights_summary.csv`
- `Output/charts/`

## Notes

The analyzer is designed to work beyond this single dataset. If another CSV or Excel file is added to the project folder, the script will detect it and generate report sections based on its available columns.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.


<!-- SAGNIK10-STYLE-FOOTER -->
---
<p align="center">
  Maintained by <a href="https://github.com/sagnik10"><strong>Sagnik Sen</strong></a> ·
  <a href="https://sagnik10.github.io/SagnikSenPortfolio/">Portfolio &amp; 1-on-1 Teaching</a> ·
  <a href="https://www.linkedin.com/in/sagnik-sen-0a70431a8/">LinkedIn</a>
</p>
<!-- /SAGNIK10-STYLE-FOOTER -->