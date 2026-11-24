# Optimizing NCAA FBS Viewership: An Analytical Approach to Scheduling and Realignment

This repository contains the data collection pipeline, exploratory analysis, regression inputs, and multi-season simulation framework supporting the research paper ***"Optimizing NCAA FBS Viewership: An Analytical Approach to Scheduling and Realignment."***

By combining a century of historical college football ratings (1924–2024) with broadcast viewership regression modeling and longitudinal Monte Carlo simulations, this project evaluates how alternative conference structures and scheduling policies impact competitive balance and television viewership across the NCAA Football Bowl Subdivision (FBS).

---

## Project Overview

Recent waves of conference realignment have reshaped college football. To quantify the trade-offs between traditional conference affiliations, strength-based scheduling, and European-style promotion/relegation systems, this repository simulates **50-season longitudinal trajectories** across four distinct NCAA FBS scheduling policies:

1. **Status Quo**: Traditional conference alignments paired with standard out-of-conference (OOC) scheduling (normalized baseline $= 100$).
2. **Status Quo + OOC Performance**: Retains existing conferences while dynamically matching teams in out-of-conference games based on prior-season performance/power ratings.
3. **Conference Tiers**: Reorganizes FBS programs into static, strength-based conference tiers without year-over-year movement.
4. **Promotion Tiers**: Groups teams into hierarchical tiers featuring dynamic promotion and relegation based on seasonal performance.

---

## Repository Structure

~~~text
├── Import_Massey_Data.py                        # Headless Selenium & BeautifulSoup scraper for Massey Ratings
├── Parse_Massey.ipynb                           # Notebook driver to scrape 1924–2024 FBS ratings into CSV
├── Data_Exploration.ipynb                       # Longitudinal data cleaning & time-series pivoting of Power ratings
├── NCAAPolicySimulator_HottovyModel.ipynb       # 50-season NCAA FBS scheduling & viewership policy simulator
│
├── massey_ratings.csv                           # Full historical dataset (25,501 team-season records, 19 columns)
├── PwrRankings for IP.csv                       # Subset of 1,270 team-season records formatted for integer programming
├── with and without conferences.xlsx            # Log-viewership regression coefficients (with vs. without conference effects)
│
├── power_comparison_8lines_2.png                # 50-season power rating trajectories: Top 4 Conferences vs. Promotion Tiers
├── viewership_boxplot_normalized_2.png          # Normalized relative viewership distributions across the 4 policies
└── Sloan_Football_Schedule_Policy_Submission-3.pdf  # Two-page research summary paper, methodology, and key findings
~~~

---

## Getting Started

### Prerequisites

Ensure you have **Python 3.9+** installed along with Jupyter Notebook or JupyterLab. Install the required Python dependencies via `pip`:

~~~bash
pip install pandas numpy scipy matplotlib seaborn beautifulsoup4 selenium openpyxl jupyter
~~~

> **Note on Web Scraping:** Running `Import_Massey_Data.py` or `Parse_Massey.ipynb` from scratch requires Google Chrome and a compatible `chromedriver` binary available on your system `PATH` for headless Selenium execution. Because `massey_ratings.csv` is already included in this repository, re-scraping is optional.

---

## Workflow & Usage

### 1. Data Acquisition (`Import_Massey_Data.py` & `Parse_Massey.ipynb`)
* `Import_Massey_Data.py` defines `Scrape_Massey_Ratings(start_year, end_year)`, which queries annual FBS team rating tables from [MasseyRatings.com](https://masseyratings.com) using headless Chrome and parses team records, ranks, offense/defense metrics, and overall `Power` ratings.
* Run `Parse_Massey.ipynb` to execute the scraper across the $1924\text{--}2024$ seasons and export the consolidated output to `massey_ratings.csv`.

### 2. Exploratory Analysis & Optimization Prep (`Data_Exploration.ipynb`)
* Loads `massey_ratings.csv` ($25,501$ team-season observations) to inspect historical coverage, handle naming standardizations, and pivot team `Power` ratings into longitudinal time series.
* Supports preparation of `PwrRankings for IP.csv` (`Team`, `Power`, `Year`), providing clean power-ranking inputs for integer programming (IP) schedule optimization.

### 3. Viewership Regression Inputs (`with and without conferences.xlsx`)
* Contains log-viewership regression specifications (`Sheet2`) comparing model coefficients estimated with and without conference indicator variables, isolating intrinsic team brand/quality effects from conference affiliation effects.

### 4. Policy Simulation (`NCAAPolicySimulator_HottovyModel.ipynb`)
* Implements the multi-season simulation engine (`HottovyModel`) over a $50$-season horizon.
* Evolves team power ratings dynamically, generates annual schedules under each of the four candidate policies, and estimates broadcast viewership outcomes.
* Generates comparative visualizations:
  * **`viewership_boxplot_normalized_2.png`**: Compares normalized relative viewership distributions across all four scheduling policies (`Status Quo` $= 100$).
  * **`power_comparison_8lines_2.png`**: Tracks $50$-season average power ratings comparing the top $4$ Status Quo conferences against the top $4$ hypothetical Promotion Tiers.

---

## Documentation & Paper

For full methodological details, mathematical formulation, and policy recommendations, refer to the included summary paper:
* **`Sloan_Football_Schedule_Policy_Submission-3.pdf`** — *Optimizing NCAA FBS Viewership: An Analytical Approach to Scheduling and Realignment*.