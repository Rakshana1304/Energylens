# EnergyLens – Retail Energy Intelligence

A Streamlit dashboard for retail-store energy monitoring, anomaly detection, energy forecasting, solar PV performance, and data-quality analysis.

**Note:** The default dataset is simulated for demonstration. Replace assumptions such as tariff, emissions factor, store area, and solar capacity with verified data before using results operationally.

## Features

- Portfolio-level electricity, cost, emissions, anomaly, and solar KPIs
- Daily consumption and store comparison charts
- Rule-based and Isolation Forest anomaly detection
- Recommended operational checks
- Monthly consumption forecasting
- Solar PV performance ratio and estimated losses
- CSV upload and analyzed-data/KPI downloads

## Run locally

1. Install Python 3.10 or newer.
2. Clone or download this repository.
3. Open a terminal in the project folder.
4. Create and activate a virtual environment (recommended).
5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

7. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploy with Streamlit Community Cloud

This project is a Streamlit Python application. Streamlit Community Cloud is the intended straightforward hosting option.

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and authorize access if prompted.
4. Choose **Create app**.
5. Select this repository, the `main` branch, and `app.py` as the main file.
6. Deploy the app.

No secrets are required for the default simulated-data version.

## GitHub repository setup

Create a new repository named `EnergyLens-Retail-Energy-Intelligence`, then upload the files in this folder or push them using Git.

If using Git from a terminal:

```bash
git init
git add .
git commit -m "Prepare EnergyLens Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/EnergyLens-Retail-Energy-Intelligence.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If the repository already exists and has commits, follow GitHub's instructions for connecting an existing repository rather than blindly overwriting it.

## Project structure

```text
EnergyLens-Retail-Energy-Intelligence/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```
