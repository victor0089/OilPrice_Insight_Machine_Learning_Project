# OilPrice_Insight_Machine_Learning_Project


Files included:
  commodity_price_app.py           Executable Python application
  wti_daily_2022_2026.csv          Bundled historical WTI prices (1,214 rows)
  Developing_a_Machine_Learning_App_Report.txt   Academic report
  requirements.txt                 Python dependencies
  results/run_results.txt          Verified execution output
  results/console_run.png          Image of the actual printed output
  results/app_results.png          Application dashboard with forecast
  results/test_predictions.png     Test-set predictions vs. actual prices
  results/feature_importance.png   Random Forest feature ranking
  results/test_predictions.csv     Detailed holdout results

QUICK START - WINDOWS
---------------------
1. Install Python 3.10 or newer from https://www.python.org/.
2. Extract the complete project folder. Keep the .py and .csv together.
3. Open Command Prompt in the extracted folder.
4. Run these commands one at a time:

      py -m venv .venv
      .venv\Scripts\activate
      python -m pip install -r requirements.txt
      python commodity_price_app.py

If PowerShell blocks the activation script, use Command Prompt or run:
      .venv\Scripts\python.exe -m pip install -r requirements.txt
      .venv\Scripts\python.exe commodity_price_app.py

QUICK START - MACOS OR LINUX
----------------------------
      python3 -m venv .venv
      source .venv/bin/activate
      python -m pip install -r requirements.txt
      python commodity_price_app.py

OPTIONAL ARGUMENTS
------------------
      python commodity_price_app.py --data wti_daily_2022_2026.csv --output results

The --data file must contain Date,Price columns. The app also accepts the
observation_date,DCOILWTICO column names exported by FRED.

The app runs offline using the bundled data. It does not fetch or claim to
provide the current market price automatically. Forecasts are one published
observation ahead; weekends and holidays affect the calendar date.

DATA SOURCE
-----------
Origin: U.S. Energy Information Administration, daily Cushing WTI spot prices
Series cross-check: https://fred.stlouisfed.org/series/DCOILWTICO
Download mirror: https://github.com/datasets/oil-prices
Bundled snapshot: January 3, 2022 to November 11, 2026 (inclusive)
Units: U.S. dollars per barrel; 1,214 dated observed prices

EDUCATIONAL USE
---------------
This app is a classroom forecasting exercise, not a trading system.
It compares the model against a persistence baseline on a chronological
unseen test segment. A good statistical fit does not ensure future accuracy.
