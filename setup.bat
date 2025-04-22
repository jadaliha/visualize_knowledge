@echo off
REM Setup script for visualize_knowledge FastAPI app (Windows)

REM Create virtual environment if not exists
if not exist .venv (
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

@echo Setup complete. To run the app:
@echo .venv\Scripts\activate && python app.py
