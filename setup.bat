@echo off
REM ============================================================
REM HR Copilot — Setup Script (Windows)
REM Run: setup.bat
REM ============================================================
echo.
echo  ============================================================
echo   HR Copilot -- Multi-Agent Setup -- Windows
echo   Maneesh Kumar
echo  ============================================================
echo.

REM 1. Python check
python --version 2>nul || (echo [ERROR] Python not found. Install Python 3.10+ from python.org && exit /b 1)
echo [OK] Python found

REM 2. Virtual environment
IF EXIST ".venv" (echo [SKIP] .venv already exists) ELSE (python -m venv .venv && echo [OK] Virtual environment created)
call .venv\Scripts\activate
echo [OK] Virtual environment activated

REM 3. pip upgrade
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

REM 4. Dependencies (includes streamlit)
echo Installing all dependencies including Streamlit...
pip install -r requirements.txt --quiet
echo [OK] All dependencies installed

REM 5. Download AI models
echo Downloading AI models...
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; from transformers import pipeline; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2'); pipeline('text-classification', model='cross-encoder/nli-deberta-v3-small'); print('Models ready')"
echo [OK] Models downloaded

REM 6. Create sample data and build index
echo Creating HR sample data...
python create_sample_data.py
echo Building knowledge index...
python component_a_hr_indexing.py
echo [OK] Knowledge index built

REM 7. Ollama install + Mistral pull
where ollama >nul 2>nul
IF NOT ERRORLEVEL 1 (
    echo [OK] Ollama found
) ELSE (
    echo Installing Ollama...
    winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements >nul 2>nul
    IF ERRORLEVEL 1 (
        echo [WARN] winget install failed. Please install Ollama manually from https://ollama.com
        echo [WARN] After installing, run: ollama pull mistral
        goto :ollama_done
    )
    echo [OK] Ollama installed
)

echo Starting Ollama server...
start /b ollama serve
timeout /t 5 /nobreak >nul
echo Pulling Mistral 7B model (~4.1 GB, first time only)...
ollama pull mistral && echo [OK] Mistral 7B ready || echo [WARN] Mistral pull failed - reinstall Ollama from https://ollama.com
:ollama_done

echo.
echo  ============================================================
echo   Setup complete! Ollama + Mistral 7B are installed and ready.
echo   Launch the app:
echo.
echo   .venv\Scripts\activate
echo   streamlit run hr_copilot_ui.py
echo   Opens at: http://localhost:8501
echo  ============================================================
echo.
pause
