# dataset_insight_generator
Automated Dataset Insight Generator

Prerequisites
- Python 3.9+ (3.9.11 used during development)
- Homebrew (macOS) — optional but convenient for installing Ollama
- Enough disk space / RAM for local model(s) you plan to use (models can be large)

Install Python dependencies
1. Create and activate a virtual environment (recommended)
   - macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```

Install Ollama (local model runtime)
- macOS (Homebrew):
  ```bash
  brew install ollama
  ```
- Or follow the official installer for your OS: https://ollama.com/docs

Prepare Ollama models
- Pull the model (`llama3.1:8b`):
  ```bash
  ollama pull llama3.1:8b
  ```
- Ensure Ollama is running and the HTTP API is reachable at `http://localhost:11434`. You can run Ollama Desktop or ensure the CLI daemon/process is started per the Ollama docs.

Quick test (optional)
- Verify Ollama is reachable (replace with appropriate endpoint if needed):
  ```bash
  curl http://localhost:11434
  ```
- If your app cannot reach Ollama, check that Ollama is running and the model was pulled successfully.

Run the Streamlit app
```bash
streamlit run src/app.py
```
This opens the app in your browser. Upload a CSV and optionally a data dictionary (CSV/XLSX/TXT). Click "Run Analysis" to trigger the AI-driven insights.

Notes / Troubleshooting
- The app posts prompts to Ollama at `http://localhost:11434/api/generate` (see `ollama_chat` in `app.py`). Make sure the local Ollama API endpoint and model name match the code.
- If you see pandas EmptyDataError when uploading a schema file, ensure the uploader file is read/reset before parsing (the app expects file-like objects; re-seek is needed if reading multiple times).
- If models are large, downloading/pulling will take time and require disk space.
- Edit `app.py` to change model name, prompt, or Ollama endpoint if desired.