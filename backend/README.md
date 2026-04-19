# RoboData Pipeline Backend (FastAPI)

This is the Python REST API server designed to orchestrate the 5-stage egocentric robotics video pipeline on the backend. It integrates perfectly with your React + Tailwind frontend.

## Endpoints Provided

- **`POST /upload`**: Accepts multiple MP4/AVI files via multipart-form-data, sanitizes their filenames, handles validation (500MB max bounds), and saves them directly to `/dataset/raw_videos/`.
- **`POST /run-pipeline`**: Kicks off a FastAPI `BackgroundTask` to sequentially run the 5 pipeline stages Python scripts.
- **`GET /pipeline-status`**: Used by the React frontend to heavily poll (every 3s) the current completion percentage, stage execution, and estimated time remaining.
- **`GET /results`**: Fetches the structured output payload mapping segment IDs to descriptions, metrics, and summary card data.
- **`GET /download`**: Auto-zips the `/dataset/final_dataset/` directory securely on demand and returns the generated `robodata_dataset_TIMESTAMP.zip`.
- **`GET /health`**: Standard infrastructure ping endpoint (`{"status": "ok"}`).

## Running the Server Locally

1. Create a virtual environment or use your current Python 3.10 setup.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Uvicorn server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

## Connecting your React Frontend

Ensure your Vite React development server is running locally on `http://localhost:5173`. 
The `main.py` has CORS correctly set up to natively intercept and allow all operations inbound from the Vite host. No extra frontend proxy mapping is required.
