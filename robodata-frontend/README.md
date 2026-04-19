# RoboData Pipeline Frontend

A clean, modern React + Tailwind CSS web application built for the Robotics Data Hackathon.

This frontend allows users to:
1. Upload egocentric `.mp4` and `.avi` videos.
2. Trigger the automated 5-stage robotics data pipeline.
3. Monitor pipeline execution in real-time with an animated stepper and progress bars.
4. View the final dataset structure (action labels, bounding boxes, natural language descriptions) and download the compiled files.

## Project Architecture

- **React (`App.jsx`)**: Manages high-level state (routing and toast notifications). No external routers are used to keep the SPA extremely lightweight.
- **Tailwind CSS**: 100% utility-classes implementation with custom extension in `tailwind.config.js`. Dark navy and electric blue theme logic.
- **Vite Setup**: Utilized for extremely fast HMR and compilation. No Redux; strictly `useState` and `useEffect`.
- **API File (`api.js`)**: All endpoints simulate backend connections using asynchronous `Promise` delays and mock payloads. Seamlessly plug into your FastAPI server by substituting the endpoints with `fetch()`.

## Installation & Setup

1. **Prerequisites**: Ensure you have Node.js and `npm` installed.
2. **Install Dependencies**:
   ```bash
   npm install
   ```
3. **Run the Development Server**:
   ```bash
   npm run dev
   ```

The application will be running locally (typically `http://localhost:5173`).

## Backend Integration

The app is currently configured with mocked API calls in `/src/api.js`. To attach it to your FastAPI server:

1. Open `/src/api.js`.
2. Replace the mocked `uploadVideos` with:
   ```javascript
   export const uploadVideos = async (files) => {
     const formData = new FormData();
     Array.from(files).forEach(f => formData.append('videos', f));
     const res = await fetch('http://localhost:8000/upload', { method: 'POST', body: formData });
     return await res.json();
   };
   ```
3. Do the same for `triggerPipeline`, `getPipelineStatus`, `getResults`, and `downloadDataset` by targeting `http://localhost:8000/run-pipeline`, `/pipeline-status`, `/results`, and `/download` respectively.
