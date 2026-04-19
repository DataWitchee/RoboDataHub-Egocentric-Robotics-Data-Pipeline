/**
 * Real API calls connecting to the FastAPI backend.
 */

const BASE_URL = 'http://localhost:8000';

export const uploadVideos = async (files) => {
  const formData = new FormData();
  
  Array.from(files).forEach(f => {
    // files[] is an array of wrapper objects {file: File, name, size, ...}
    // We must pass the real File blob, not the wrapper object
    const realFile = f.file ?? f;   // handle both wrapped and raw File
    formData.append('videos', realFile, realFile.name);
  });
  
  const res = await fetch(`${BASE_URL}/upload`, { 
    method: 'POST', 
    body: formData   // do NOT set Content-Type header — browser sets it with the boundary automatically
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`Upload failed (${res.status}): ${errBody}`);
  }
  return await res.json();
};

export const triggerPipeline = async () => {
  const res = await fetch(`${BASE_URL}/run-pipeline`, { method: 'POST' });
  if (!res.ok) throw new Error("Trigger pipeline failed");
  return await res.json();
};

export const cancelPipeline = async () => {
  // Our backend doesn't have a cancel endpoint yet, so we mock returning success
  console.log("Mock cancelling pipeline...");
  return { success: true };
};

export const getPipelineStatus = async () => {
  const res = await fetch(`${BASE_URL}/pipeline-status`);
  if (!res.ok) throw new Error("Status fetch failed");
  return await res.json();
};

export const getResults = async () => {
  const res = await fetch(`${BASE_URL}/results`);
  if (!res.ok) throw new Error("Results fetch failed");
  return await res.json();
};

export const downloadDataset = async () => {
  const res = await fetch(`${BASE_URL}/download`);
  if (!res.ok) throw new Error("Download failed");
  
  // Create a blob URL and trigger a browser download
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  
  // Extract filename from the Content-Disposition header if possible, otherwise use a generic name
  const contentDisposition = res.headers.get('Content-Disposition');
  let filename = "robodata_dataset.zip";
  if (contentDisposition && contentDisposition.includes('filename=')) {
    filename = contentDisposition.split('filename=')[1].replace(/"/g, '');
  }
  
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
  
  return { success: true };
};
