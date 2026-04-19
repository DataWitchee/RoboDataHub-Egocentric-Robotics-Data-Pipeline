import { useState, useRef } from 'react';
import { UploadCloud, FileVideo, Trash2, Rocket, AlertCircle } from 'lucide-react';
import { uploadVideos, triggerPipeline } from '../api';

const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB
const ALLOWED_TYPES = ['video/mp4', 'video/x-msvideo', 'video/avi'];

const UploadPage = ({ onStartPipeline, showToast }) => {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndAddFiles = (newFiles) => {
    const validFiles = Array.from(newFiles).filter(file => {
      if (!ALLOWED_TYPES.includes(file.type) && !file.name.endsWith('.avi') && !file.name.endsWith('.mp4')) {
        showToast(`Invalid file type: ${file.name}. Only MP4/AVI allowed.`, 'error');
        return false;
      }
      if (file.size > MAX_FILE_SIZE) {
        showToast(`File too large: ${file.name} exceeds 500MB.`, 'error');
        return false;
      }
      // Check for duplicates
      if (files.some(f => f.name === file.name)) {
        showToast(`Duplicate file ignored: ${file.name}`, 'info');
        return false;
      }
      return true;
    });

    if (validFiles.length > 0) {
      // Map to add mock preview properties
      const processedFiles = validFiles.map(f => ({
        file: f,
        name: f.name,
        size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
        // Mocking a thumbnail and duration since extracting from real video in browser requires more complex canvas operations
        thumbnail: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMzMzMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1zaXplPSIxMiIgZmlsbD0id2hpdGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGRvbWluYW50LWJhc2VsaW5lPSJtaWRkbGUiPlZJREVPPC90ZXh0Pjwvc3ZnPg==",
        duration: "0:" + Math.floor(Math.random() * 50 + 10)
      }));
      setFiles(prev => [...prev, ...processedFiles]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    validateAndAddFiles(e.dataTransfer.files);
  };

  const handleRemove = (name) => {
    setFiles(prev => prev.filter(f => f.name !== name));
  };

  const handleRunPipeline = async () => {
    if (files.length === 0) return;
    
    setIsUploading(true);
    try {
      // 1. Upload files
      await uploadVideos(files);
      showToast('Files uploaded successfully', 'success');
      
      // 2. Trigger pipeline
      await triggerPipeline();
      onStartPipeline();
    } catch (err) {
      showToast('Failed to start pipeline', 'error');
      setIsUploading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center py-10 animate-in fade-in zoom-in-95 duration-500">
      
      <div className="text-center mb-12 space-y-4">
        <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white">
          RoboData Pipeline
        </h2>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
          Automated action segmentation, object tracking, and natural language descriptions for egocentric robotics videos.
        </p>
      </div>

      <div className="w-full max-w-3xl bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-2xl p-6 md:p-8 shadow-2xl">
        
        {/* Dropzone */}
        <div 
          className={`relative border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center transition-all duration-300 ease-in-out cursor-pointer
            ${isDragActive ? 'border-primary bg-primary/10 scale-[1.01]' : 'border-slate-600 hover:border-primary/50 hover:bg-slate-700/30'}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
          onDragLeave={() => setIsDragActive(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            multiple 
            accept=".mp4,.avi"
            onChange={(e) => validateAndAddFiles(e.target.files)}
          />
          <div className="bg-slate-800 p-4 rounded-full shadow-lg mb-4 text-primary">
            <UploadCloud className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold mb-2">Drag & drop your videos here</h3>
          <p className="text-slate-400 text-sm mb-4">or click to browse from your computer</p>
          
          <div className="flex gap-4 text-xs font-medium text-slate-500">
            <span className="flex items-center gap-1"><AlertCircle className="w-3 h-3" /> MP4 / AVI</span>
            <span className="flex items-center gap-1"><AlertCircle className="w-3 h-3" /> Max 500MB</span>
          </div>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="mt-8 space-y-3">
            <div className="flex justify-between items-center px-1">
              <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                Ready to Process ({files.length})
              </h4>
              <span className="text-xs text-slate-500">
                {files.reduce((acc, f) => acc + parseFloat(f.size), 0).toFixed(2)} MB total
              </span>
            </div>
            
            <div className="max-h-60 overflow-y-auto pr-2 space-y-2">
              {files.map((file) => (
                <div key={file.name} className="flex items-center gap-4 bg-slate-900/50 p-3 rounded-lg border border-slate-700/50 group hover:border-slate-600 transition-colors">
                  <div className="w-16 h-10 bg-slate-800 rounded overflow-hidden flex-shrink-0 border border-slate-700">
                    <img src={file.thumbnail} alt="preview" className="w-full h-full object-cover opacity-80" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
                    <div className="flex gap-3 text-xs text-slate-500 mt-1">
                      <span>{file.size}</span>
                      <span>•</span>
                      <span>{file.duration}</span>
                    </div>
                  </div>
                  
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleRemove(file.name); }}
                    className="p-2 text-slate-500 hover:text-rose-400 hover:bg-rose-400/10 rounded transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Bar */}
        <div className="mt-8 pt-6 border-t border-slate-700/60 flex justify-end">
          <button
            onClick={handleRunPipeline}
            disabled={files.length === 0 || isUploading}
            className={`
              flex items-center gap-2 px-6 py-3 rounded-lg font-bold transition-all duration-300
              ${files.length > 0 && !isUploading
                ? 'bg-primary text-white hover:bg-blue-400 hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] hover:-translate-y-0.5' 
                : 'bg-slate-700 text-slate-500 cursor-not-allowed'}
            `}
          >
            {isUploading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                Uploading...
              </>
            ) : (
              <>
                <Rocket className="w-5 h-5" />
                Run Pipeline
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
};

export default UploadPage;
