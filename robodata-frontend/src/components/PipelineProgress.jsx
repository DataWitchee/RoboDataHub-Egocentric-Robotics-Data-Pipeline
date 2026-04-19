import { useState, useEffect } from 'react';
import { getPipelineStatus, cancelPipeline } from '../api';
import { CheckCircle2, Circle, Clock, Loader2, XCircle } from 'lucide-react';

const STAGES = [
  { id: 1, name: "Video Ingestion & QA" },
  { id: 2, name: "Action Segmentation (PySceneDetect + CLIP)" },
  { id: 3, name: "Object Annotation (YOLOv8 + ByteTrack)" },
  { id: 4, name: "Natural Language Generation (BLIP-2 + Claude)" },
  { id: 5, name: "Dataset Formatting (HuggingFace format)" }
];

const PipelineProgress = ({ onComplete, onCancel, showToast }) => {
  const [status, setStatus] = useState({
    current_stage: 1,
    progress: 0,
    status: 'running',
    estimated_time_remaining_sec: 0,
  });

  useEffect(() => {
    let pollInterval;
    let completed = false;  // guard against double-firing onComplete
    
    const checkStatus = async () => {
      try {
        const res = await getPipelineStatus();
        setStatus(res);
        
        if (!completed && res.status === 'completed') {
          completed = true;
          clearInterval(pollInterval);
          setTimeout(() => onComplete(), 1000);
        } else if (res.status === 'failed') {
          clearInterval(pollInterval);
          showToast('Pipeline failed at stage ' + res.current_stage, 'error');
        } else if (res.status === 'cancelled') {
           clearInterval(pollInterval);
        }
      } catch (e) {
        console.error("Failed to poll status", e);
      }
    };

    // Poll every 3 seconds
    pollInterval = setInterval(checkStatus, 3000);
    checkStatus(); // Initial fetch

    return () => clearInterval(pollInterval);
  }, [onComplete, showToast]);

  const handleCancel = async () => {
    await cancelPipeline();
    onCancel();
  };

  // Format estimated time
  const formatEta = (seconds) => {
    if (!seconds || seconds <= 0) return "0s";
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  // Use the backend field names (snake_case)
  const currentStage = status.current_stage || 0;
  const currentProgress = status.progress || 0;
  const pipelineStatus = status.status || 'idle';

  return (
    <div className="flex-1 flex flex-col items-center justify-center py-10 animate-in fade-in duration-500">
      
      <div className="w-full max-w-2xl bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-2xl p-8 shadow-2xl">
        
        <div className="flex justify-between items-end mb-8 border-b border-slate-700/60 pb-6">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-3">
              {pipelineStatus === 'running' && <Loader2 className="w-6 h-6 text-primary animate-spin" />}
              {pipelineStatus === 'completed' && <CheckCircle2 className="w-6 h-6 text-green-500" />}
              Processing Pipeline
            </h2>
            <p className="text-slate-400 mt-2 text-sm">Do not close this window</p>
          </div>
          
          <div className="flex items-center gap-2 text-slate-300 bg-slate-900/50 px-4 py-2 rounded-lg border border-slate-700">
            <Clock className="w-4 h-4 text-primary" />
            <span className="font-mono text-sm">{formatEta(status.estimated_time_remaining_sec)} remaining</span>
          </div>
        </div>

        <div className="space-y-6">
          {STAGES.map((s) => {
            const isCompleted = currentStage > s.id || (currentStage === s.id && currentProgress === 100 && pipelineStatus === 'completed');
            const isCurrent = currentStage === s.id && pipelineStatus === 'running';
            const isPending = currentStage < s.id;
            const isFailed = currentStage === s.id && pipelineStatus === 'failed';

            return (
              <div key={s.id} className={`relative flex items-start gap-4 transition-opacity duration-300 ${isPending ? 'opacity-40' : 'opacity-100'}`}>
                
                {/* Connecting line (hide for last item) */}
                {s.id !== STAGES.length && (
                  <div className={`absolute left-4 top-10 w-0.5 h-10 -ml-px ${isCompleted ? 'bg-primary' : 'bg-slate-700'}`}></div>
                )}
                
                {/* Icon marker */}
                <div className="relative mt-1 z-10 shrink-0 bg-slate-800">
                  {isCompleted ? (
                    <CheckCircle2 className="w-8 h-8 text-primary fill-primary/20" />
                  ) : isCurrent ? (
                    <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
                  ) : isFailed ? (
                    <XCircle className="w-8 h-8 text-rose-500 fill-rose-500/20" />
                  ) : (
                    <Circle className="w-8 h-8 text-slate-600" />
                  )}
                </div>
                
                {/* Content */}
                <div className="flex-1 pb-4">
                  <h4 className={`text-lg font-semibold ${isCurrent ? 'text-white' : 'text-slate-300'}`}>
                    Stage {s.id}: {s.name}
                  </h4>
                  
                  {isCurrent && (
                    <div className="mt-3 animate-in slide-in-from-top-2 duration-300">
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>Progress</span>
                        <span className="font-mono">{currentProgress}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-700/50">
                        <div 
                          className="h-full bg-primary relative rounded-full transition-all duration-500 ease-out"
                          style={{ width: `${currentProgress}%` }}
                        >
                          <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/40 to-transparent transform -skew-x-12"></div>
                        </div>
                      </div>
                    </div>
                  )}

                  {isFailed && (
                    <div className="mt-2 text-sm text-rose-400 bg-rose-500/10 px-3 py-2 rounded border border-rose-500/20 inline-block">
                      Error: Process aborted during execution.
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-10 pt-6 border-t border-slate-700/60 flex justify-end">
          <button
            onClick={handleCancel}
            disabled={pipelineStatus !== 'running'}
            className="text-slate-400 hover:text-white px-6 py-2 rounded-lg font-medium transition-colors hover:bg-slate-700/50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel Pipeline
          </button>
        </div>

      </div>
    </div>
  );
};

export default PipelineProgress;
