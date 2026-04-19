import React, { useState, useEffect } from 'react';
import { getResults, downloadDataset } from '../api';
import { DownloadCloud, RotateCcw, Video, Target, FileText, Database, ChevronDown, ChevronRight, AlertTriangle } from 'lucide-react';

const ResultsPage = ({ onRunAgain, showToast }) => {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedRows, setExpandedRows] = useState(new Set());
  const [isDownloading, setIsDownloading] = useState(false);
  const [showCSV, setShowCSV] = useState(false);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const data = await getResults();
        setResults(data);
        setError(null);
      } catch (err) {
        setError(err.message || 'Failed to load results');
        showToast('Failed to load results', 'error');
      } finally {
        setIsLoading(false);
      }
    };
    fetchResults();
  }, [showToast]);

  const toggleRow = (id) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(id)) newExpanded.delete(id);
    else newExpanded.add(id);
    setExpandedRows(newExpanded);
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      await downloadDataset();
      showToast('Dataset ready! Download starting...', 'success');
    } catch (err) {
      showToast('Download failed', 'error');
    } finally {
      setIsDownloading(false);
    }
  };

  // ── Loading state ──────────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center space-y-4">
        <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
        <p className="text-slate-400 font-medium">Compiling final dataset...</p>
      </div>
    );
  }

  // ── Error state ────────────────────────────────────────────────────────────
  if (error || !results) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center space-y-6">
        <div className="p-4 bg-rose-500/10 rounded-full">
          <AlertTriangle className="w-12 h-12 text-rose-400" />
        </div>
        <div className="text-center space-y-2">
          <h3 className="text-xl font-bold text-white">No Results Available</h3>
          <p className="text-slate-400 max-w-md">
            {error || 'The pipeline has not produced any results yet. Upload videos and run the pipeline first.'}
          </p>
        </div>
        <button
          onClick={onRunAgain}
          className="flex items-center gap-2 px-5 py-2 bg-primary text-white rounded-lg hover:bg-blue-500 font-bold transition-all"
        >
          <RotateCcw className="w-4 h-4" />
          Go Back & Upload
        </button>
      </div>
    );
  }

  // ── Live data from API response ────────────────────────────────────────────
  const { summary, segments } = results;

  // Read fields using exact backend snake_case names
  const totalSegments      = summary.total_segments ?? 0;
  const totalObjects       = summary.total_objects_annotated ?? 0;
  const totalDescriptions  = summary.descriptions_generated ?? 0;
  const splits             = summary.splits ?? { train: 0, val: 0, test: 0 };
  const avgDuration        = summary.avg_segment_duration ?? 0;
  const actionCategories   = summary.action_categories ?? [];

  return (
    <div className="w-full flex-1 animate-in fade-in slide-in-from-bottom-6 duration-500 py-6">
      
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">Pipeline Completed</h2>
          <p className="text-slate-400 mt-1 block">
            Processed {totalSegments} segment{totalSegments !== 1 ? 's' : ''} across {actionCategories.length} action categor{actionCategories.length !== 1 ? 'ies' : 'y'}.
          </p>
        </div>
        
        <div className="flex gap-4">
          <button 
            onClick={onRunAgain}
            className="flex items-center gap-2 px-4 py-2 text-slate-300 bg-slate-800 hover:bg-slate-700 hover:text-white rounded-lg transition-colors border border-slate-700"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="hidden sm:inline">Run Again</span>
          </button>
          
          <button 
            onClick={handleDownload}
            disabled={isDownloading}
            className="flex items-center gap-2 px-5 py-2 bg-primary text-white rounded-lg hover:bg-blue-500 font-bold shadow-lg shadow-primary/20 transition-all hover:-translate-y-0.5 disabled:opacity-70 disabled:hover:translate-y-0"
          >
            {isDownloading ? (
               <div className="w-4 h-4 rounded-full border-2 border-white/50 border-t-white animate-spin"></div>
            ) : (
               <DownloadCloud className="w-5 h-5" />
            )}
            Download ZIP
          </button>
        </div>
      </div>

      {/* Summary Cards — ALL values from live API response */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard 
          icon={<Video className="w-6 h-6 text-emerald-400" />} 
          title="Total Segments" 
          value={totalSegments}
          subtitle={`Avg ${avgDuration}s each`}
        />
        <StatCard 
          icon={<Target className="w-6 h-6 text-blue-400" />} 
          title="Objects Annotated" 
          value={totalObjects}
          subtitle="Bounding boxes"
        />
        <StatCard 
          icon={<FileText className="w-6 h-6 text-purple-400" />} 
          title="NL Descriptions" 
          value={totalDescriptions}
          subtitle="Generated"
        />
        <StatCard 
          icon={<Database className="w-6 h-6 text-orange-400" />} 
          title="Data Split" 
          value={`${splits.train} / ${splits.val} / ${splits.test}`}
          subtitle="Train / Val / Test"
        />
      </div>

      {/* Action Categories Chips */}
      {actionCategories.length > 0 && (
        <div className="mb-6 flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider mr-2">Actions:</span>
          {actionCategories.map(cat => (
            <span key={cat} className="text-xs px-2.5 py-1 rounded-full bg-primary/10 text-primary border border-primary/20 font-medium">
              {cat}
            </span>
          ))}
        </div>
      )}

      {/* Segments Table — ALL rows from live API response */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700/60 overflow-hidden backdrop-blur-sm">
        <div className="p-5 border-b border-slate-700/60 flex justify-between items-center bg-slate-800/80">
          <h3 className="text-lg font-bold text-white">Segment Details</h3>
          <span className="text-xs bg-slate-700 text-slate-300 px-2 py-1 rounded-md font-mono">
            {segments.length} segment{segments.length !== 1 ? 's' : ''}
          </span>
        </div>
        
        {segments.length === 0 ? (
          <div className="p-12 text-center text-slate-500">
            <p>No segments were produced by the pipeline.</p>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
            <table className="w-full text-left border-collapse">
              <thead className="sticky top-0 z-10">
                <tr className="bg-slate-900/90 text-xs uppercase tracking-wider text-slate-400 border-b border-slate-700/60">
                  <th className="px-6 py-4 font-medium w-8"></th>
                  <th className="px-4 py-4 font-medium">Segment ID</th>
                  <th className="px-4 py-4 font-medium">Action Label</th>
                  <th className="px-4 py-4 font-medium">Duration</th>
                  <th className="px-4 py-4 font-medium">Objects Detected</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/40">
                {segments.map((seg) => {
                  const isExpanded = expandedRows.has(seg.segment_id);
                  const objects = seg.objects_present || [];
                  return (
                    <React.Fragment key={seg.segment_id}>
                      <tr 
                        className={`group hover:bg-slate-700/20 cursor-pointer transition-colors ${isExpanded ? 'bg-slate-700/20' : ''}`}
                        onClick={() => toggleRow(seg.segment_id)}
                      >
                        <td className="px-6 py-4">
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4 text-slate-500" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-primary transition-colors" />
                          )}
                        </td>
                        <td className="px-4 py-4 text-sm font-mono text-slate-300">
                          {seg.segment_id}
                        </td>
                        <td className="px-4 py-4">
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                            {seg.action_label}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-sm text-slate-400">
                          {seg.duration}s
                        </td>
                        <td className="px-4 py-4 text-sm">
                          {objects.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {objects.slice(0, 3).map(obj => (
                                <span key={obj} className="text-xs px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">
                                  {obj}
                                </span>
                              ))}
                              {objects.length > 3 && (
                                <span className="text-xs px-1.5 py-0.5 rounded bg-slate-700 text-slate-400">
                                  +{objects.length - 3}
                                </span>
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-slate-500 italic">none detected</span>
                          )}
                        </td>
                      </tr>
                      
                      {/* Expanded Row — NL Description */}
                      {isExpanded && (
                        <tr className="bg-slate-900/30 border-b border-slate-700/60">
                          <td colSpan={5} className="px-6 py-5">
                            <div className="pl-6 border-l-2 border-primary/30">
                              <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">Natural Language Description</h5>
                              {seg.nl_description ? (
                                <p className="text-slate-300 text-sm leading-relaxed max-w-4xl text-pretty">
                                  "{seg.nl_description}"
                                </p>
                              ) : (
                                <p className="text-slate-500 text-sm italic">
                                  No description generated for this segment.
                                </p>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Raw CSV Data Display */}
      {results.csv_content && (
        <div className="mt-8 bg-slate-800/60 rounded-xl border border-slate-700/60 overflow-hidden backdrop-blur-sm">
          <div 
            className="p-5 border-b border-slate-700/60 flex justify-between items-center bg-slate-800/80 cursor-pointer hover:bg-slate-700/60 transition-colors"
            onClick={() => setShowCSV(!showCSV)}
          >
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-emerald-400" />
              <h3 className="text-lg font-bold text-white">Raw Final CSV Data</h3>
            </div>
            {showCSV ? <ChevronDown className="w-5 h-5 text-slate-400" /> : <ChevronRight className="w-5 h-5 text-slate-400" />}
          </div>
          
          {showCSV && (
            <div className="p-4 bg-slate-900 overflow-x-auto max-h-[400px] overflow-y-auto">
              <pre className="text-xs text-slate-300 font-mono whitespace-pre">
                {results.csv_content}
              </pre>
            </div>
          )}
        </div>
      )}
      
    </div>
  );
};

const StatCard = ({ icon, title, value, subtitle }) => (
  <div className="p-5 rounded-xl bg-slate-800/50 border border-slate-700/50 backdrop-blur hover:bg-slate-800 transition-colors">
    <div className="flex items-center gap-3 mb-3">
      <div className="p-2 bg-slate-900/50 rounded-lg">
        {icon}
      </div>
      <h3 className="text-sm font-medium text-slate-400">{title}</h3>
    </div>
    <div className="mt-1">
      <span className="text-2xl font-bold text-white">{value}</span>
    </div>
    <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
  </div>
);

export default ResultsPage;
