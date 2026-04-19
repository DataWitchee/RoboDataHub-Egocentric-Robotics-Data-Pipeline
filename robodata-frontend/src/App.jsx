import { useState } from 'react';
import Navbar from './components/Navbar';
import UploadPage from './components/UploadPage';
import PipelineProgress from './components/PipelineProgress';
import ResultsPage from './components/ResultsPage';

function App() {
  // 'upload', 'progress', 'results'
  const [currentPage, setCurrentPage] = useState('upload');
  
  // State for toasts
  const [toast, setToast] = useState({ show: false, message: '', type: 'info' });

  const showToast = (message, type = 'info') => {
    setToast({ show: true, message, type });
    setTimeout(() => {
      setToast({ show: false, message: '', type: 'info' });
    }, 3000);
  };

  const handlePipelineStart = () => {
    setCurrentPage('progress');
  };

  const handlePipelineComplete = () => {
    showToast('Pipeline completed successfully!', 'success');
    setCurrentPage('results');
  };

  const handlePipelineCancel = () => {
    showToast('Pipeline cancelled', 'error');
    setCurrentPage('upload');
  };

  const handleRunAgain = () => {
    setCurrentPage('upload');
  };

  return (
    <div className="min-h-screen bg-background text-slate-100 flex flex-col font-sans">
      <Navbar currentStage={currentPage} />
      
      <main className="flex-1 flex flex-col p-4 md:p-8 max-w-6xl mx-auto w-full">
        {currentPage === 'upload' && (
          <UploadPage 
            onStartPipeline={handlePipelineStart} 
            showToast={showToast} 
          />
        )}
        
        {currentPage === 'progress' && (
          <PipelineProgress 
            onComplete={handlePipelineComplete}
            onCancel={handlePipelineCancel}
            showToast={showToast}
          />
        )}
        
        {currentPage === 'results' && (
          <ResultsPage 
            onRunAgain={handleRunAgain} 
            showToast={showToast} 
          />
        )}
      </main>

      {/* Toast Notification Base */}
      {toast.show && (
        <div className={`fixed bottom-4 right-4 px-6 py-3 rounded-lg shadow-lg font-medium transition-opacity z-50
          ${toast.type === 'success' ? 'bg-green-600 text-white' : 
            toast.type === 'error' ? 'bg-rose-600 text-white' : 
            'bg-slate-700 text-white'}`}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}

export default App;
