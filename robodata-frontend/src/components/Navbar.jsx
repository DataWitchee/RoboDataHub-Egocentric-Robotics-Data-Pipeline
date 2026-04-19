import { Activity } from 'lucide-react';

const Navbar = ({ currentStage }) => {
  return (
    <header className="bg-slate-800/50 backdrop-blur-md border-b border-slate-700/50 sticky top-0 z-40">
      <div className="max-w-6xl mx-auto px-4 md:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-primary/20 p-2 rounded-lg border border-primary/30">
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            RoboData <span className="text-primary opacity-90">Pipeline</span>
          </h1>
        </div>

        <div className="flex items-center gap-2 text-sm font-medium">
          <Pill active={currentStage === 'upload'}>1. Upload</Pill>
          <div className="w-4 h-[1px] bg-slate-600 rounded"></div>
          <Pill active={currentStage === 'progress'}>2. Processing</Pill>
          <div className="w-4 h-[1px] bg-slate-600 rounded"></div>
          <Pill active={currentStage === 'results'}>3. Results</Pill>
        </div>
      </div>
    </header>
  );
};

const Pill = ({ children, active }) => (
  <span className={`px-3 py-1.5 rounded-full transition-colors duration-300 ${
    active ? 'bg-primary/20 text-primary border border-primary/30 shadow-[0_0_10px_rgba(59,130,246,0.1)]' : 'text-slate-400 border border-transparent'
  }`}>
    {children}
  </span>
);

export default Navbar;
