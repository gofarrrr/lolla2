/**
 * Assumption Diff Bars - Clean before/after visualization
 * Shows assumption confidence changes with filled bars
 */

interface AssumptionDiffBarsProps {
  before: number; // 0-1
  after: number; // 0-1
  showPercentage?: boolean;
  barHeight?: 'sm' | 'md' | 'lg';
  className?: string;
}

export default function AssumptionDiffBars({
  before,
  after,
  showPercentage = true,
  barHeight = 'md',
  className = ''
}: AssumptionDiffBarsProps) {
  const beforePercent = Math.round(before * 100);
  const afterPercent = Math.round(after * 100);

  const heightClass = barHeight === 'sm' ? 'h-2' : barHeight === 'lg' ? 'h-4' : 'h-3';

  const isImprovement = after > before;
  const isDecline = after < before;

  return (
    <div className={`space-y-1 ${className}`}>
      {/* BEFORE bar */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-bold text-gray-500 w-16">BEFORE</span>
        <div className="flex-1 border border-gray-200 bg-white rounded overflow-hidden">
          <div
            className={`${heightClass} transition-all ${isDecline ? 'bg-red-600' : 'bg-gray-400'}`}
            style={{ width: `${beforePercent}%` }}
          />
        </div>
        {showPercentage && (
          <span className={`text-xs font-bold w-12 ${isDecline ? 'text-red-600' : 'text-gray-600'}`}>
            {beforePercent}%
          </span>
        )}
      </div>

      {/* AFTER bar */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-bold text-gray-500 w-16">AFTER</span>
        <div className="flex-1 border border-gray-200 bg-white rounded overflow-hidden">
          <div
            className={`${heightClass} transition-all ${isImprovement ? 'bg-gray-900' : 'bg-gray-400'}`}
            style={{ width: `${afterPercent}%` }}
          />
        </div>
        {showPercentage && (
          <span className={`text-xs font-bold w-12 ${isImprovement ? 'text-gray-900' : 'text-gray-600'}`}>
            {afterPercent}%
          </span>
        )}
      </div>

      {/* Change indicator */}
      {isImprovement && (
        <div className="text-xs text-gray-900 font-bold flex items-center gap-1">
          ↑ +{afterPercent - beforePercent}% improvement
        </div>
      )}
      {isDecline && (
        <div className="text-xs text-red-600 font-bold flex items-center gap-1">
          ↓ {afterPercent - beforePercent}% decline
        </div>
      )}
    </div>
  );
}
