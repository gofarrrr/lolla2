/**
 * Orthogonality Indicator - Visual representation of cognitive diversity
 * Shows overlapping rectangles with gap representing independence of perspectives
 */

interface OrthogonalityIndicatorProps {
  score: number; // 0-1, where higher = more orthogonal (independent)
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

export default function OrthogonalityIndicator({
  score,
  size = 'md',
  showLabel = true,
  className = ''
}: OrthogonalityIndicatorProps) {
  // Calculate gap based on score (0-1 maps to 0-40px)
  const maxGap = size === 'sm' ? 20 : size === 'lg' ? 60 : 40;
  const gap = Math.round(score * maxGap);

  // Box sizes based on size prop
  const boxSize = size === 'sm' ? 'w-8 h-8' : size === 'lg' ? 'w-16 h-16' : 'w-12 h-12';

  // Quality assessment
  const quality =
    score >= 0.7 ? 'HIGH' :
    score >= 0.4 ? 'GOOD' :
    'LOW';

  const qualityColor =
    score >= 0.7 ? 'text-gray-900' :
    score >= 0.4 ? 'text-gray-700' :
    'text-red-600';

  return (
    <div className={`flex flex-col items-center gap-2 ${className}`}>
      {/* Visual indicator */}
      <div className="flex items-center relative" style={{ gap: `${gap}px` }}>
        {/* First rectangle */}
        <div className={`${boxSize} border border-gray-200 bg-white transition-all rounded-xl`} />

        {/* Second rectangle */}
        <div className={`${boxSize} border border-gray-200 bg-white transition-all rounded-xl`} />

        {/* Third rectangle (optional, only for higher scores) */}
        {score > 0.6 && (
          <div className={`${boxSize} border border-gray-200 bg-white transition-all rounded-xl`} />
        )}
      </div>

      {/* Label */}
      {showLabel && (
        <div className="text-center">
          <div className={`text-xs font-bold ${qualityColor}`}>
            {quality} DIVERSITY
          </div>
          <div className="text-xs text-gray-600 font-medium">
            {(score).toFixed(2)} orthogonality
          </div>
        </div>
      )}
    </div>
  );
}
