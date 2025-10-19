/**
 * Evidence Strength Blocks - Stacked blocks showing evidence weight
 * More blocks = stronger evidence
 */

interface EvidenceStrengthProps {
  strength: number; // 0-1
  maxBlocks?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

export default function EvidenceStrength({
  strength,
  maxBlocks = 10,
  size = 'md',
  showLabel = true,
  className = ''
}: EvidenceStrengthProps) {
  const filledBlocks = Math.round(strength * maxBlocks);

  const blockHeight = size === 'sm' ? 'h-3' : size === 'lg' ? 'h-5' : 'h-4';
  const blockWidth = size === 'sm' ? 'w-3' : size === 'lg' ? 'w-5' : 'w-4';

  const quality =
    strength >= 0.8 ? 'STRONG' :
    strength >= 0.5 ? 'MEDIUM' :
    'WEAK';

  const qualityColor =
    strength >= 0.8 ? 'text-gray-900' :
    strength >= 0.5 ? 'text-gray-700' :
    'text-red-600';

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Blocks */}
      <div className="flex items-end gap-0.5">
        {Array.from({ length: maxBlocks }).map((_, index) => (
          <div
            key={index}
            className={`${blockWidth} ${blockHeight} border border-gray-200 transition-all rounded-sm ${
              index < filledBlocks ? 'bg-gray-900' : 'bg-white'
            }`}
          />
        ))}
      </div>

      {/* Label */}
      {showLabel && (
        <div className="flex flex-col">
          <span className={`text-xs font-bold ${qualityColor}`}>{quality}</span>
          <span className="text-xs text-gray-600">{Math.round(strength * 100)}%</span>
        </div>
      )}
    </div>
  );
}
