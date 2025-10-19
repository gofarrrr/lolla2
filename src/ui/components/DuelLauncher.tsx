import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Sword,
  Target,
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  Info,
  Zap
} from 'lucide-react';

// Types
interface DuelConfig {
  goldenCaseId: string;
  challengerPromptId: string;
  stationToTest: string;
}

interface DuelResult {
  duel_id: string;
  winner: string;
  quality_delta: number;
  lolla_score: number;
  challenger_score: number;
  execution_time: number;
  status: 'completed' | 'failed';
  error?: string;
}

interface GoldenCase {
  case_id: string;
  user_query: string;
  expected_domain: string;
  expected_outcome_type: string;
}

interface ChallengerPrompt {
  prompt_id: string;
  prompt_name: string;
  target_station: string;
  version: string;
  status: string;
}

interface DuelLauncherProps {
  onDuelComplete: (result: DuelResult) => void;
  preselectedChallengerId?: string;
}

const STATION_OPTIONS = [
  { value: 'FULL_PIPELINE', label: 'Full Pipeline', description: 'Complete 8-station analysis', icon: '⚔️' },
  { value: 'STATION_1', label: 'QUICKTHINK', description: 'Initial rapid analysis', icon: '⚡' },
  { value: 'STATION_2', label: 'DEEPTHINK', description: 'Comprehensive exploration', icon: '🔍' },
  { value: 'STATION_3', label: 'BLUETHINK', description: 'Conservative analysis', icon: '🔵' },
  { value: 'STATION_4', label: 'REDTHINK', description: 'Bold innovation', icon: '🔴' },
  { value: 'STATION_5', label: 'GREYTHINK', description: 'Reality check', icon: '⚪' },
  { value: 'STATION_6', label: 'ULTRATHINK', description: 'Deep synthesis', icon: '🚀' },
  { value: 'STATION_7', label: 'DIVERGENTTHINK', description: 'Alternative perspectives', icon: '🔄' },
  { value: 'STATION_8', label: 'CONVERGENTTHINK', description: 'Final integration', icon: '🎯' }
];

export const DuelLauncher: React.FC<DuelLauncherProps> = ({ 
  onDuelComplete, 
  preselectedChallengerId 
}) => {
  const [config, setConfig] = useState<DuelConfig>({
    goldenCaseId: '',
    challengerPromptId: preselectedChallengerId || '',
    stationToTest: 'FULL_PIPELINE'
  });

  const [goldenCases, setGoldenCases] = useState<GoldenCase[]>([]);
  const [challengers, setChallengers] = useState<ChallengerPrompt[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [lastResult, setLastResult] = useState<DuelResult | null>(null);
  
  // Load data
  useEffect(() => {
    loadGoldenCases();
    loadChallengers();
  }, []);

  useEffect(() => {
    if (preselectedChallengerId) {
      setConfig(prev => ({ ...prev, challengerPromptId: preselectedChallengerId }));
    }
  }, [preselectedChallengerId]);

  const loadGoldenCases = async () => {
    try {
      // Mock golden cases for now
      setGoldenCases(mockGoldenCases);
    } catch (error) {
      console.error('Error loading golden cases:', error);
      setGoldenCases(mockGoldenCases);
    }
  };

  const loadChallengers = async () => {
    try {
      const response = await fetch('/api/proving-ground/challenger-prompts?status=active');
      if (response.ok) {
        const data = await response.json();
        setChallengers(data);
      } else {
        setChallengers(mockChallengers);
      }
    } catch (error) {
      console.error('Error loading challengers:', error);
      setChallengers(mockChallengers);
    }
  };

  const handleLaunchDuel = async () => {
    if (!config.goldenCaseId || !config.challengerPromptId) {
      setError('Please select both a golden case and challenger prompt');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/proving-ground/launch-duel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      const result = await response.json();

      if (response.ok) {
        const duelResult: DuelResult = {
          duel_id: result.duel_id,
          winner: result.comparison.winner,
          quality_delta: result.comparison.quality_delta,
          lolla_score: result.comparison.lolla_score,
          challenger_score: result.comparison.challenger_score,
          execution_time: result.execution_metadata.total_execution_time,
          status: 'completed'
        };

        setLastResult(duelResult);
        onDuelComplete(duelResult);
      } else {
        setError(result.detail || 'Failed to launch duel');
        setLastResult({
          duel_id: '',
          winner: '',
          quality_delta: 0,
          lolla_score: 0,
          challenger_score: 0,
          execution_time: 0,
          status: 'failed',
          error: result.detail || 'Unknown error'
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Network error';
      setError(errorMsg);
      setLastResult({
        duel_id: '',
        winner: '',
        quality_delta: 0,
        lolla_score: 0,
        challenger_score: 0,
        execution_time: 0,
        status: 'failed',
        error: errorMsg
      });
    } finally {
      setLoading(false);
    }
  };

  const selectedGoldenCase = goldenCases.find(gc => gc.case_id === config.goldenCaseId);
  const selectedChallenger = challengers.find(c => c.prompt_id === config.challengerPromptId);
  const selectedStation = STATION_OPTIONS.find(s => s.value === config.stationToTest);

  const isFormValid = config.goldenCaseId && config.challengerPromptId && config.stationToTest;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="flex justify-center mb-4">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full">
            <Sword className="w-8 h-8 text-white" />
          </div>
        </div>
        <h2 className="text-2xl font-bold text-gray-900">Launch Proving Ground Duel</h2>
        <p className="text-gray-600 mt-2">
          Test the Lolla pipeline against monolithic challengers in head-to-head combat
        </p>
      </div>

      {/* Configuration Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="w-5 h-5" />
            <span>Duel Configuration</span>
          </CardTitle>
          <CardDescription>
            Configure the parameters for your proving ground test
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Golden Case Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Golden Case</label>
            <Select
              value={config.goldenCaseId}
              onValueChange={(value) => setConfig({ ...config, goldenCaseId: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a golden case to test against" />
              </SelectTrigger>
              <SelectContent>
                {goldenCases.map(goldenCase => (
                  <SelectItem key={goldenCase.case_id} value={goldenCase.case_id}>
                    <div className="py-1">
                      <div className="font-medium">{goldenCase.case_id}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        {goldenCase.user_query.substring(0, 100)}...
                      </div>
                      <div className="flex items-center space-x-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {goldenCase.expected_domain}
                        </Badge>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedGoldenCase && (
              <div className="mt-2 p-3 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-700">
                  <strong>Query:</strong> {selectedGoldenCase.user_query}
                </p>
                <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                  <span><strong>Domain:</strong> {selectedGoldenCase.expected_domain}</span>
                  <span><strong>Type:</strong> {selectedGoldenCase.expected_outcome_type}</span>
                </div>
              </div>
            )}
          </div>

          {/* Challenger Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Challenger Prompt</label>
            <Select
              value={config.challengerPromptId}
              onValueChange={(value) => setConfig({ ...config, challengerPromptId: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a challenger to face Lolla" />
              </SelectTrigger>
              <SelectContent>
                {challengers.map(challenger => (
                  <SelectItem key={challenger.prompt_id} value={challenger.prompt_id}>
                    <div className="py-1">
                      <div className="font-medium">{challenger.prompt_name}</div>
                      <div className="flex items-center space-x-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {challenger.target_station}
                        </Badge>
                        <span className="text-xs text-gray-500">v{challenger.version}</span>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedChallenger && (
              <div className="mt-2 p-3 bg-purple-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{selectedChallenger.prompt_name}</span>
                  <Badge className="text-xs">
                    {selectedChallenger.status}
                  </Badge>
                </div>
              </div>
            )}
          </div>

          {/* Station Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Test Scope</label>
            <Select
              value={config.stationToTest}
              onValueChange={(value) => setConfig({ ...config, stationToTest: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Choose what to test" />
              </SelectTrigger>
              <SelectContent>
                {STATION_OPTIONS.map(option => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex items-center space-x-2 py-1">
                      <span className="text-base">{option.icon}</span>
                      <div>
                        <div className="font-medium">{option.label}</div>
                        <div className="text-xs text-gray-500">{option.description}</div>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedStation && (
              <div className="mt-2 p-3 bg-green-50 rounded-lg">
                <div className="flex items-center space-x-2 text-sm">
                  <span className="text-lg">{selectedStation.icon}</span>
                  <span className="font-medium">{selectedStation.label}</span>
                  <span className="text-gray-600">- {selectedStation.description}</span>
                </div>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <XCircle className="w-4 h-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Launch Button */}
          <Button
            onClick={handleLaunchDuel}
            disabled={!isFormValid || loading}
            className="w-full h-12 text-lg"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Duel in Progress...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                ⚔️ Launch Duel
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Last Result */}
      {lastResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              {lastResult.status === 'completed' ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : (
                <XCircle className="w-5 h-5 text-red-600" />
              )}
              <span>Last Duel Result</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {lastResult.status === 'completed' ? (
              <div className="space-y-4">
                {/* Winner Announcement */}
                <div className="text-center p-6 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50">
                  <div className="text-3xl font-bold mb-2">
                    🏆 {lastResult.winner.toUpperCase()} WINS!
                  </div>
                  <div className="text-lg text-gray-600">
                    Quality Delta: 
                    <span className={`ml-2 font-bold ${lastResult.quality_delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {lastResult.quality_delta > 0 ? '+' : ''}{lastResult.quality_delta.toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Score Comparison */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-lg font-bold text-blue-800">Lolla Pipeline</div>
                    <div className="text-2xl font-bold text-blue-600 mt-2">
                      {lastResult.lolla_score.toFixed(3)}
                    </div>
                  </div>
                  
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-lg font-bold text-purple-800">Challenger</div>
                    <div className="text-2xl font-bold text-purple-600 mt-2">
                      {lastResult.challenger_score.toFixed(3)}
                    </div>
                  </div>
                </div>

                {/* Metadata */}
                <div className="flex justify-between items-center text-sm text-gray-600 pt-4 border-t">
                  <span>Duel ID: {lastResult.duel_id}</span>
                  <span>Execution Time: {lastResult.execution_time.toFixed(2)}s</span>
                </div>
              </div>
            ) : (
              <div className="text-center py-6">
                <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <div className="text-lg font-medium text-red-800 mb-2">Duel Failed</div>
                <div className="text-sm text-red-600">{lastResult.error}</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Info Card */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardContent className="pt-6">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-600 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-medium mb-2">How Proving Ground Duels Work:</p>
              <ul className="space-y-1 text-sm">
                <li>• The selected golden case is analyzed by both Lolla pipeline and the challenger</li>
                <li>• Both outputs are scored using the RIVA quality rubric</li>
                <li>• The system with the higher score wins the duel</li>
                <li>• Quality delta shows the performance difference</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Mock data for development
const mockGoldenCases: GoldenCase[] = [
  {
    case_id: 'fintech_expansion_001',
    user_query: 'Analyze the strategic implications of expanding our fintech platform into the European market, considering regulatory challenges and competitive landscape.',
    expected_domain: 'strategy',
    expected_outcome_type: 'strategic_analysis'
  },
  {
    case_id: 'ai_integration_002',
    user_query: 'Evaluate the potential risks and benefits of integrating AI-powered decision making into our core business processes.',
    expected_domain: 'technology',
    expected_outcome_type: 'risk_assessment'
  },
  {
    case_id: 'sustainability_003',
    user_query: 'Develop a comprehensive sustainability strategy for our manufacturing operations that balances environmental impact with cost efficiency.',
    expected_domain: 'operations',
    expected_outcome_type: 'strategic_plan'
  }
];

const mockChallengers: ChallengerPrompt[] = [
  {
    prompt_id: '1',
    prompt_name: 'Strategic Analysis Monolith',
    target_station: 'FULL_PIPELINE',
    version: '1.0',
    status: 'active'
  },
  {
    prompt_id: '2',
    prompt_name: 'Quick Analysis Challenger',
    target_station: 'STATION_1',
    version: '1.0',
    status: 'active'
  }
];
