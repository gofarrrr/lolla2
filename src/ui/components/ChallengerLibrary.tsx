import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from '@/components/ui/dialog';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, 
  Plus, 
  Edit, 
  Trash2, 
  MoreVertical, 
  Code2,
  Target,
  Calendar,
  Zap
} from 'lucide-react';

// Types
interface ChallengerPrompt {
  prompt_id: string;
  prompt_name: string;
  prompt_text: string;
  version: string;
  status: 'active' | 'archived' | 'draft';
  target_station: string;
  golden_case_id?: string;
  compilation_metadata?: any;
  created_at: string;
  updated_at: string;
}

interface ChallengerLibraryProps {
  onSelectChallenger: (challengerId: string) => void;
  onCompileMonolith: (goldenCaseId: string) => void;
  onLaunchDuel: (config: DuelConfig) => void;
}

interface DuelConfig {
  goldenCaseId: string;
  challengerPromptId: string;
  stationToTest: string;
}

const STATION_OPTIONS = [
  { value: 'FULL_PIPELINE', label: 'Full Pipeline', description: 'Complete 8-station analysis' },
  { value: 'STATION_1', label: 'QUICKTHINK', description: 'Initial rapid analysis' },
  { value: 'STATION_2', label: 'DEEPTHINK', description: 'Comprehensive exploration' },
  { value: 'STATION_3', label: 'BLUETHINK', description: 'Conservative analysis' },
  { value: 'STATION_4', label: 'REDTHINK', description: 'Bold innovation' },
  { value: 'STATION_5', label: 'GREYTHINK', description: 'Reality check' },
  { value: 'STATION_6', label: 'ULTRATHINK', description: 'Deep synthesis' },
  { value: 'STATION_7', label: 'DIVERGENTTHINK', description: 'Alternative perspectives' },
  { value: 'STATION_8', label: 'CONVERGENTTHINK', description: 'Final integration' }
];

const STATUS_COLORS = {
  active: 'bg-green-100 text-green-800',
  draft: 'bg-yellow-100 text-yellow-800',
  archived: 'bg-gray-100 text-gray-800'
};

export const ChallengerLibrary: React.FC<ChallengerLibraryProps> = ({
  onSelectChallenger,
  onCompileMonolith,
  onLaunchDuel
}) => {
  const [challengers, setChallengers] = useState<ChallengerPrompt[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStation, setSelectedStation] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('active');
  const [searchTerm, setSearchTerm] = useState('');
  
  // Dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isCompileDialogOpen, setIsCompileDialogOpen] = useState(false);
  const [selectedChallenger, setSelectedChallenger] = useState<ChallengerPrompt | null>(null);

  // Form states
  const [newChallenger, setNewChallenger] = useState({
    prompt_name: '',
    prompt_text: '',
    version: '1.0',
    target_station: 'FULL_PIPELINE',
    golden_case_id: ''
  });

  const [compileConfig, setCompileConfig] = useState({
    golden_case_id: '',
    prompt_name: '',
    version: '1.0'
  });

  // Load challengers
  useEffect(() => {
    loadChallengers();
  }, [selectedStation, selectedStatus]);

  const loadChallengers = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (selectedStation !== 'all') params.set('target_station', selectedStation);
      if (selectedStatus !== 'all') params.set('status', selectedStatus);
      
      const response = await fetch(`/api/proving-ground/challenger-prompts?${params}`);
      if (response.ok) {
        const data = await response.json();
        setChallengers(data);
      } else {
        console.error('Failed to load challengers');
        // Mock data for development
        setChallengers(mockChallengers);
      }
    } catch (error) {
      console.error('Error loading challengers:', error);
      setChallengers(mockChallengers);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateChallenger = async () => {
    try {
      const response = await fetch('/api/proving-ground/challenger-prompts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...newChallenger,
          status: 'draft'
        }),
      });

      if (response.ok) {
        setIsCreateDialogOpen(false);
        setNewChallenger({
          prompt_name: '',
          prompt_text: '',
          version: '1.0',
          target_station: 'FULL_PIPELINE',
          golden_case_id: ''
        });
        loadChallengers();
      }
    } catch (error) {
      console.error('Error creating challenger:', error);
    }
  };

  const handleCompileMonolith = async () => {
    try {
      const response = await fetch('/api/proving-ground/compile-monolith', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(compileConfig),
      });

      if (response.ok) {
        setIsCompileDialogOpen(false);
        setCompileConfig({
          golden_case_id: '',
          prompt_name: '',
          version: '1.0'
        });
        loadChallengers();
        onCompileMonolith(compileConfig.golden_case_id);
      }
    } catch (error) {
      console.error('Error compiling monolith:', error);
    }
  };

  const handleDeleteChallenger = async (promptId: string) => {
    if (confirm('Are you sure you want to delete this challenger?')) {
      try {
        const response = await fetch(`/api/proving-ground/challenger-prompts/${promptId}`, {
          method: 'DELETE',
        });

        if (response.ok) {
          loadChallengers();
        }
      } catch (error) {
        console.error('Error deleting challenger:', error);
      }
    }
  };

  const filteredChallengers = challengers.filter(challenger =>
    challenger.prompt_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    challenger.prompt_text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStationLabel = (station: string) => {
    const option = STATION_OPTIONS.find(opt => opt.value === station);
    return option ? option.label : station;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Challenger Library</h2>
          <p className="text-gray-600">Manage monolithic prompt challengers for proving ground duels</p>
        </div>
        
        <div className="flex space-x-2">
          <Dialog open={isCompileDialogOpen} onOpenChange={setIsCompileDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" className="flex items-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>Compile Monolith</span>
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Compile Perfect Monolith</DialogTitle>
                <DialogDescription>
                  Generate a monolithic challenger by extracting cognitive DNA from all 8 stations
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Golden Case ID</label>
                  <Input
                    value={compileConfig.golden_case_id}
                    onChange={(e) => setCompileConfig({...compileConfig, golden_case_id: e.target.value})}
                    placeholder="e.g. fintech_expansion_001"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Prompt Name (Optional)</label>
                  <Input
                    value={compileConfig.prompt_name}
                    onChange={(e) => setCompileConfig({...compileConfig, prompt_name: e.target.value})}
                    placeholder="Auto-generated if empty"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Version</label>
                  <Input
                    value={compileConfig.version}
                    onChange={(e) => setCompileConfig({...compileConfig, version: e.target.value})}
                  />
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsCompileDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCompileMonolith}>
                  <Zap className="w-4 h-4 mr-2" />
                  Compile
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button className="flex items-center space-x-2">
                <Plus className="w-4 h-4" />
                <span>New Challenger</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Create New Challenger</DialogTitle>
                <DialogDescription>
                  Create a custom monolithic prompt to challenge the Lolla pipeline
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Prompt Name</label>
                  <Input
                    value={newChallenger.prompt_name}
                    onChange={(e) => setNewChallenger({...newChallenger, prompt_name: e.target.value})}
                    placeholder="Descriptive name for the challenger"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Target Station</label>
                  <Select
                    value={newChallenger.target_station}
                    onValueChange={(value) => setNewChallenger({...newChallenger, target_station: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {STATION_OPTIONS.map(option => (
                        <SelectItem key={option.value} value={option.value}>
                          <div>
                            <div className="font-medium">{option.label}</div>
                            <div className="text-sm text-gray-500">{option.description}</div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Prompt Text</label>
                  <Textarea
                    value={newChallenger.prompt_text}
                    onChange={(e) => setNewChallenger({...newChallenger, prompt_text: e.target.value})}
                    placeholder="Enter the complete prompt text..."
                    rows={8}
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Version</label>
                    <Input
                      value={newChallenger.version}
                      onChange={(e) => setNewChallenger({...newChallenger, version: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Golden Case ID (Optional)</label>
                    <Input
                      value={newChallenger.golden_case_id}
                      onChange={(e) => setNewChallenger({...newChallenger, golden_case_id: e.target.value})}
                      placeholder="Associated golden case"
                    />
                  </div>
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateChallenger}>
                  Create Challenger
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 p-4 bg-gray-50 rounded-lg">
        <div className="flex-1 min-w-64">
          <Input
            placeholder="Search challengers..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <Select value={selectedStation} onValueChange={setSelectedStation}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Filter by station" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Stations</SelectItem>
            {STATION_OPTIONS.map(option => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        <Select value={selectedStatus} onValueChange={setSelectedStatus}>
          <SelectTrigger className="w-32">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="draft">Draft</SelectItem>
            <SelectItem value="archived">Archived</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Challengers Grid */}
      {loading ? (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading challengers...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredChallengers.map((challenger) => (
            <Card key={challenger.prompt_id} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <CardTitle className="text-lg">{challenger.prompt_name}</CardTitle>
                    <CardDescription className="flex items-center space-x-2 mt-1">
                      <Target className="w-3 h-3" />
                      <span>{getStationLabel(challenger.target_station)}</span>
                      <span className="text-gray-300">•</span>
                      <span>v{challenger.version}</span>
                    </CardDescription>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Badge className={STATUS_COLORS[challenger.status]}>
                      {challenger.status}
                    </Badge>
                    
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm">
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={() => onSelectChallenger(challenger.prompt_id)}
                        >
                          <Play className="w-4 h-4 mr-2" />
                          Launch Duel
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => {
                            setSelectedChallenger(challenger);
                            setIsEditDialogOpen(true);
                          }}
                        >
                          <Edit className="w-4 h-4 mr-2" />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => handleDeleteChallenger(challenger.prompt_id)}
                          className="text-red-600"
                        >
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-3">
                <div className="text-sm text-gray-600 line-clamp-3">
                  {challenger.prompt_text}
                </div>
                
                <div className="flex justify-between items-center text-xs text-gray-500">
                  <div className="flex items-center space-x-1">
                    <Calendar className="w-3 h-3" />
                    <span>{new Date(challenger.created_at).toLocaleDateString()}</span>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    <Code2 className="w-3 h-3" />
                    <span>{challenger.prompt_text.length} chars</span>
                  </div>
                </div>
                
                {challenger.compilation_metadata?.compilation_type === 'automated' && (
                  <Badge variant="outline" className="text-xs">
                    <Zap className="w-3 h-3 mr-1" />
                    Auto-compiled
                  </Badge>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {filteredChallengers.length === 0 && !loading && (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <Target className="w-16 h-16 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No challengers found</h3>
          <p className="text-gray-600 mb-4">
            {searchTerm ? 'No challengers match your search.' : 'Create your first challenger to get started.'}
          </p>
          {!searchTerm && (
            <Button onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Create New Challenger
            </Button>
          )}
        </div>
      )}
    </div>
  );
};

// Mock data for development
const mockChallengers: ChallengerPrompt[] = [
  {
    prompt_id: '1',
    prompt_name: 'Strategic Analysis Monolith',
    prompt_text: 'You are a comprehensive strategic analysis system. Analyze the following input considering market dynamics, competitive landscape, risk factors, innovation opportunities, and provide actionable recommendations.',
    version: '1.0',
    status: 'active',
    target_station: 'FULL_PIPELINE',
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-15T10:00:00Z'
  },
  {
    prompt_id: '2',
    prompt_name: 'Quick Analysis Challenger',
    prompt_text: 'Perform rapid analysis of the input, identifying key themes and immediate insights.',
    version: '1.0',
    status: 'draft',
    target_station: 'STATION_1',
    created_at: '2024-01-14T15:30:00Z',
    updated_at: '2024-01-14T15:30:00Z'
  }
];
