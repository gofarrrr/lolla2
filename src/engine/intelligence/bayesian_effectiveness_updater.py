"""Bayesian effectiveness updater module."""
from typing import Dict, Any

class BayesianEffectivenessUpdater:
    """Updates effectiveness scores using Bayesian inference."""
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.alpha = prior_alpha
        self.beta = prior_beta
        
    def update(self, success: bool) -> float:
        """Update effectiveness score with new observation."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1
        return self.alpha / (self.alpha + self.beta)

def get_bayesian_updater():
    """Get a BayesianEffectivenessUpdater instance."""
    return BayesianEffectivenessUpdater()

def run_bayesian_learning_cycle(updater: BayesianEffectivenessUpdater, success: bool) -> Dict[str, Any]:
    """Run a single Bayesian learning cycle."""
    score = updater.update(success)
    return {
        "score": score,
        "success": success,
        "alpha": updater.alpha,
        "beta": updater.beta
    }
