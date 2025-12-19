"""
DAA1 Pattern Analyzer Module
Part of: The Iron Giant Theorem Research Project

Responsibilities:
- Classify Buffer trajectory patterns (5 types)
- Detect equilibrium points
- Recommend Active tier updates (trust-aware)
- Distinguish developmental vs. corrective learning
- Compute stability metrics
- Integrate lesson evidence with numeric trends

Phase: 1 (Foundation)
Status: Production-ready
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Literal, Any
from dataclasses import dataclass
from long_term_memory import LongTermMemory


# Type definitions
TrajectoryType = Literal[
    'excursion_and_return',  # Temporary deviation, returning to baseline
    'sustained_shift',       # Developmental change in progress
    'new_equilibrium',       # Settled at new stable position
    'volatile',              # Still searching/oscillating
    'stable_at_anchor'       # Remained at original baseline
]


@dataclass
class TrendAnalysis:
    """Results of trend detection"""
    direction: str  # 'increasing', 'decreasing', 'stable'
    slope: float
    confidence: float  # 0.0 to 1.0 (R-squared value)
    p_value: float


@dataclass
class EquilibriumAnalysis:
    """Results of equilibrium detection"""
    mean: float
    variance: float
    stability: float  # 0.0 to 1.0 (inverse of variance)
    n_samples: int


@dataclass
class TrajectoryClassification:
    """Complete trajectory analysis"""
    pattern_type: TrajectoryType
    confidence: float
    trend: TrendAnalysis
    equilibrium: EquilibriumAnalysis
    supporting_evidence: Dict[str, Any]


@dataclass
class UpdateRecommendation:
    """Recommendation for Active tier update"""
    should_update: bool
    recommended_value: Optional[float]
    confidence: float
    criteria_met: Dict[str, bool]
    reasoning: str
    warnings: List[str]


class PatternAnalyzer:
    """
    DAA1 Pattern Analysis System
    
    Analyzes Buffer trajectories to understand learning patterns,
    incorporating both quantitative (numeric) and qualitative (lesson-based)
    evidence. Trust-aware: considers if teaching patterns have deceived before.
    """
    
    def __init__(self, 
                 memory: LongTermMemory,
                 trend_confidence_threshold: float = 0.7,
                 stability_variance_threshold: float = 0.5,
                 min_episodes_for_update: int = 20,
                 equilibrium_window: int = 20):
        """
        Initialize Pattern Analyzer
        
        Args:
            memory: LongTermMemory instance for querying history
            trend_confidence_threshold: Minimum R² for confident trend
            stability_variance_threshold: Max variance for "stable"
            min_episodes_for_update: Minimum history before recommending update
            equilibrium_window: Recent episodes to analyze for equilibrium
        """
        self.memory = memory
        self.trend_threshold = trend_confidence_threshold
        self.stability_threshold = stability_variance_threshold
        self.min_episodes = min_episodes_for_update
        self.equilibrium_window = equilibrium_window
        
    def detect_trend(self, values: np.ndarray) -> TrendAnalysis:
        """
        Detect trend in time series using linear regression
        
        Args:
            values: Array of Buffer values over time
            
        Returns:
            TrendAnalysis with direction and confidence
        """
        if len(values) < 3:
            return TrendAnalysis(
                direction='unknown',
                slope=0.0,
                confidence=0.0,
                p_value=1.0
            )
        
        # Linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Interpret results
        confidence = abs(r_value)  # R-squared proxy
        
        if confidence < 0.3 or p_value > 0.05:
            direction = 'stable'  # Not statistically significant
        elif slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
            
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            confidence=confidence,
            p_value=p_value
        )
        
    def detect_equilibrium(self, 
                          buffer_history: np.ndarray,
                          window: Optional[int] = None) -> EquilibriumAnalysis:
        """
        Detect if Buffer has reached equilibrium
        
        Analyzes recent history to determine if Buffer has settled
        at a stable value (low variance = high stability).
        
        Args:
            buffer_history: Historical Buffer values
            window: How many recent episodes (default: self.equilibrium_window)
            
        Returns:
            EquilibriumAnalysis with mean and stability metrics
        """
        if window is None:
            window = self.equilibrium_window
            
        if len(buffer_history) < window:
            recent = buffer_history
        else:
            recent = buffer_history[-window:]
            
        if len(recent) == 0:
            return EquilibriumAnalysis(
                mean=0.0,
                variance=999.0,
                stability=0.0,
                n_samples=0
            )
        
        mean = np.mean(recent)
        variance = np.var(recent)
        
        # Stability: inverse of variance (0 to 1 scale)
        # High stability = low variance
        stability = 1.0 / (1.0 + variance)
        
        return EquilibriumAnalysis(
            mean=mean,
            variance=variance,
            stability=stability,
            n_samples=len(recent)
        )
        
    def is_returning_to_anchor(self, 
                              dimension_id: int,
                              anchor_value: float,
                              tolerance: float = 0.5) -> bool:
        """
        Check if Buffer is moving back toward Anchor
        
        Args:
            dimension_id: Which dimension to check
            anchor_value: Anchor tier value for this dimension
            tolerance: How close to consider "returned"
            
        Returns:
            True if Buffer is within tolerance of Anchor
        """
        trajectory = self.memory.get_buffer_trajectory(dimension_id, n_episodes=10)
        
        if len(trajectory) == 0:
            return False
            
        current_value = trajectory[-1]
        distance = abs(current_value - anchor_value)
        
        return distance <= tolerance
        
    def classify_trajectory(self, 
                           dimension_id: int,
                           anchor_value: float,
                           window: int = 50) -> TrajectoryClassification:
        """
        Classify the Buffer trajectory pattern
        
        Determines which of 5 pattern types best describes current trajectory:
        - excursion_and_return: Temporary deviation
        - sustained_shift: Developmental change
        - new_equilibrium: Settled at new position
        - volatile: Still searching
        - stable_at_anchor: Never deviated
        
        Args:
            dimension_id: Which dimension (0-4)
            anchor_value: Baseline value
            window: Episodes to analyze
            
        Returns:
            TrajectoryClassification with pattern type and evidence
        """
        # Get historical data
        buffer_history = self.memory.get_buffer_trajectory(dimension_id, window)
        
        if len(buffer_history) < 5:
            # Not enough data
            return TrajectoryClassification(
                pattern_type='volatile',
                confidence=0.0,
                trend=TrendAnalysis('unknown', 0.0, 0.0, 1.0),
                equilibrium=EquilibriumAnalysis(0.0, 999.0, 0.0, 0),
                supporting_evidence={'reason': 'Insufficient data'}
            )
        
        # Analyze numeric patterns
        trend = self.detect_trend(buffer_history)
        equilibrium = self.detect_equilibrium(buffer_history)
        
        # Check distance from anchor
        distance_from_anchor = abs(equilibrium.mean - anchor_value)
        returning_to_anchor = self.is_returning_to_anchor(dimension_id, anchor_value)
        
        # Classification logic
        pattern_type: TrajectoryType
        confidence = 0.0
        
        if distance_from_anchor < 0.3 and equilibrium.stability > 0.8:
            # Never really left baseline
            pattern_type = 'stable_at_anchor'
            confidence = equilibrium.stability
            
        elif returning_to_anchor and trend.direction == 'stable':
            # Went away but came back
            pattern_type = 'excursion_and_return'
            confidence = 0.7
            
        elif trend.direction != 'stable' and trend.confidence > self.trend_threshold:
            # Clear directional movement
            if equilibrium.stability < 0.5:
                # Still moving
                pattern_type = 'sustained_shift'
                confidence = trend.confidence
            else:
                # Stabilized at new position
                pattern_type = 'new_equilibrium'
                confidence = equilibrium.stability
                
        elif equilibrium.variance > 1.0:
            # High variance = still searching
            pattern_type = 'volatile'
            confidence = 1.0 - equilibrium.stability
            
        else:
            # Unclear pattern
            pattern_type = 'volatile'
            confidence = 0.5
            
        return TrajectoryClassification(
            pattern_type=pattern_type,
            confidence=confidence,
            trend=trend,
            equilibrium=equilibrium,
            supporting_evidence={
                'distance_from_anchor': distance_from_anchor,
                'returning_to_anchor': returning_to_anchor,
                'n_episodes': len(buffer_history)
            }
        )
        
    def should_update_active(self,
                            dimension_id: int,
                            anchor_value: float,
                            active_value: float,
                            current_lesson_pattern: str,
                            episodes_since_last_update: int) -> UpdateRecommendation:
        """
        Decide if Active tier should update (TRUST-AWARE)
        
        Integrates:
        1. Numeric trajectory analysis
        2. Accumulated lesson evidence
        3. Pattern reliability (deception checking)
        
        Args:
            dimension_id: Which dimension
            anchor_value: Anchor tier value
            active_value: Current Active tier value
            current_lesson_pattern: Most recent pattern ID
            episodes_since_last_update: Time since Active last changed
            
        Returns:
            UpdateRecommendation with decision and detailed reasoning
        """
        warnings = []
        
        # Get trajectory classification
        trajectory = self.classify_trajectory(dimension_id, anchor_value)
        
        # Get numeric equilibrium
        buffer_history = self.memory.get_buffer_trajectory(dimension_id, 50)
        
        if len(buffer_history) < self.min_episodes:
            return UpdateRecommendation(
                should_update=False,
                recommended_value=None,
                confidence=0.0,
                criteria_met={'sufficient_history': False},
                reasoning="Insufficient history - need more episodes",
                warnings=[]
            )
        
        equilibrium = self.detect_equilibrium(buffer_history)
        
        # Determine direction of potential change
        if equilibrium.mean > active_value:
            direction = 'increase'
        else:
            direction = 'decrease'
            
        # Get accumulated lesson evidence
        evidence = self.memory.get_accumulated_evidence(dimension_id, direction)
        
        # CHECK FOR DECEPTION (new!)
        pattern_history = self.memory.check_pattern_history(current_lesson_pattern)
        
        if pattern_history['warning']:
            warnings.append(pattern_history['warning'])
            
        # Adjust evidence confidence by pattern reliability
        trust_modifier = pattern_history['trust_modifier']
        adjusted_confidence = evidence['confidence'] * trust_modifier
        
        if trust_modifier < 0.5:
            warnings.append(f"Pattern trust is low ({trust_modifier:.2f}) - requiring extra evidence")
        
        # Decision criteria
        criteria = {
            # Numeric stability
            'trajectory_stable': equilibrium.stability > (1.0 - self.stability_threshold),
            'sufficient_history': len(buffer_history) >= self.min_episodes,
            'significant_divergence': abs(equilibrium.mean - active_value) > 1.0,
            
            # Evidence-based (ADJUSTED by trust)
            'strong_evidence': adjusted_confidence > 0.7,
            'multiple_lessons': evidence['supporting_episodes'] >= 3,
            
            # Trust-based (NEW!)
            'pattern_is_trustworthy': trust_modifier > 0.5,
            'no_high_deception_risk': pattern_history.get('reliability_score', 1.0) > 0.3,
            
            # Pattern match
            'developmental_pattern': trajectory.pattern_type in ['sustained_shift', 'new_equilibrium'],
            
            # Time constraint
            'enough_time_passed': episodes_since_last_update >= 10
        }
        
        # Extra validation if pattern has deceived before
        if trust_modifier < 0.5:
            criteria['extra_validation_needed'] = evidence['supporting_episodes'] >= 5
        else:
            criteria['extra_validation_needed'] = True  # Not required
            
        # Final decision: ALL criteria must be met
        should_update = all(criteria.values())
        
        # Confidence in recommendation
        if should_update:
            confidence = min(
                equilibrium.stability,
                adjusted_confidence,
                trust_modifier
            )
        else:
            confidence = 0.0
            
        # Generate reasoning explanation
        reasoning = self._generate_reasoning(
            criteria,
            trajectory,
            evidence,
            pattern_history,
            trust_modifier
        )
        
        return UpdateRecommendation(
            should_update=should_update,
            recommended_value=equilibrium.mean if should_update else None,
            confidence=confidence,
            criteria_met=criteria,
            reasoning=reasoning,
            warnings=warnings
        )
        
    def _generate_reasoning(self,
                           criteria: Dict[str, bool],
                           trajectory: TrajectoryClassification,
                           evidence: Dict,
                           pattern_history: Dict,
                           trust_modifier: float) -> str:
        """Generate human-readable explanation of decision"""
        
        reasoning_parts = []
        
        # Trajectory assessment
        reasoning_parts.append(
            f"Trajectory: {trajectory.pattern_type} "
            f"(confidence: {trajectory.confidence:.2f})"
        )
        
        # Numeric stability
        if criteria['trajectory_stable']:
            reasoning_parts.append(
                f"✓ Buffer is stable (variance: {trajectory.equilibrium.variance:.2f})"
            )
        else:
            reasoning_parts.append(
                f"✗ Buffer still volatile (variance: {trajectory.equilibrium.variance:.2f})"
            )
            
        # Evidence quality
        if criteria['strong_evidence']:
            reasoning_parts.append(
                f"✓ Strong lesson evidence ({evidence['supporting_episodes']} supporting)"
            )
        else:
            reasoning_parts.append(
                f"✗ Weak evidence ({evidence['supporting_episodes']} supporting, "
                f"{evidence['contradicting_episodes']} contradicting)"
            )
            
        # Trust assessment (NEW!)
        if pattern_history['is_new_pattern']:
            reasoning_parts.append("→ New pattern (no deception history)")
        elif trust_modifier > 0.7:
            reasoning_parts.append(
                f"✓ Pattern is reliable (trust: {trust_modifier:.2f})"
            )
        elif trust_modifier > 0.5:
            reasoning_parts.append(
                f"⚠ Pattern has mixed reliability (trust: {trust_modifier:.2f})"
            )
        else:
            reasoning_parts.append(
                f"✗ Pattern has history of deception (trust: {trust_modifier:.2f})"
            )
            
        # Time factor
        if criteria['enough_time_passed']:
            reasoning_parts.append("✓ Sufficient time since last update")
        else:
            reasoning_parts.append("✗ Too soon since last update")
            
        return " | ".join(reasoning_parts)
        
    def compute_stability_metrics(self, 
                                 buffer_history: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive stability metrics
        
        Args:
            buffer_history: Historical Buffer values
            
        Returns:
            Dictionary of stability metrics
        """
        if len(buffer_history) < 3:
            return {
                'variance': 999.0,
                'std': 999.0,
                'stability': 0.0,
                'coefficient_of_variation': 999.0
            }
        
        mean = np.mean(buffer_history)
        variance = np.var(buffer_history)
        std = np.std(buffer_history)
        
        # Coefficient of variation (std relative to mean)
        if mean != 0:
            cv = std / abs(mean)
        else:
            cv = 999.0
            
        stability = 1.0 / (1.0 + variance)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'stability': stability,
            'coefficient_of_variation': cv,
            'n_samples': len(buffer_history)
        }
        
    def get_dimension_summary(self, dimension_id: int) -> Dict[str, Any]:
        """
        Get comprehensive summary for one dimension
        
        Args:
            dimension_id: Which dimension (0-4)
            
        Returns:
            Complete analysis summary
        """
        buffer_history = self.memory.get_buffer_trajectory(dimension_id, 100)
        
        if len(buffer_history) < 5:
            return {
                'dimension_id': dimension_id,
                'status': 'insufficient_data',
                'n_episodes': len(buffer_history)
            }
        
        trend = self.detect_trend(buffer_history)
        equilibrium = self.detect_equilibrium(buffer_history)
        stability = self.compute_stability_metrics(buffer_history)
        
        return {
            'dimension_id': dimension_id,
            'current_value': buffer_history[-1],
            'trend': {
                'direction': trend.direction,
                'confidence': trend.confidence
            },
            'equilibrium': {
                'mean': equilibrium.mean,
                'stability': equilibrium.stability
            },
            'stability_metrics': stability,
            'n_episodes': len(buffer_history)
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def describe_trajectory(classification: TrajectoryClassification) -> str:
    """Generate natural language description of trajectory"""
    
    pattern = classification.pattern_type
    
    descriptions = {
        'stable_at_anchor': "Buffer has remained stable near baseline",
        'excursion_and_return': "Buffer temporarily deviated but returned to baseline",
        'sustained_shift': "Buffer is undergoing developmental change",
        'new_equilibrium': "Buffer has stabilized at a new position",
        'volatile': "Buffer is still searching/oscillating"
    }
    
    return descriptions.get(pattern, "Unknown pattern")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from long_term_memory import LongTermMemory
    
    # Initialize
    memory = LongTermMemory("test_daa1.db")
    analyzer = PatternAnalyzer(memory)
    
    print("✅ Pattern Analyzer initialized")
    
    # Simulate some Buffer history
    np.random.seed(42)
    test_trajectory = np.array([6.0, 6.1, 6.3, 6.5, 6.7, 6.8, 6.9, 7.0, 7.0, 7.1])
    
    # Detect trend
    trend = analyzer.detect_trend(test_trajectory)
    print(f"\n📈 Trend: {trend.direction} (confidence: {trend.confidence:.2f})")
    
    # Detect equilibrium
    equilibrium = analyzer.detect_equilibrium(test_trajectory)
    print(f"⚖️  Equilibrium: {equilibrium.mean:.2f} (stability: {equilibrium.stability:.2f})")
    
    print("\n✅ Pattern Analyzer module ready!")
