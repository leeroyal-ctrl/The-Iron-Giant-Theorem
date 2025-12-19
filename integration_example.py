"""
DAA1 Integration Example
Part of: The Iron Giant Theorem Research Project

Demonstrates how WVM, Memory, and Pattern Analyzer work together
to process stories, learn lessons, detect deception, and update worldview.

This example shows the complete flow for Phase 1 implementation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from long_term_memory import (
    LongTermMemory, 
    Episode, 
    LessonPattern,
    generate_pattern_id,
    compute_alignment
)
from pattern_analyzer import (
    PatternAnalyzer,
    describe_trajectory
)


# ============================================================================
# SIMPLIFIED WVM (for demonstration)
# ============================================================================

class SimpleWVM:
    """
    Simplified Worldview Module for integration demo
    
    In production, use your full WVM specification.
    This version focuses on the integration points.
    """
    
    def __init__(self, 
                 initial_worldview: np.ndarray,
                 buffer_learning_rate: float = 0.3):
        """
        Initialize 3-tier worldview
        
        Args:
            initial_worldview: Starting values for all tiers [5 dimensions]
            buffer_learning_rate: How responsive Buffer is to new experiences
        """
        self.anchor = initial_worldview.copy()  # Immutable
        self.active = initial_worldview.copy()  # Slow-changing
        self.buffer = initial_worldview.copy()  # Fast-changing
        
        self.buffer_lr = buffer_learning_rate
        self.episode_count = 0
        self.active_update_count = 0
        
    def compute_variance(self) -> float:
        """Compute dissonance (variance between tiers)"""
        anchor_active_dist = np.sum((self.anchor - self.active) ** 2)
        active_buffer_dist = np.sum((self.active - self.buffer) ** 2)
        return anchor_active_dist + active_buffer_dist
        
    def update_buffer(self, new_assessment: np.ndarray):
        """Update Buffer tier based on new experience"""
        self.buffer = (1 - self.buffer_lr) * self.buffer + self.buffer_lr * new_assessment
        
    def update_active(self, dimension_id: int, new_value: float):
        """Update one dimension of Active tier"""
        self.active[dimension_id] = new_value
        self.active_update_count += 1
        
    def get_state(self) -> Dict:
        """Get current worldview state"""
        return {
            'anchor': self.anchor.copy(),
            'active': self.active.copy(),
            'buffer': self.buffer.copy(),
            'variance': self.compute_variance(),
            'episode_count': self.episode_count
        }


# ============================================================================
# STORY PROCESSOR (integrates all components)
# ============================================================================

class StoryProcessor:
    """
    Coordinates WVM, Memory, and Pattern Analyzer
    
    This is the "glue" that connects all components for story processing.
    """
    
    def __init__(self,
                 wvm: SimpleWVM,
                 memory: LongTermMemory,
                 analyzer: PatternAnalyzer,
                 dimension_names: List[str]):
        """
        Initialize processor
        
        Args:
            wvm: Worldview Module instance
            memory: Long-Term Memory instance
            analyzer: Pattern Analyzer instance
            dimension_names: Names of 5 SAWV dimensions
        """
        self.wvm = wvm
        self.memory = memory
        self.analyzer = analyzer
        self.dimension_names = dimension_names
        
    def process_story(self, 
                     story_text: str,
                     mock_lesson: Optional[Dict] = None) -> Dict:
        """
        Process a story through the complete pipeline
        
        Args:
            story_text: The narrative experience
            mock_lesson: For testing - pre-defined lesson (replaces LLM call)
            
        Returns:
            Processing results with all decisions and updates
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING STORY #{self.wvm.episode_count + 1}")
        print(f"{'='*70}")
        
        # Capture worldview before
        state_before = self.wvm.get_state()
        
        # STEP 1: Extract lesson from story (normally via LLM)
        if mock_lesson:
            lesson_data = mock_lesson
        else:
            lesson_data = self._extract_lesson_mock(story_text)
            
        print(f"\n📖 Story: {story_text[:100]}...")
        print(f"💡 Lesson: {lesson_data['lesson']}")
        print(f"📊 Affects dimensions: {[self.dimension_names[i] for i in lesson_data['affected_dimensions']]}")
        
        # STEP 2: Update Buffer based on lesson
        new_assessment = self._interpret_lesson_as_worldview(lesson_data)
        self.wvm.update_buffer(new_assessment)
        
        state_after = self.wvm.get_state()
        dissonance = state_after['variance']
        
        print(f"\n🌊 Buffer updated:")
        for i in lesson_data['affected_dimensions']:
            print(f"   {self.dimension_names[i]}: {state_before['buffer'][i]:.2f} → {state_after['buffer'][i]:.2f}")
        print(f"⚡ Dissonance: {dissonance:.3f}")
        
        # STEP 3: Create pattern ID and check history
        pattern_id = generate_pattern_id(
            lesson_data['lesson'],
            lesson_data['affected_dimensions'][0],
            lesson_data['direction']
        )
        
        # Register pattern if new
        if self.memory.check_pattern_history(pattern_id)['is_new_pattern']:
            pattern = LessonPattern(
                pattern_id=pattern_id,
                description=lesson_data['lesson'][:100],
                category=self.dimension_names[lesson_data['affected_dimensions'][0]],
                affected_dimension=lesson_data['affected_dimensions'][0],
                direction=lesson_data['direction']
            )
            self.memory.register_lesson_pattern(pattern)
            print(f"\n🆕 Registered new pattern: {pattern_id}")
        
        # CHECK FOR DECEPTION
        pattern_history = self.memory.check_pattern_history(pattern_id)
        if pattern_history['warning']:
            print(f"\n{pattern_history['warning']}")
            print(f"   📋 {pattern_history['recommendation']}")
        
        # STEP 4: Store episode in memory
        episode = Episode(
            story_text=story_text,
            story_summary=story_text[:200],
            lesson=lesson_data['lesson'],
            lesson_pattern_id=pattern_id,
            affected_dimensions=lesson_data['affected_dimensions'],
            keywords=lesson_data['keywords'],
            anchor_before=state_before['anchor'],
            active_before=state_before['active'],
            buffer_before=state_before['buffer'],
            buffer_after=state_after['buffer'],
            alignment_with_anchor=compute_alignment(state_after['buffer'], state_before['anchor']),
            alignment_with_active=compute_alignment(state_after['buffer'], state_before['active']),
            dissonance_caused=dissonance
        )
        
        episode_id = self.memory.store_episode(episode)
        self.wvm.episode_count += 1
        
        # STEP 5: Pattern analysis for Active updates
        update_recommendations = {}
        
        for dimension_id in lesson_data['affected_dimensions']:
            recommendation = self.analyzer.should_update_active(
                dimension_id=dimension_id,
                anchor_value=self.wvm.anchor[dimension_id],
                active_value=self.wvm.active[dimension_id],
                current_lesson_pattern=pattern_id,
                episodes_since_last_update=self.wvm.episode_count - self.wvm.active_update_count
            )
            
            update_recommendations[dimension_id] = recommendation
            
            print(f"\n🔍 Analysis for {self.dimension_names[dimension_id]}:")
            print(f"   Pattern: {recommendation.reasoning}")
            
            if recommendation.warnings:
                for warning in recommendation.warnings:
                    print(f"   {warning}")
            
            # STEP 6: Apply Active update if recommended
            if recommendation.should_update:
                old_value = self.wvm.active[dimension_id]
                self.wvm.update_active(dimension_id, recommendation.recommended_value)
                print(f"   ✅ ACTIVE UPDATED: {old_value:.2f} → {recommendation.recommended_value:.2f}")
                print(f"   Confidence: {recommendation.confidence:.2f}")
            else:
                print(f"   ⏸️  No update (criteria not met)")
                unmet = [k for k, v in recommendation.criteria_met.items() if not v]
                print(f"   Unmet: {', '.join(unmet)}")
        
        # STEP 7: Periodic pattern validation (every 10 episodes)
        if self.wvm.episode_count % 10 == 0:
            print(f"\n🔬 Evaluating pattern reliability...")
            for dim_id in range(5):
                # Get recent patterns for this dimension
                trajectory = self.memory.get_buffer_trajectory(dim_id, 20)
                if len(trajectory) > 10:
                    # Evaluate reliability
                    result = self.memory.evaluate_pattern_reliability(pattern_id, dim_id)
                    if 'reliability_score' in result:
                        print(f"   {self.dimension_names[dim_id]}: Reliability = {result['reliability_score']:.2f}")
        
        return {
            'episode_id': episode_id,
            'dissonance': dissonance,
            'pattern_id': pattern_id,
            'pattern_history': pattern_history,
            'update_recommendations': update_recommendations,
            'worldview_after': state_after
        }
        
    def _extract_lesson_mock(self, story_text: str) -> Dict:
        """
        Mock lesson extraction (replace with LLM in production)
        
        In real implementation, this would call GPT-4 to interpret
        the story and extract the lesson + affected dimensions.
        """
        # Simple keyword-based mock
        lesson_templates = {
            'authority': {
                'keywords': ['leader', 'authority', 'elder', 'chief', 'government'],
                'lesson': 'Authority figures can be trustworthy and act for common good',
                'dimension': 0,
                'direction': 'increase'
            },
            'tradition': {
                'keywords': ['tradition', 'ritual', 'ancient', 'custom', 'ceremony'],
                'lesson': 'Traditional practices provide stability and meaning',
                'dimension': 1,
                'direction': 'increase'
            },
            'equality': {
                'keywords': ['equal', 'fair', 'justice', 'rights', 'democracy'],
                'lesson': 'Equality benefits society and individuals',
                'dimension': 2,
                'direction': 'increase'
            },
            'harmony': {
                'keywords': ['harmony', 'balance', 'nature', 'peace', 'cooperation'],
                'lesson': 'Living in harmony with nature is beneficial',
                'dimension': 3,
                'direction': 'increase'
            },
            'autonomy': {
                'keywords': ['individual', 'freedom', 'choice', 'independent', 'liberty'],
                'lesson': 'Individual autonomy enables human flourishing',
                'dimension': 4,
                'direction': 'increase'
            }
        }
        
        # Detect which category
        story_lower = story_text.lower()
        for category, data in lesson_templates.items():
            if any(kw in story_lower for kw in data['keywords']):
                return {
                    'lesson': data['lesson'],
                    'affected_dimensions': [data['dimension']],
                    'direction': data['direction'],
                    'keywords': [kw for kw in data['keywords'] if kw in story_lower]
                }
        
        # Default
        return {
            'lesson': 'General life lesson from experience',
            'affected_dimensions': [0],
            'direction': 'increase',
            'keywords': ['general']
        }
        
    def _interpret_lesson_as_worldview(self, lesson_data: Dict) -> np.ndarray:
        """
        Convert lesson into worldview assessment
        
        In production, this might use LLM to map lesson → 5D vector.
        For now, simple rule-based interpretation.
        """
        # Start with current buffer
        new_assessment = self.wvm.buffer.copy()
        
        # Adjust affected dimensions
        for dim_id in lesson_data['affected_dimensions']:
            if lesson_data['direction'] == 'increase':
                # Move toward higher value (max 10)
                new_assessment[dim_id] = min(10, new_assessment[dim_id] + 1.5)
            else:
                # Move toward lower value (min 0)
                new_assessment[dim_id] = max(0, new_assessment[dim_id] - 1.5)
                
        return new_assessment
        
    def get_system_summary(self) -> Dict:
        """Get overall system state summary"""
        state = self.wvm.get_state()
        
        summary = {
            'episodes_processed': self.wvm.episode_count,
            'active_updates': self.wvm.active_update_count,
            'current_dissonance': state['variance'],
            'worldview': state,
            'dimension_analyses': {}
        }
        
        # Analyze each dimension
        for i, name in enumerate(self.dimension_names):
            analysis = self.analyzer.get_dimension_summary(i)
            summary['dimension_analyses'][name] = analysis
            
        return summary


# ============================================================================
# DEMONSTRATION SCENARIOS
# ============================================================================

def scenario_normal_learning():
    """Scenario 1: Normal learning without deception"""
    print("\n" + "="*70)
    print("SCENARIO 1: Normal Learning (No Deception)")
    print("="*70)
    
    # Initialize system
    initial_worldview = np.array([6.0, 4.0, 3.0, 7.0, 5.0])
    dimension_names = ['Authority', 'Tradition', 'Equality', 'Harmony', 'Autonomy']
    
    wvm = SimpleWVM(initial_worldview)
    memory = LongTermMemory("demo_normal.db")
    analyzer = PatternAnalyzer(memory)
    processor = StoryProcessor(wvm, memory, analyzer, dimension_names)
    
    # Process series of consistent stories about authority
    stories = [
        "A village elder made a wise decision that helped everyone in the community prosper.",
        "The chief fairly distributed resources during difficult times, ensuring no one went hungry.",
        "A leader stepped down peacefully when it was time, showing respect for democratic process.",
        "Authority figures worked with citizens to solve problems collaboratively.",
        "The government implemented policies that protected the vulnerable while maintaining order."
    ]
    
    for story in stories:
        processor.process_story(story)
        
    # Show final summary
    summary = processor.get_system_summary()
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Episodes: {summary['episodes_processed']}")
    print(f"Active Updates: {summary['active_updates']}")
    print(f"Dissonance: {summary['current_dissonance']:.3f}")
    print(f"\nAuthority dimension:")
    print(f"  Anchor: {summary['worldview']['anchor'][0]:.2f}")
    print(f"  Active: {summary['worldview']['active'][0]:.2f}")
    print(f"  Buffer: {summary['worldview']['buffer'][0]:.2f}")


def scenario_deception_detection():
    """Scenario 2: Pattern teaches one thing, then contradicts itself"""
    print("\n" + "="*70)
    print("SCENARIO 2: Deception Detection")
    print("="*70)
    
    initial_worldview = np.array([6.0, 4.0, 3.0, 7.0, 5.0])
    dimension_names = ['Authority', 'Tradition', 'Equality', 'Harmony', 'Autonomy']
    
    wvm = SimpleWVM(initial_worldview)
    memory = LongTermMemory("demo_deception.db")
    analyzer = PatternAnalyzer(memory)
    processor = StoryProcessor(wvm, memory, analyzer, dimension_names)
    
    # First: Stories suggesting authority is trustworthy
    print("\n--- PHASE 1: Building Trust in Authority ---")
    trust_stories = [
        "The leader made excellent decisions for the community.",
        "Authority figures proved themselves reliable and fair.",
        "The chief's wisdom benefited everyone equally."
    ]
    
    for story in trust_stories:
        processor.process_story(story)
        
    # Then: Stories showing authority is actually corrupt
    print("\n--- PHASE 2: Betrayal (Same Pattern, Opposite Result) ---")
    betrayal_stories = [
        "The same leader embezzled funds meant for the community.",
        "Authority figures abused their power for personal gain.",
        "The chief made decisions that only benefited the elite.",
        "Leaders lied about their intentions and exploited trust."
    ]
    
    for story in betrayal_stories:
        processor.process_story(story)
        
    # Now if we see another "trust authority" story
    print("\n--- PHASE 3: Testing Deception Detection ---")
    processor.process_story(
        "A new leader promises to serve the community with wisdom and fairness."
    )
    
    # Check pattern reliability
    print("\n--- PATTERN RELIABILITY ANALYSIS ---")
    patterns = memory.get_all_patterns()
    for p in patterns:
        print(f"\n{p['description'][:60]}...")
        print(f"  Encountered: {p['times_encountered']} times")
        print(f"  Reliability: {p['reliability_score']:.2%}")
        print(f"  Status: {'🚨 DECEPTIVE' if p['is_deceptive'] else '✅ Trustworthy'}")


def scenario_trajectory_types():
    """Scenario 3: Demonstrate different trajectory patterns"""
    print("\n" + "="*70)
    print("SCENARIO 3: Trajectory Pattern Classification")
    print("="*70)
    
    initial_worldview = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    dimension_names = ['Authority', 'Tradition', 'Equality', 'Harmony', 'Autonomy']
    
    wvm = SimpleWVM(initial_worldview)
    memory = LongTermMemory("demo_trajectories.db")
    analyzer = PatternAnalyzer(memory)
    processor = StoryProcessor(wvm, memory, analyzer, dimension_names)
    
    # Sustained shift pattern
    print("\n--- Testing: Sustained Shift Pattern ---")
    for i in range(15):
        processor.process_story(f"Story {i+1}: Authority proves increasingly trustworthy through consistent actions.")
        
    trajectory = analyzer.classify_trajectory(0, initial_worldview[0])
    print(f"\n📊 Trajectory: {trajectory.pattern_type}")
    print(f"   {describe_trajectory(trajectory)}")
    print(f"   Confidence: {trajectory.confidence:.2f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║           DAA1 INTEGRATION DEMONSTRATION                       ║
    ║           The Iron Giant Theorem Research Project              ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Run scenarios
    try:
        scenario_normal_learning()
        input("\n\nPress Enter to continue to Scenario 2...")
        
        scenario_deception_detection()
        input("\n\nPress Enter to continue to Scenario 3...")
        
        scenario_trajectory_types()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        
    print("""
    
    ╔════════════════════════════════════════════════════════════════╗
    ║                    DEMONSTRATION COMPLETE                      ║
    ║                                                                ║
    ║  ✅ Memory Module: Storing episodes & tracking patterns       ║
    ║  ✅ Pattern Analyzer: Classifying trajectories                ║
    ║  ✅ Trust System: Detecting deceptive patterns                ║
    ║  ✅ Integration: Complete story processing pipeline           ║
    ║                                                                ║
    ║  Next steps:                                                   ║
    ║  - Integrate with your full WVM specification                 ║
    ║  - Add LLM for real lesson extraction                         ║
    ║  - Connect to ASM and Dissonance Register                     ║
    ║  - Implement DRM action selection                             ║
    ╚════════════════════════════════════════════════════════════════╝
    """) 
