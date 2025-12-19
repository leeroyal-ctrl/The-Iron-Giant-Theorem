"""
Quick tests to verify modules work correctly
"""

import numpy as np
from long_term_memory import LongTermMemory, Episode, LessonPattern
from pattern_analyzer import PatternAnalyzer


def test_memory_storage():
    """Test basic memory storage and retrieval"""
    print("Testing Memory Storage...")
    
    memory = LongTermMemory("test_memory.db")
    
    # Store test episode
    episode = Episode(
        story_text="Test story",
        story_summary="Test",
        lesson="Test lesson",
        lesson_pattern_id="test_001",
        affected_dimensions=[0],
        keywords=["test"],
        anchor_before=np.array([6,4,3,7,5]),
        active_before=np.array([6,4,3,7,5]),
        buffer_before=np.array([6,4,3,7,5]),
        buffer_after=np.array([6.5,4,3,7,5]),
        alignment_with_anchor=0.95,
        alignment_with_active=0.95,
        dissonance_caused=0.5
    )
    
    episode_id = memory.store_episode(episode)
    assert episode_id > 0, "Failed to store episode"
    
    # Retrieve
    trajectory = memory.get_buffer_trajectory(0, 10)
    assert len(trajectory) > 0, "Failed to retrieve trajectory"
    
    print("✅ Memory storage test passed")


def test_pattern_analysis():
    """Test pattern detection"""
    print("Testing Pattern Analysis...")
    
    memory = LongTermMemory("test_patterns.db")
    analyzer = PatternAnalyzer(memory)
    
    # Test trend detection
    increasing = np.array([1, 2, 3, 4, 5, 6, 7])
    trend = analyzer.detect_trend(increasing)
    assert trend.direction == 'increasing', f"Expected increasing, got {trend.direction}"
    
    # Test equilibrium
    stable = np.array([5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0])
    eq = analyzer.detect_equilibrium(stable)
    assert eq.stability > 0.5, "Should detect stable equilibrium"
    
    print("✅ Pattern analysis test passed")


def test_deception_tracking():
    """Test pattern reliability tracking"""
    print("Testing Deception Detection...")
    
    memory = LongTermMemory("test_deception.db")
    
    # Register pattern
    pattern = LessonPattern(
        pattern_id="deceptive_test",
        description="Test deceptive pattern",
        category="test",
        affected_dimension=0,
        direction="increase"
    )
    memory.register_lesson_pattern(pattern)
    
    # Check history (should be new)
    history = memory.check_pattern_history("deceptive_test")
    assert history['is_new_pattern'], "Should be new pattern"
    
    # Simulate deception (would happen through evaluation)
    pattern.times_contradicted = 5
    pattern.times_validated = 1
    pattern.reliability_score = 1 / 6
    memory.register_lesson_pattern(pattern)
    
    # Check again
    history = memory.check_pattern_history("deceptive_test")
    assert history['trust_modifier'] < 0.5, "Should have low trust"
    assert history['warning'] is not None, "Should generate warning"
    
    print("✅ Deception detection test passed")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Running Integration Tests")
    print("="*50 + "\n")
    
    test_memory_storage()
    test_pattern_analysis()
    test_deception_tracking()
    
    print("\n" + "="*50)
    print("✅ All Tests Passed!")
    print("="*50 + "\n")
