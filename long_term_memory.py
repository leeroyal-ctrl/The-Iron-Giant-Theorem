"""
DAA1 Long-Term Memory Module
Part of: The Iron Giant Theorem Research Project

Responsibilities:
- Store episode experiences
- Track lesson patterns and reliability
- Detect deception (patterns that mislead)
- Provide historical queries for Pattern Analyzer
- Support trust-aware decision making

Phase: 1 (Minimal - SQLite only)
Status: Production-ready
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Episode:
    """Single experience record"""
    story_text: str
    story_summary: str
    lesson: str
    lesson_pattern_id: str
    affected_dimensions: List[int]
    keywords: List[str]
    
    # Worldview states
    anchor_before: np.ndarray
    active_before: np.ndarray
    buffer_before: np.ndarray
    buffer_after: np.ndarray
    
    # Metrics
    alignment_with_anchor: float
    alignment_with_active: float
    dissonance_caused: float
    
    # Resolution
    action_taken: Optional[str] = None
    outcome: Optional[str] = None
    
    # Metadata
    timestamp: Optional[str] = None
    episode_id: Optional[int] = None


@dataclass
class LessonPattern:
    """Tracks a recurring teaching pattern"""
    pattern_id: str
    description: str
    category: str  # SAWV dimension name
    affected_dimension: int
    direction: str  # 'increase' or 'decrease'
    
    # Reliability tracking
    times_encountered: int = 0
    times_validated: int = 0
    times_contradicted: int = 0
    
    # Computed metrics
    reliability_score: float = 0.5  # Neutral until proven
    last_encounter_episode: Optional[int] = None


class LongTermMemory:
    """
    DAA1 Long-Term Memory System
    
    Stores experiences and tracks lesson pattern reliability over time.
    Enables trust-aware learning by detecting when patterns deceive.
    """
    
    def __init__(self, db_path: str = "daa1_memory.db", validation_window: int = 10):
        """
        Initialize memory system
        
        Args:
            db_path: SQLite database file path
            validation_window: Episodes to look ahead when validating patterns
        """
        self.db_path = db_path
        self.validation_window = validation_window
        self._init_database()
        
    def _init_database(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                story_text TEXT NOT NULL,
                story_summary TEXT,
                lesson TEXT NOT NULL,
                lesson_pattern_id TEXT,
                affected_dimensions TEXT NOT NULL,
                keywords TEXT NOT NULL,
                
                -- Worldview states (JSON arrays)
                anchor_before TEXT NOT NULL,
                active_before TEXT NOT NULL,
                buffer_before TEXT NOT NULL,
                buffer_after TEXT NOT NULL,
                
                -- Metrics
                alignment_anchor REAL NOT NULL,
                alignment_active REAL NOT NULL,
                dissonance REAL NOT NULL,
                
                -- Resolution
                action_taken TEXT,
                outcome TEXT
            )
        ''')
        
        # Lesson patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lesson_patterns (
                pattern_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                affected_dimension INTEGER NOT NULL,
                direction TEXT NOT NULL,
                
                times_encountered INTEGER DEFAULT 0,
                times_validated INTEGER DEFAULT 0,
                times_contradicted INTEGER DEFAULT 0,
                reliability_score REAL DEFAULT 0.5,
                
                last_encounter_episode INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern ON episodes(lesson_pattern_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dimensions ON episodes(affected_dimensions)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)')
        
        conn.commit()
        conn.close()
        
    def store_episode(self, episode: Episode) -> int:
        """
        Store new episode experience
        
        Args:
            episode: Episode data to store
            
        Returns:
            episode_id: ID of stored episode
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if episode.timestamp is None:
            episode.timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO episodes (
                timestamp, story_text, story_summary, lesson, lesson_pattern_id,
                affected_dimensions, keywords,
                anchor_before, active_before, buffer_before, buffer_after,
                alignment_anchor, alignment_active, dissonance,
                action_taken, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            episode.timestamp,
            episode.story_text,
            episode.story_summary,
            episode.lesson,
            episode.lesson_pattern_id,
            json.dumps(episode.affected_dimensions),
            json.dumps(episode.keywords),
            json.dumps(episode.anchor_before.tolist()),
            json.dumps(episode.active_before.tolist()),
            json.dumps(episode.buffer_before.tolist()),
            json.dumps(episode.buffer_after.tolist()),
            episode.alignment_with_anchor,
            episode.alignment_with_active,
            episode.dissonance_caused,
            episode.action_taken,
            episode.outcome
        ))
        
        episode_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update pattern tracking
        self._update_pattern_encounter(episode.lesson_pattern_id, episode_id)
        
        return episode_id
        
    def _update_pattern_encounter(self, pattern_id: str, episode_id: int):
        """Increment encounter count for pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE lesson_patterns
            SET times_encountered = times_encountered + 1,
                last_encounter_episode = ?,
                updated_at = ?
            WHERE pattern_id = ?
        ''', (episode_id, datetime.now().isoformat(), pattern_id))
        
        conn.commit()
        conn.close()
        
    def register_lesson_pattern(self, pattern: LessonPattern):
        """
        Register a new lesson pattern type
        
        Args:
            pattern: LessonPattern to register
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO lesson_patterns (
                pattern_id, description, category, affected_dimension, direction,
                times_encountered, times_validated, times_contradicted,
                reliability_score, last_encounter_episode,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.description,
            pattern.category,
            pattern.affected_dimension,
            pattern.direction,
            pattern.times_encountered,
            pattern.times_validated,
            pattern.times_contradicted,
            pattern.reliability_score,
            pattern.last_encounter_episode,
            now, now
        ))
        
        conn.commit()
        conn.close()
        
    def get_buffer_trajectory(self, 
                             dimension_id: int, 
                             n_episodes: int = 50) -> np.ndarray:
        """
        Get historical Buffer values for one dimension
        
        Args:
            dimension_id: Which SAWV dimension (0-4)
            n_episodes: How many recent episodes to retrieve
            
        Returns:
            Array of Buffer values in chronological order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT buffer_after 
            FROM episodes 
            ORDER BY id DESC 
            LIMIT ?
        ''', (n_episodes,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return np.array([])
            
        # Extract dimension values
        values = []
        for row in rows:
            buffer_state = json.loads(row[0])
            values.append(buffer_state[dimension_id])
            
        return np.array(values[::-1])  # Reverse to chronological order
        
    def get_accumulated_evidence(self, 
                                dimension_id: int,
                                direction: str,
                                n_episodes: int = 100) -> Dict[str, Any]:
        """
        Count lessons supporting movement in a direction
        
        Args:
            dimension_id: Which SAWV dimension
            direction: 'increase' or 'decrease'
            n_episodes: How far back to look
            
        Returns:
            Evidence summary with confidence score
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get episodes affecting this dimension
        cursor.execute('''
            SELECT buffer_before, buffer_after, lesson
            FROM episodes
            WHERE affected_dimensions LIKE ?
            ORDER BY id DESC
            LIMIT ?
        ''', (f'%{dimension_id}%', n_episodes))
        
        rows = cursor.fetchall()
        conn.close()
        
        supporting = 0
        contradicting = 0
        example_lessons = []
        
        for row in rows:
            buffer_before = json.loads(row[0])
            buffer_after = json.loads(row[1])
            lesson = row[2]
            
            # Did Buffer move in requested direction?
            change = buffer_after[dimension_id] - buffer_before[dimension_id]
            
            if direction == 'increase' and change > 0.1:
                supporting += 1
                if len(example_lessons) < 3:
                    example_lessons.append(lesson)
            elif direction == 'decrease' and change < -0.1:
                supporting += 1
                if len(example_lessons) < 3:
                    example_lessons.append(lesson)
            elif abs(change) > 0.1:
                contradicting += 1
                
        total = supporting + contradicting
        confidence = supporting / total if total > 0 else 0.0
        
        return {
            'supporting_episodes': supporting,
            'contradicting_episodes': contradicting,
            'total_episodes': total,
            'confidence': confidence,
            'example_lessons': example_lessons
        }
        
    def check_pattern_history(self, pattern_id: str) -> Dict[str, Any]:
        """
        Check if a lesson pattern has been reliable or deceptive
        
        Args:
            pattern_id: Pattern to check
            
        Returns:
            Pattern history with trust metrics and warnings
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                times_encountered,
                times_validated,
                times_contradicted,
                reliability_score,
                description
            FROM lesson_patterns
            WHERE pattern_id = ?
        ''', (pattern_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {
                'is_new_pattern': True,
                'warning': None,
                'trust_modifier': 1.0,  # Neutral for new patterns
                'recommendation': 'New pattern - proceed with normal caution'
            }
            
        times_encountered, validated, contradicted, reliability, description = row
        
        # Generate warnings based on reliability
        warning = None
        recommendation = None
        
        if reliability < 0.3:
            warning = f"🚨 HIGH RISK: Pattern '{description}' has deceived us {contradicted}/{times_encountered} times"
            recommendation = "Require extra validation before updating Active tier"
        elif reliability < 0.5:
            warning = f"⚠️  CAUTION: Pattern '{description}' has mixed reliability ({reliability:.1%})"
            recommendation = "Gather more evidence before acting"
        else:
            recommendation = f"Pattern is reliable ({reliability:.1%}) - normal confidence"
            
        # Trust modifier scales confidence in Pattern Analyzer
        trust_modifier = max(0.1, reliability)  # Never completely distrust (allow learning)
        
        return {
            'is_new_pattern': False,
            'times_encountered': times_encountered,
            'times_validated': validated,
            'times_contradicted': contradicted,
            'reliability_score': reliability,
            'trust_modifier': trust_modifier,
            'warning': warning,
            'recommendation': recommendation,
            'history_summary': f"Validated {validated} times, contradicted {contradicted} times"
        }
        
    def evaluate_pattern_reliability(self, 
                                    pattern_id: str,
                                    dimension_id: int) -> Dict[str, Any]:
        """
        Retrospectively evaluate if a pattern's predictions held true
        
        Looks at episodes where pattern appeared, then checks if Buffer
        continued in predicted direction or reversed (deception).
        
        Args:
            pattern_id: Pattern to evaluate
            dimension_id: Which dimension to analyze
            
        Returns:
            Updated reliability metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all episodes with this pattern
        cursor.execute('''
            SELECT id, buffer_after, affected_dimensions
            FROM episodes
            WHERE lesson_pattern_id = ?
            ORDER BY id
        ''', (pattern_id,))
        
        pattern_episodes = cursor.fetchall()
        
        if not pattern_episodes:
            conn.close()
            return {'error': 'Pattern not found in episodes'}
        
        validated = 0
        contradicted = 0
        
        # Get pattern direction
        cursor.execute('SELECT direction FROM lesson_patterns WHERE pattern_id = ?', (pattern_id,))
        pattern_direction = cursor.fetchone()[0]
        
        for episode_id, buffer_state, dimensions_json in pattern_episodes:
            # Look ahead to see if prediction held
            cursor.execute('''
                SELECT buffer_after
                FROM episodes
                WHERE id > ? AND id <= ?
                ORDER BY id
            ''', (episode_id, episode_id + self.validation_window))
            
            future_episodes = cursor.fetchall()
            
            if not future_episodes:
                continue  # Not enough future data yet
                
            # Did Buffer continue in predicted direction?
            dimension_id = json.loads(dimensions_json)[0]
            initial_value = json.loads(buffer_state)[dimension_id]
            
            # Check trajectory over validation window
            moves_predicted = 0
            moves_opposite = 0
            
            for (future_buffer,) in future_episodes:
                future_value = json.loads(future_buffer)[dimension_id]
                change = future_value - initial_value
                
                if abs(change) < 0.1:
                    continue  # No significant movement
                    
                if pattern_direction == 'increase' and change > 0:
                    moves_predicted += 1
                elif pattern_direction == 'decrease' and change < 0:
                    moves_predicted += 1
                else:
                    moves_opposite += 1
                    
            # Verdict
            if moves_predicted > moves_opposite:
                validated += 1
            elif moves_opposite > moves_predicted:
                contradicted += 1
                
        # Update pattern record
        total = validated + contradicted
        if total > 0:
            reliability = validated / total
        else:
            reliability = 0.5  # Neutral if no data
            
        cursor.execute('''
            UPDATE lesson_patterns
            SET times_validated = times_validated + ?,
                times_contradicted = times_contradicted + ?,
                reliability_score = ?,
                updated_at = ?
            WHERE pattern_id = ?
        ''', (validated, contradicted, reliability, datetime.now().isoformat(), pattern_id))
        
        conn.commit()
        conn.close()
        
        return {
            'pattern_id': pattern_id,
            'newly_validated': validated,
            'newly_contradicted': contradicted,
            'reliability_score': reliability,
            'is_deceptive': reliability < 0.3
        }
        
    def get_recent_episodes(self, n: int = 10) -> List[Dict]:
        """Get N most recent episodes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM episodes
            ORDER BY id DESC
            LIMIT ?
        ''', (n,))
        
        rows = cursor.fetchall()
        conn.close()
        
        episodes = []
        for row in rows:
            episodes.append({
                'id': row[0],
                'timestamp': row[1],
                'story_text': row[2],
                'lesson': row[4],
                'dissonance': row[15]
            })
            
        return episodes
        
    def get_statistics(self, dimension_id: int) -> Dict[str, Any]:
        """
        Compute summary statistics for a dimension
        
        Args:
            dimension_id: Which dimension (0-4)
            
        Returns:
            Stats including mean, variance, trend
        """
        trajectory = self.get_buffer_trajectory(dimension_id, n_episodes=1000)
        
        if len(trajectory) == 0:
            return {'error': 'No data'}
            
        return {
            'dimension_id': dimension_id,
            'n_episodes': len(trajectory),
            'current_value': trajectory[-1],
            'mean': np.mean(trajectory),
            'std': np.std(trajectory),
            'variance': np.var(trajectory),
            'min': np.min(trajectory),
            'max': np.max(trajectory),
            'range': np.max(trajectory) - np.min(trajectory)
        }
        
    def count_episodes(self) -> int:
        """Total episodes stored"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM episodes')
        count = cursor.fetchone()[0]
        conn.close()
        return count
        
    def get_all_patterns(self) -> List[Dict]:
        """Get all tracked lesson patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM lesson_patterns ORDER BY times_encountered DESC')
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in<span class="ml-2" /><span class="inline-block w-3 h-3 rounded-full bg-neutral-a12 align-middle mb-[0.1rem]" />
