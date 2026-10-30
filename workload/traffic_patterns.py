import random
from enum import Enum
from datetime import datetime, timedelta

class TrafficPattern(Enum):
    LIGHT = "light"
    HEAVY = "heavy"
    BURST = "burst"
    PEAK = "peak"
    OFF_PEAK = "off_peak"
    MIXED = "mixed"

class TrafficManager:
    def __init__(self):
        # Traffic pattern configurations
        self.patterns = {
            TrafficPattern.LIGHT: {
                "requests_per_step": (1, 3),
                "description": "Light load - minimal traffic",
                "cpu_intensity": "low",
                "memory_pressure": "low"
            },
            TrafficPattern.HEAVY: {
                "requests_per_step": (8, 15),
                "description": "Heavy load - sustained high traffic",
                "cpu_intensity": "high",
                "memory_pressure": "high"
            },
            TrafficPattern.BURST: {
                "requests_per_step": (0, 20),
                "description": "Burst load - sudden traffic spikes",
                "cpu_intensity": "variable",
                "memory_pressure": "variable"
            },
            TrafficPattern.PEAK: {
                "requests_per_step": (12, 18),
                "description": "Peak hours - maximum capacity usage",
                "cpu_intensity": "high",
                "memory_pressure": "high"
            },
            TrafficPattern.OFF_PEAK: {
                "requests_per_step": (0, 2),
                "description": "Off-peak hours - minimal usage",
                "cpu_intensity": "low",
                "memory_pressure": "low"
            },
            TrafficPattern.MIXED: {
                "requests_per_step": (2, 12),
                "description": "Mixed load - variable traffic patterns",
                "cpu_intensity": "variable",
                "memory_pressure": "variable"
            }
        }
        
        # Current traffic state
        self.current_pattern = TrafficPattern.MIXED
        self.pattern_duration = 0
        self.step_counter = 0
        
        # Peak/off-peak simulation
        self.peak_hours = [(9, 12), (14, 18)]  # 9-12 AM, 2-6 PM
        self.current_hour = 9  # Start at 9 AM
        
        # Statistics tracking
        self.pattern_history = []
        self.total_requests = 0
        self.pattern_switches = 0

    def set_pattern(self, pattern, duration=10):
        """
        Set traffic pattern for specified duration
        """
        if isinstance(pattern, str):
            pattern = TrafficPattern(pattern)
        
        self.current_pattern = pattern
        self.pattern_duration = duration
        self.pattern_switches += 1
        
        # Record pattern change
        self.pattern_history.append({
            "step": self.step_counter,
            "pattern": pattern.value,
            "duration": duration
        })

    def get_current_requests(self):
        """
        Get number of requests for current step based on pattern
        """
        pattern_config = self.patterns[self.current_pattern]
        min_requests, max_requests = pattern_config["requests_per_step"]
        
        # Special handling for burst patterns
        if self.current_pattern == TrafficPattern.BURST:
            # Burst patterns have 70% chance of 0 requests, 30% chance of high burst
            if random.random() < 0.7:
                return random.randint(0, 2)  # Minimal traffic
            else:
                return random.randint(15, 20)  # High burst
        elif self.current_pattern == TrafficPattern.MIXED:
            # Mixed patterns vary more
            return random.randint(min_requests, max_requests)
        else:
            # Standard patterns
            return random.randint(min_requests, max_requests)

    def update_peak_off_peak(self):
        """
        Update traffic based on peak/off-peak hours
        """
        self.current_hour = (self.current_hour + 1) % 24
        
        # Check if current hour is peak
        is_peak = False
        for start, end in self.peak_hours:
            if start <= self.current_hour < end:
                is_peak = True
                break
        
        # Auto-set pattern based on time of day
        if is_peak:
            if self.current_pattern != TrafficPattern.PEAK:
                self.set_pattern(TrafficPattern.PEAK, duration=3)
        else:
            if self.current_pattern != TrafficPattern.OFF_PEAK:
                self.set_pattern(TrafficPattern.OFF_PEAK, duration=3)

    def step(self):
        """
        Advance one time step and update traffic
        """
        self.step_counter += 1
        
        # Update pattern duration
        if self.pattern_duration > 0:
            self.pattern_duration -= 1
        
        # Auto-switch patterns if duration expired
        if self.pattern_duration == 0:
            self._auto_select_next_pattern()
        
        # Update peak/off-peak simulation
        self.update_peak_off_peak()
        
        # Get requests for this step
        requests = self.get_current_requests()
        self.total_requests += requests
        
        return requests

    def _auto_select_next_pattern(self):
        """
        Automatically select next traffic pattern
        """
        # Pattern transition probabilities
        transitions = {
            TrafficPattern.LIGHT: {
                TrafficPattern.LIGHT: 0.6,
                TrafficPattern.MIXED: 0.3,
                TrafficPattern.BURST: 0.1
            },
            TrafficPattern.HEAVY: {
                TrafficPattern.HEAVY: 0.5,
                TrafficPattern.MIXED: 0.3,
                TrafficPattern.BURST: 0.2
            },
            TrafficPattern.BURST: {
                TrafficPattern.BURST: 0.3,
                TrafficPattern.MIXED: 0.4,
                TrafficPattern.LIGHT: 0.3
            },
            TrafficPattern.PEAK: {
                TrafficPattern.PEAK: 0.6,
                TrafficPattern.HEAVY: 0.3,
                TrafficPattern.MIXED: 0.1
            },
            TrafficPattern.OFF_PEAK: {
                TrafficPattern.OFF_PEAK: 0.7,
                TrafficPattern.LIGHT: 0.2,
                TrafficPattern.MIXED: 0.1
            },
            TrafficPattern.MIXED: {
                TrafficPattern.MIXED: 0.4,
                TrafficPattern.LIGHT: 0.2,
                TrafficPattern.HEAVY: 0.2,
                TrafficPattern.BURST: 0.2
            }
        }
        
        current_transitions = transitions.get(self.current_pattern, transitions[TrafficPattern.MIXED])
        patterns = list(current_transitions.keys())
        weights = list(current_transitions.values())
        
        next_pattern = random.choices(patterns, weights=weights)[0]
        duration = random.randint(5, 15)  # Random duration for next pattern
        
        self.set_pattern(next_pattern, duration)

    def get_mixed_workload_profile(self):
        """
        Get mixed workload configuration for current pattern
        """
        base_weights = {"LLM": 0.4, "Image": 0.35, "Video": 0.25}
        
        # Adjust weights based on traffic pattern
        if self.current_pattern == TrafficPattern.HEAVY:
            # Heavy load favors faster tasks (Image)
            base_weights = {"LLM": 0.3, "Image": 0.5, "Video": 0.2}
        elif self.current_pattern == TrafficPattern.PEAK:
            # Peak load balanced for efficiency
            base_weights = {"LLM": 0.35, "Image": 0.4, "Video": 0.25}
        elif self.current_pattern == TrafficPattern.OFF_PEAK:
            # Off-peak can handle more LLM tasks
            base_weights = {"LLM": 0.6, "Image": 0.25, "Video": 0.15}
        elif self.current_pattern == TrafficPattern.BURST:
            # Burst favors quick tasks
            base_weights = {"LLM": 0.2, "Image": 0.6, "Video": 0.2}
        
        return base_weights

    def get_traffic_statistics(self):
        """
        Get traffic pattern statistics
        """
        pattern_counts = {}
        for entry in self.pattern_history:
            pattern = entry["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return {
            "current_pattern": self.current_pattern.value,
            "current_hour": self.current_hour,
            "total_requests": self.total_requests,
            "pattern_switches": self.pattern_switches,
            "pattern_distribution": pattern_counts,
            "steps_simulated": self.step_counter,
            "avg_requests_per_step": self.total_requests / max(1, self.step_counter)
        }

    def simulate_day_cycle(self, steps=24):
        """
        Simulate a full day cycle (24 hours)
        """
        print(f"Simulating {steps} hours of traffic patterns...")
        
        for step in range(steps):
            requests = self.step()
            pattern = self.current_pattern.value
            hour = self.current_hour
            
            if step % 6 == 0:  # Print every 6 hours
                print(f"Hour {hour:2d}: {pattern:12s} - {requests:2d} requests")
        
        return self.get_traffic_statistics()

    def reset(self):
        """
        Reset traffic manager state
        """
        self.current_pattern = TrafficPattern.MIXED
        self.pattern_duration = 0
        self.step_counter = 0
        self.current_hour = 9
        self.pattern_history = []
        self.total_requests = 0
        self.pattern_switches = 0
