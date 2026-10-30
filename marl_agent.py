import threading
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class RiskAwareRewardConfig:
    """Configurable weights and boundaries for the Risk-Aware MARL reward function."""
    base_success_reward: float = 4.0
    task_completion_bonus: float = 3.0
    weight_performance: float = 1.0
    weight_risk_penalty: float = 3.0
    weight_expected_failure_cost: float = 1.5
    weight_recovery_penalty: float = 1.0
    weight_constraint_violation: float = 5.0
    failure_penalty: float = 20.0
    min_reward: float = -20.0
    max_reward: float = 12.0
@dataclass
class ProbeTask:
    """Lightweight task probe for computing expected failure cost (EFC) and telemetry."""
    memory_required: float = 2.0
    execution_time: float = 4.0
    workload_severity: str = "MEDIUM"
    id: str = "probe_task"


DEFAULT_PROBE_TASK = ProbeTask()


class MARLAgent:
    """Multi-Agent RL Agent for individual GPU management with risk awareness"""
    ACTION_ACCEPT = 0    # Accept / Bid to schedule task on this GPU
    ACTION_DECLINE = 1   # Decline / Yield task to other cluster GPUs
    ACTION_DEFER = 2     # Defer / Low-priority execution
    ACTION_EXPEDITE = 3  # Expedite / High-priority execution
    
    def __init__(self, gpu_id, state_size=16, action_size=4, learning_rate=0.01):
        self._lock = threading.RLock()
        self.lock = self._lock
        self.gpu_id = gpu_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Enhanced exploration parameters for better early training
        self.epsilon = 1.0  # Start with maximum exploration
        self.epsilon_min = 0.05  # Higher minimum for continued exploration
        self.epsilon_decay = 0.997  # Slower decay for more exploration
        self.training_steps = 0  # Track training progress
        
        # Adaptive exploration schedule
        self.initial_exploration_phase = 100  # First 100 steps high exploration
        self.learning_phase = 500  # Main learning phase
        
        # Q-table for this specific GPU (tabular Q-learning with discrete states)
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        # Core abstract Q-table for cross-state generalization
        self.core_q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Performance tracking
        self.total_reward = 0
        self.tasks_assigned = 0
        self.successful_assignments = 0
        
    def get_state(self, gpu, cluster_state):
        """
        Extract a finite, discrete, repeatable 16-element state tuple for tabular Q-learning.
        Incorporate local GPU state, risk metrics, reliability, status, task context,
        expected failure cost, and cluster contention.
        """
        # Safely extract risk state with explicit defaults
        risk_state = gpu.get_risk_state() if hasattr(gpu, 'get_risk_state') else None
        if risk_state is not None:
            risk_score = float(risk_state.risk_score)
            failure_rate = float(risk_state.reliability_history.failure_rate)
            is_quarantined = bool(risk_state.is_quarantined)
        else:
            risk_score = 0.0
            failure_rate = 0.0
            is_quarantined = False

        # 1. Memory usage percentage bin (0: <50%, 1: 50-80%, 2: >80%)
        mem_pct = gpu.get_memory_usage_percentage()
        mem_bin = 0 if mem_pct < 50.0 else (1 if mem_pct <= 80.0 else 2)

        # 2. Local GPU load bin (running + queued) (0: idle, 1: 1-2 tasks, 2: >=3 tasks)
        total_load = len(gpu.task_queue) + len(gpu.running_tasks)
        load_bin = 0 if total_load == 0 else (1 if total_load <= 2 else 2)

        # 3. Temperature bin (0: <=65C, 1: 65-80C, 2: >80C)
        temp = gpu.temperature
        temp_bin = 0 if temp <= 65.0 else (1 if temp <= 80.0 else 2)

        # 4. Risk score bin (0: <=0.25, 1: 0.25-0.50, 2: 0.50-0.75, 3: >0.75)
        risk_bin = 0 if risk_score <= 0.25 else (1 if risk_score <= 0.50 else (2 if risk_score <= 0.75 else 3))

        # 5. Failure rate bin (0: <=5%, 1: 5-20%, 2: >20%)
        fail_bin = 0 if failure_rate <= 5.0 else (1 if failure_rate <= 20.0 else 2)

        # 6. Operational status bin (0: healthy active, 1: in cooldown, 2: crashed/quarantined)
        if gpu.crashed or is_quarantined:
            status_bin = 2
        elif gpu.is_in_cooldown():
            status_bin = 1
        else:
            status_bin = 0

        # 7. Memory fragmentation bin (0: <10%, 1: 10-30%, 2: >30%)
        frag_pct = gpu.get_fragmentation_percentage()
        frag_bin = 0 if frag_pct < 10.0 else (1 if frag_pct <= 30.0 else 2)

        # 8. Preemptions bin (0: 0, 1: 1-2, 2: >2)
        preempt_count = gpu.get_preempted_tasks_count()
        preempt_bin = 0 if preempt_count == 0 else (1 if preempt_count <= 2 else 2)

        # 9. Task memory requirement bin (0: <=2GB, 1: 2-4GB, 2: >4GB)
        task_mem = cluster_state.get('current_task_memory', 0.0)
        task_mem_bin = 0 if task_mem <= 2.0 else (1 if task_mem <= 4.0 else 2)

        # 10. Task execution time bin (0: <=2 steps, 1: 3-5 steps, 2: >5 steps)
        task_time = cluster_state.get('current_task_time', 0.0)
        task_time_bin = 0 if task_time <= 2.0 else (1 if task_time <= 5.0 else 2)

        # 11. Task criticality / impact bin (0: LOW, 1: MEDIUM, 2: HIGH/EXTREME)
        task_impact = cluster_state.get('current_task_impact', 0.5)
        task_impact_bin = 0 if task_impact < 0.4 else (1 if task_impact <= 0.7 else 2)

        # 12. Expected Failure Cost (EFC) bin (0: <0.5, 1: 0.5-2.0, 2: >2.0)
        efc = cluster_state.get('current_efc', None)
        if efc is None:
            base_restart = max(1.0, task_time * 0.3 + task_mem * 0.1)
            efc = risk_score * task_impact * base_restart
        efc_bin = 0 if efc < 0.5 else (1 if efc <= 2.0 else 2)

        # 13. Cluster utilization bin (0: <40%, 1: 40-75%, 2: >75%)
        cluster_util = cluster_state.get('total_cluster_utilization', 0.0)
        cluster_util_bin = 0 if cluster_util < 40.0 else (1 if cluster_util <= 75.0 else 2)

        # 14. Idle GPUs in cluster (0: 0, 1: 1-2, 2: >2)
        idle_count = cluster_state.get('idle_gpu_count', 0)
        idle_gpu_bin = 0 if idle_count == 0 else (1 if idle_count <= 2 else 2)

        # 15. Overloaded GPUs in cluster (0: 0, 1: 1-2, 2: >2)
        overloaded_count = cluster_state.get('overloaded_gpu_count', 0)
        overloaded_gpu_bin = 0 if overloaded_count == 0 else (1 if overloaded_count <= 2 else 2)

        # 16. Cluster available memory bin (0: <8GB, 1: 8-16GB, 2: >16GB)
        avail_cluster_mem = cluster_state.get('available_cluster_memory', 0.0)
        cluster_avail_mem_bin = 0 if avail_cluster_mem < 8.0 else (1 if avail_cluster_mem <= 16.0 else 2)

        state = (
            mem_bin, load_bin, temp_bin, risk_bin, fail_bin, status_bin,
            frag_bin, preempt_bin, task_mem_bin, task_time_bin, task_impact_bin,
            efc_bin, cluster_util_bin, idle_gpu_bin, overloaded_gpu_bin, cluster_avail_mem_bin
        )

        return tuple(state[:self.state_size])

    def get_continuous_state(self, gpu, cluster_state):
        """Extract continuous normalized state representation for diagnostics and logging."""
        risk_state = gpu.get_risk_state() if hasattr(gpu, 'get_risk_state') else None
        risk_score = float(risk_state.risk_score) if risk_state else 0.0
        failure_rate = float(risk_state.reliability_history.failure_rate) if risk_state else 0.0
        is_quarantined = 1.0 if (risk_state and risk_state.is_quarantined) else 0.0

        state = [
            gpu.get_memory_usage_percentage() / 100.0,
            min(1.0, gpu.temperature / 100.0),
            min(1.0, len(gpu.task_queue) / 10.0),
            min(1.0, len(gpu.running_tasks) / 3.0),
            gpu.get_fragmentation_percentage() / 100.0,
            1.0 if gpu.crashed else is_quarantined,
            1.0 if gpu.is_in_cooldown() else 0.0,
            min(1.0, gpu.get_preempted_tasks_count() / 5.0),
            min(1.0, risk_score),
            min(1.0, failure_rate / 100.0),
            min(1.0, cluster_state.get('current_task_memory', 0) / 8.0),
            min(1.0, cluster_state.get('current_task_time', 0) / 10.0),
            min(1.0, cluster_state.get('current_efc', 0) / 5.0),
            min(1.0, cluster_state.get('total_cluster_utilization', 0) / 100.0),
            min(1.0, cluster_state.get('idle_gpu_count', 0) / 4.0),
            min(1.0, cluster_state.get('available_cluster_memory', 0) / 32.0),
        ]
        return tuple(state[:self.state_size])
    
    def _get_core_state(self, state):
        """Extract compact core state for generalization across similar tasks and conditions."""
        if isinstance(state, tuple) and len(state) >= 12:
            # Core dimensions: mem_bin, load_bin, temp_bin, risk_bin, status_bin, task_mem_bin, efc_bin
            return (state[0], state[1], state[2], state[3], state[5], state[8], state[11])
        return state

    def get_core_state(self, state):
        """Public alias to extract compact core state for generalization."""
        return self._get_core_state(state)

    def choose_action(self, state, available_actions=None):
        """Choose action using epsilon-greedy policy with core state generalization (thread-safe)."""
        with self._lock:
            if available_actions is None:
                available_actions = list(range(self.action_size))
            
            if random.random() <= self.epsilon:
                return random.choice(available_actions)
            
            # Get Q-values for current state (with core state generalization fallback)
            if state in self.q_table:
                q_values = self.q_table[state]
            else:
                core_state = self._get_core_state(state)
                q_values = self.core_q_table[core_state]
            
            # Filter to available actions
            masked_q_values = np.array([q_values[action] if action in available_actions else -np.inf 
                                       for action in range(self.action_size)])
            
            return np.argmax(masked_q_values)
    
    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning with core state generalization (thread-safe)."""
        with self._lock:
            # Q-learning update rule for exact tabular state
            old_value = self.q_table[state][action]
            next_max = np.max(self.q_table[next_state])
            
            # Bellman equation for exact state
            new_value = old_value + self.learning_rate * (reward + 0.95 * next_max - old_value)
            self.q_table[state][action] = new_value

            # Generalize update to core state abstraction
            core_state = self._get_core_state(state)
            core_next_state = self._get_core_state(next_state)
            core_old = self.core_q_table[core_state][action]
            core_next_max = np.max(self.core_q_table[core_next_state])
            self.core_q_table[core_state][action] = core_old + self.learning_rate * (reward + 0.95 * core_next_max - core_old)
            
            # Track performance
            self.total_reward += reward
            self.tasks_assigned += 1
            if reward > 0:
                self.successful_assignments += 1
    
    def decay_epsilon(self):
        """Adaptive epsilon decay with enhanced exploration (thread-safe)."""
        with self._lock:
            self.training_steps += 1
            
            # Adaptive exploration schedule
            if self.training_steps < self.initial_exploration_phase:
                # High exploration phase - very slow decay
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.998)
            elif self.training_steps < self.learning_phase:
                # Learning phase - moderate decay
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
            else:
                # Mature phase - faster decay to exploitation
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.992)
            
            # Ensure epsilon doesn't go below minimum
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_performance_stats(self):
        """Get performance statistics for this agent (thread-safe)."""
        with self._lock:
            success_rate = self.successful_assignments / max(1, self.tasks_assigned)
            return {
                'gpu_id': self.gpu_id,
                'total_reward': self.total_reward,
                'tasks_assigned': self.tasks_assigned,
                'success_rate': success_rate,
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table)
            }

class MARLManager:
    """Multi-Agent RL Manager for coordinating multiple GPU agents with risk-aware cooperative bidding"""
    
    def __init__(self, num_gpus=4, reward_config: Optional[RiskAwareRewardConfig] = None, learning_rate: float = 0.01):
        self._lock = threading.RLock()
        self.lock = self._lock
        self.num_gpus = num_gpus
        self.reward_config = reward_config or RiskAwareRewardConfig()
        self.learning_rate = learning_rate
        # Fine-tuned learning parameters
        self.agents = [MARLAgent(i, learning_rate=self.learning_rate) for i in range(num_gpus)]
        self.global_step = 0
        self.total_reward = 0
        
        # Coordination metrics
        self.coordination_history = []
        self.load_balancing_score = 0
        self.last_candidate_breakdowns = []
        
        # Enhanced coordination parameters
        self.coordination_frequency = 5  # Coordinate every 5 steps
        self.exploration_boost_threshold = 100  # Boost exploration if performance is low
        self.performance_window = 20  # Track performance over last 20 steps
        
    def get_cluster_state(self, gpus):
        """Get cluster-level state for coordination"""
        total_memory_capacity = sum(g.memory_capacity for g in gpus)
        current_cluster_memory = sum(g.current_memory for g in gpus)
        total_cluster_utilization = sum(g.get_total_load() for g in gpus)
        idle_gpu_count = sum(1 for g in gpus if g.get_total_load() == 0)
        overloaded_gpu_count = sum(1 for g in gpus if g.get_total_load() >= 4)

        return {
            'avg_memory_usage': sum(g.get_memory_usage_percentage() for g in gpus) / len(gpus),
            'avg_temperature': sum(g.temperature for g in gpus) / len(gpus),
            'total_queued_tasks': sum(g.get_queue_length() for g in gpus),
            'total_running_tasks': sum(g.get_running_tasks_count() for g in gpus),
            'avg_queue_length': sum(g.get_queue_length() for g in gpus) / len(gpus),
            'idle_gpu_count': idle_gpu_count,
            'overloaded_gpu_count': overloaded_gpu_count,
            'total_cluster_utilization': (total_cluster_utilization / max(1, len(gpus) * 6)) * 100.0,
            'available_cluster_memory': max(0.0, total_memory_capacity - current_cluster_memory),
            'gpus_in_cooldown': sum(1 for g in gpus if g.is_in_cooldown()),
            'avg_fragmentation': sum(g.get_fragmentation_percentage() for g in gpus) / len(gpus),
            'total_preemptions': sum(g.total_preemptions for g in gpus)
        }

    def calculate_expected_failure_cost(self, gpu, task: Optional[Any] = None, recovery_cost: Optional[float] = None) -> float:
        """
        Calculate estimated expected failure cost for a candidate GPU assignment:
        EFC = failure_probability * failure_impact * recovery_cost

        Args:
            gpu: Candidate GPU object
            task: Task object requiring placement (defaults to DEFAULT_PROBE_TASK if None)
            recovery_cost: Optional explicit recovery cost in steps (calculated if None)

        Returns:
            Expected failure cost (bounded float >= 0.0)
        """
        if task is None:
            task = DEFAULT_PROBE_TASK
        risk_state = gpu.get_risk_state() if hasattr(gpu, 'get_risk_state') else None

        # 1. Failure probability estimation from risk engine & reliability history
        if risk_state is not None:
            risk_score = float(risk_state.risk_score)
            failure_rate = float(risk_state.reliability_history.failure_rate) / 100.0  # normalize [0, 100] to [0, 1]
            failure_prob = 0.70 * risk_score + 0.30 * min(1.0, max(0.0, failure_rate))
        else:
            # Safe documented fallback: temperature and memory pressure heuristics
            temp_norm = min(1.0, max(0.0, (gpu.temperature - 50.0) / 45.0))
            mem_norm = min(1.0, gpu.current_memory / max(1.0, gpu.memory_capacity))
            failure_prob = 0.50 * temp_norm + 0.50 * mem_norm

        # 2. Failure impact estimation (task importance and complexity)
        severity = getattr(task, 'severity', None) or getattr(task, 'workload_severity', 'MEDIUM')
        if isinstance(severity, str):
            sev_map = {'LOW': 0.25, 'MEDIUM': 0.5, 'HIGH': 0.75, 'VERY HIGH': 1.0, 'EXTREME': 1.0}
            severity_mult = sev_map.get(severity.upper(), 0.5)
        else:
            severity_mult = 0.5

        mem_req = getattr(task, 'memory_required', 1.0)
        exec_time = getattr(task, 'execution_time', 1.0)

        mem_intensity = min(1.0, mem_req / 8.0)
        time_intensity = min(1.0, exec_time / 20.0)
        resource_intensity = (mem_intensity + time_intensity) / 2.0
        failure_impact = (severity_mult * 0.6) + (resource_intensity * 0.4)

        # 3. Recovery cost estimation
        if recovery_cost is None:
            base_restart = exec_time * 0.3
            checkpoint_overhead = mem_req * 0.1
            cooldown_penalty = float(getattr(gpu, 'crash_cooldown_timer', 0))
            recovery_cost = max(1.0, base_restart + checkpoint_overhead) + cooldown_penalty

        expected_cost = failure_prob * failure_impact * recovery_cost
        return float(max(0.0, min(20.0, expected_cost)))

    def compute_bid_score_breakdown(self, gpu, task, agent, cluster_state) -> dict:
        """
        Compute detailed components of the risk-aware MARL bid score for candidate observability:
        - Q-value and weighted Q-value contribution (0.50 * Q)
        - Performance score and subcomponents (memory headroom, queue, temp, running, fragmentation)
        - Risk score and risk penalty (0.25 * risk_score)
        - Expected failure cost (EFC) and EFC penalty (0.20 * min(1.0, efc / 4.0))
        - Eligibility flags and reason for exclusion
        - Final bid score: (q_value * 0.50) + perf_score - risk_penalty - efc_penalty
        """
        # Strict admission checks
        is_crashed = bool(gpu.crashed)
        is_cooldown = bool(gpu.is_in_cooldown())
        risk_state = gpu.get_risk_state() if hasattr(gpu, 'get_risk_state') else None
        is_quarantined = bool((risk_state and risk_state.is_quarantined) or (hasattr(gpu, 'is_quarantined') and gpu.is_quarantined()))
        mem_available = max(0.0, float(gpu.memory_capacity - gpu.current_memory))
        has_capacity = bool(gpu.current_memory + task.memory_required <= gpu.memory_capacity)

        ineligible_reason = None
        if is_crashed:
            ineligible_reason = "CRASHED"
        elif is_cooldown:
            ineligible_reason = f"COOLDOWN ({gpu.get_cooldown_remaining() if hasattr(gpu, 'get_cooldown_remaining') else 0}s)"
        elif is_quarantined:
            ineligible_reason = f"QUARANTINED ({risk_state.quarantine_reason if risk_state and risk_state.quarantine_reason else 'high risk'})"
        elif not has_capacity:
            ineligible_reason = f"VRAM_LIMIT ({task.memory_required:.1f}G > {mem_available:.1f}G free)"

        is_eligible = (ineligible_reason is None)

        # Calculate Expected Failure Cost (EFC)
        efc = float(self.calculate_expected_failure_cost(gpu, task))
        risk_score = float(risk_state.risk_score) if risk_state else 0.0

        # Pass context into cluster_state for agent state discretization
        cluster_state['current_task_memory'] = task.memory_required
        cluster_state['current_task_time'] = task.execution_time
        cluster_state['current_efc'] = efc

        # Evaluate learned Q-value for scheduling action (ACTION_ACCEPT = 0)
        state = agent.get_state(gpu, cluster_state)
        action_idx = getattr(agent, 'ACTION_ACCEPT', 0)
        if state in agent.q_table:
            q_value = float(agent.q_table[state][action_idx])
        else:
            core_state = agent._get_core_state(state)
            if core_state in agent.core_q_table:
                q_value = float(agent.core_q_table[core_state][action_idx])
            else:
                q_value = 0.0

        q_contribution = q_value * 0.50

        # Performance components
        memory_headroom = max(0.0, 1.0 - ((gpu.current_memory + task.memory_required) / gpu.memory_capacity))
        queue_penalty = min(1.0, len(gpu.task_queue) / 10.0)
        temp_penalty = min(1.0, max(0.0, (gpu.temperature - 50.0) / 50.0))
        running_penalty = min(1.0, len(gpu.running_tasks) / 3.0)
        fragmentation_penalty = min(1.0, gpu.get_fragmentation_percentage() / 100.0)

        memory_component = memory_headroom * 0.10
        queue_component = (1.0 - queue_penalty) * 0.08
        temperature_component = (1.0 - temp_penalty) * 0.05
        running_component = (1.0 - running_penalty) * 0.04
        fragmentation_component = (1.0 - fragmentation_penalty) * 0.03
        perf_score = memory_component + queue_component + temperature_component + running_component + fragmentation_component

        # Risk-aware penalties
        risk_penalty = 0.25 * risk_score
        efc_penalty = 0.20 * min(1.0, efc / 4.0)

        # Bounded combination: Q-value guidance + performance headroom - risk penalties
        if is_eligible:
            final_bid_score = q_contribution + perf_score - risk_penalty - efc_penalty
        else:
            final_bid_score = -np.inf

        return {
            "gpu_id": gpu.id,
            "label": f"GPU {gpu.id}",
            "eligible": is_eligible,
            "ineligible_reason": ineligible_reason,
            "q_value": round(q_value, 4),
            "q_contribution": round(q_contribution, 4),
            "perf_score": round(perf_score, 4),
            "memory_headroom": round(memory_headroom, 4),
            "queue_penalty": round(queue_penalty, 4),
            "temp_penalty": round(temp_penalty, 4),
            "running_penalty": round(running_penalty, 4),
            "fragmentation_penalty": round(fragmentation_penalty, 4),
            "risk_score": round(risk_score, 4),
            "risk_penalty": round(risk_penalty, 4),
            "efc": round(efc, 4),
            "efc_penalty": round(efc_penalty, 4),
            "bid_score": round(final_bid_score, 4) if np.isfinite(final_bid_score) else -999.0,
            "is_finite": bool(np.isfinite(final_bid_score)),
        }

    def compute_bid_score(self, gpu, task, agent, cluster_state):
        """
        Compute a risk-aware cooperative bid score for a GPU using MARL Q-values,
        performance signals, GPU risk score, and expected failure cost (EFC).
        """
        bd = self.compute_bid_score_breakdown(gpu, task, agent, cluster_state)
        return float(bd["bid_score"]) if bd["is_finite"] else -np.inf

    def select_gpu_for_task(self, task, gpus, cluster_state, return_infeasible_fallback: bool = True, return_breakdown: bool = False):
        """
        Select best GPU for task using cooperative bidding with MARL coordination (thread-safe).
        
        Args:
            task: Task to schedule
            gpus: List of GPU objects
            cluster_state: Cluster-level metrics dictionary
            return_infeasible_fallback: If True, falls back to least loaded if all reject (for backward compatibility).
                                       If False, returns None when no GPU is feasible.
            return_breakdown: If True, returns tuple of (selected_gpu_id, candidate_breakdowns).
        """
        with self._lock:
            task_context = {
                'current_task_memory': task.memory_required,
                'current_task_time': task.execution_time
            }
            cluster_state.update(task_context)

            candidate_breakdowns = []
            gpu_scores = []
            for i, (gpu, agent) in enumerate(zip(gpus, self.agents)):
                bd = self.compute_bid_score_breakdown(gpu, task, agent, cluster_state)
                candidate_breakdowns.append(bd)
                gpu_scores.append((i, bd["bid_score"] if bd["is_finite"] else -np.inf))

            self.last_candidate_breakdowns = candidate_breakdowns

            gpu_scores.sort(key=lambda x: x[1], reverse=True)

            for gpu_id, score in gpu_scores:
                if np.isfinite(score):
                    if return_breakdown:
                        return gpu_id, candidate_breakdowns
                    return gpu_id

            # All candidate GPUs rejected (infeasible)
            if not return_infeasible_fallback:
                if return_breakdown:
                    return None, candidate_breakdowns
                return None

            # Fallback to least loaded if all agents reject (backward compatibility)
            fallback_id = min(range(len(gpus)), key=lambda i: (len(gpus[i].task_queue), gpus[i].get_total_load()))
            if return_breakdown:
                return fallback_id, candidate_breakdowns
            return fallback_id
    
    def update_agents(self, gpus, tasks_assigned, rewards):
        """Update all agents after task assignment with adaptive coordination (thread-safe)"""
        with self._lock:
            cluster_state = self.get_cluster_state(gpus)
            
            # Performance monitoring
            avg_performance = 0
            for agent in self.agents:
                stats = agent.get_performance_stats()
                avg_performance += stats['success_rate']
            avg_performance /= len(self.agents)
            
            # Adaptive exploration boost if performance is low
            if avg_performance < 0.3 and self.global_step < self.exploration_boost_threshold:
                for agent in self.agents:
                    agent.epsilon = min(1.0, agent.epsilon * 1.1)  # Boost exploration
            
            for i, (gpu, agent) in enumerate(zip(gpus, self.agents)):
                state = agent.get_state(gpu, cluster_state)
                next_cluster_state = self.get_cluster_state(gpus)
                next_state = agent.get_state(gpu, next_cluster_state)
                
                # Update with received reward
                action = tasks_assigned[i]
                reward = rewards.get(i, 0)
                agent.update(state, action, reward, next_state)
            
            # Periodic coordination and parameter adjustment
            if self.global_step % self.coordination_frequency == 0:
                self.coordinate_agents(gpus)
            
            self.global_step += 1
    
    def coordinate_agents(self, gpus):
        """Coordinate agents and adjust parameters based on performance"""
        with self._lock:
            # Get current performance metrics
            agent_performances = []
            for agent in self.agents:
                stats = agent.get_performance_stats()
                agent_performances.append(stats['success_rate'])
            
            avg_performance = sum(agent_performances) / len(agent_performances)
            
            # Adjust learning parameters based on performance
            for i, agent in enumerate(self.agents):
                # Boost struggling agents
                if agent_performances[i] < avg_performance * 0.8:
                    agent.learning_rate = min(0.005, agent.learning_rate * 1.1)
                    agent.epsilon = min(0.8, agent.epsilon * 1.05)
                # Penalize overperforming agents slightly to encourage exploration
                elif agent_performances[i] > avg_performance * 1.2:
                    agent.learning_rate = max(0.0005, agent.learning_rate * 0.9)
                    agent.epsilon = max(agent.epsilon * 0.95, agent.epsilon_min)
            
            # Update coordination history
            self.coordination_history.append({
                'step': self.global_step,
                'avg_performance': avg_performance,
                'agent_epsilons': [agent.epsilon for agent in self.agents],
                'agent_learning_rates': [agent.learning_rate for agent in self.agents]
            })
    
    def calculate_reward(self, gpu, task, success, expected_failure_cost: Optional[float] = None) -> float:
        """
        Risk-Aware Reward Calculation balancing performance, reliability, and risk.
        reward = performance_reward - risk_penalty - expected_failure_cost_penalty 
                 - recovery_cost_penalty - constraint_violation_penalty

        Guaranteed bounded in [min_reward, max_reward] to stabilize Bellman updates.
        """
        cfg = self.reward_config

        if not success:
            return float(-cfg.failure_penalty)

        if gpu.crashed:
            return float(-cfg.failure_penalty)

        # Performance baseline
        reward = cfg.base_success_reward + cfg.task_completion_bonus

        # Queue length penalties
        queue_length = len(gpu.task_queue)
        if queue_length >= 5:
            reward -= 5.0
        elif queue_length >= 3:
            reward -= 2.0
        elif queue_length <= 1:
            reward += 1.0

        # Running tasks pressure
        running_tasks = len(gpu.running_tasks)
        if running_tasks >= 3:
            reward -= 2.5
        elif running_tasks == 2:
            reward -= 0.5
        elif running_tasks == 1:
            reward += 0.5

        # Memory headroom
        memory_usage_ratio = gpu.current_memory / max(1.0, gpu.memory_capacity)
        if memory_usage_ratio > 0.85:
            reward -= 1.5
        elif memory_usage_ratio > 0.6:
            reward += 0.5
        else:
            reward += 0.25

        # Temperature
        if gpu.temperature > 80:
            reward -= 1.5
        elif gpu.temperature > 70:
            reward -= 0.5
        elif gpu.temperature <= 65:
            reward += 0.25

        # Risk-aware penalty from authoritative GPU risk engine
        risk_state = gpu.get_risk_state() if hasattr(gpu, 'get_risk_state') else None
        risk_score = float(risk_state.risk_score) if risk_state else 0.0
        reward -= (cfg.weight_risk_penalty * risk_score)

        # Expected Failure Cost penalty
        if expected_failure_cost is None:
            efc = self.calculate_expected_failure_cost(gpu, task)
        else:
            efc = float(expected_failure_cost)
        reward -= (cfg.weight_expected_failure_cost * min(3.0, efc))

        # Recovery / cooldown penalty
        cooldown_timer = float(getattr(gpu, 'crash_cooldown_timer', 0))
        if cooldown_timer > 0 or gpu.is_in_cooldown():
            reward -= (cfg.weight_recovery_penalty * max(1.0, cooldown_timer))

        # Quarantine constraint violation penalty
        if risk_state and risk_state.is_quarantined:
            reward -= cfg.weight_constraint_violation

        # Bounded and numerically controlled
        return float(max(cfg.min_reward, min(cfg.max_reward, reward)))
    
    def decay_all_epsilon(self):
        """Decay exploration rate for all agents (thread-safe)"""
        with self._lock:
            for agent in self.agents:
                agent.decay_epsilon()
    
    def get_global_stats(self):
        """Get global MARL statistics (thread-safe)"""
        with self._lock:
            total_reward = sum(agent.total_reward for agent in self.agents)
            total_tasks = sum(agent.tasks_assigned for agent in self.agents)
            total_successful = sum(agent.successful_assignments for agent in self.agents)
            
            return {
                'global_step': self.global_step,
                'total_reward': total_reward,
                'total_tasks_assigned': total_tasks,
                'global_success_rate': total_successful / max(1, total_tasks),
                'avg_epsilon': sum(agent.epsilon for agent in self.agents) / len(self.agents),
                'total_q_table_size': sum(len(agent.q_table) for agent in self.agents),
                'agent_stats': [agent.get_performance_stats() for agent in self.agents]
            }
    
    def snapshot_q_tables(self):
        """Take a snapshot of all agent Q-tables (thread-safe)."""
        with self._lock:
            return [{state: values.copy() for state, values in agent.q_table.items()} for agent in self.agents]

    def restore_q_tables(self, snapshot):
        """Restore Q-tables from snapshot (thread-safe)."""
        with self._lock:
            if not snapshot:
                return
            for agent, table in zip(self.agents, snapshot):
                agent.q_table = defaultdict(lambda: np.zeros(agent.action_size))
                agent.core_q_table = defaultdict(lambda: np.zeros(agent.action_size))
                if isinstance(table, dict):
                    for state, values in table.items():
                        agent.q_table[state] = values.copy()
                        core_state = agent._get_core_state(state)
                        agent.core_q_table[core_state] = values.copy()

    def save_models(self, filepath):
        """Save MARL models (thread-safe)"""
        with self._lock:
            import pickle
            models = {f'agent_{i}': agent.q_table for i, agent in enumerate(self.agents)}
            with open(filepath, 'wb') as f:
                pickle.dump(models, f)
    
    def load_models(self, filepath):
        """Load MARL models (thread-safe)"""
        with self._lock:
            import pickle
            try:
                with open(filepath, 'rb') as f:
                    models = pickle.load(f)
                for i, agent in enumerate(self.agents):
                    if f'agent_{i}' in models:
                        agent.q_table = defaultdict(lambda: np.zeros(self.agents[0].action_size))
                        agent.q_table.update(models[f'agent_{i}'])
                return True
            except:
                return False
