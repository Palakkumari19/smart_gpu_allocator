import random
from typing import List, Optional
try:
    from workload.task_generator import EnhancedTask
except ImportError:
    # Fallback for testing
    EnhancedTask = object

class TraditionalScheduler:
    """Base class for traditional scheduling algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.total_tasks_assigned = 0
        self.total_decisions = 0
        self.execution_history = []
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select GPU index for task assignment"""
        raise NotImplementedError("Subclasses must implement select_gpu method")
    
    def get_stats(self):
        """Get scheduler statistics"""
        return {
            'name': self.name,
            'total_tasks_assigned': self.total_tasks_assigned,
            'total_decisions': self.total_decisions,
            'assignment_rate': self.total_tasks_assigned / max(1, self.total_decisions),
            'avg_execution_time': sum(self.execution_history) / max(1, len(self.execution_history))
        }

class FCFSScheduler(TraditionalScheduler):
    """First-Come, First-Served Scheduler"""
    
    def __init__(self):
        super().__init__("FCFS")
        self.gpu_queue_index = 0  # Round robin through available GPUs
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select first available GPU in order"""
        self.total_decisions += 1
        
        # Find first available GPU that can handle the task
        for i in range(len(gpus)):
            gpu = gpus[i]
            
            # Check if GPU is available and can handle task
            if (not gpu.crashed and 
                not gpu.is_in_cooldown() and
                gpu.current_memory + task.memory_required <= gpu.memory_capacity * 1.2):
                
                self.total_tasks_assigned += 1
                return i
        
        # No available GPU found
        return None
    
    def get_description(self):
        return "First-Come, First-Served: Assigns tasks to first available GPU in order"

class RoundRobinScheduler(TraditionalScheduler):
    """Round Robin Scheduler"""
    
    def __init__(self):
        super().__init__("Round Robin")
        self.current_gpu = 0
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select GPU in round-robin fashion"""
        self.total_decisions += 1
        
        # Try each GPU in round-robin order
        for _ in range(len(gpus)):
            gpu = gpus[self.current_gpu]
            
            # Check if GPU is available and can handle task
            if (not gpu.crashed and 
                not gpu.is_in_cooldown() and
                gpu.current_memory + task.memory_required <= gpu.memory_capacity * 1.2):
                
                selected_gpu = self.current_gpu
                self.total_tasks_assigned += 1
                
                # Move to next GPU
                self.current_gpu = (self.current_gpu + 1) % len(gpus)
                return selected_gpu
            
            # Move to next GPU
            self.current_gpu = (self.current_gpu + 1) % len(gpus)
        
        # No available GPU found
        return None
    
    def get_description(self):
        return "Round Robin: Cycles through GPUs in order, assigning to next available"

class LeastLoadedScheduler(TraditionalScheduler):
    """Least Loaded Scheduler"""
    
    def __init__(self):
        super().__init__("Least Loaded")
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select GPU with minimum load"""
        self.total_decisions += 1
        
        # Calculate load for each GPU
        gpu_loads = []
        for i, gpu in enumerate(gpus):
            if gpu.crashed or gpu.is_in_cooldown():
                gpu_loads.append(float('inf'))
                continue
            
            # Check if GPU can handle task
            if gpu.current_memory + task.memory_required > gpu.memory_capacity * 1.2:
                gpu_loads.append(float('inf'))
                continue
            
            # Calculate total load (queue + running tasks + memory usage)
            queue_load = len(gpu.task_queue)
            running_load = len(gpu.running_tasks)
            memory_load = gpu.get_memory_usage_percentage() / 100.0
            
            # Weighted load calculation
            total_load = queue_load * 1.0 + running_load * 2.0 + memory_load * 3.0
            gpu_loads.append(total_load)
        
        # Find GPU with minimum load
        if all(load == float('inf') for load in gpu_loads):
            return None  # No available GPUs
        
        min_load_gpu = min(range(len(gpus)), key=lambda i: gpu_loads[i])
        self.total_tasks_assigned += 1
        return min_load_gpu
    
    def get_description(self):
        return "Least Loaded: Selects GPU with minimum queue + running tasks + memory usage"

class RandomScheduler(TraditionalScheduler):
    """Random Scheduler"""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select random available GPU"""
        self.total_decisions += 1
        
        # Find all available GPUs
        available_gpus = []
        for i, gpu in enumerate(gpus):
            if (not gpu.crashed and 
                not gpu.is_in_cooldown() and
                gpu.current_memory + task.memory_required <= gpu.memory_capacity * 1.2):
                available_gpus.append(i)
        
        # Select random GPU from available ones
        if available_gpus:
            selected_gpu = random.choice(available_gpus)
            self.total_tasks_assigned += 1
            return selected_gpu
        
        return None
    
    def get_description(self):
        return "Random: Selects random available GPU"

class BestFitScheduler(TraditionalScheduler):
    """Best Fit Scheduler - selects GPU with tightest memory fit"""
    
    def __init__(self):
        super().__init__("Best Fit")
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select GPU with best memory fit"""
        self.total_decisions += 1
        
        # Calculate memory fit for each GPU
        memory_fits = []
        for i, gpu in enumerate(gpus):
            if gpu.crashed or gpu.is_in_cooldown():
                memory_fits.append(float('inf'))
                continue
            
            # Check if GPU can handle task
            if gpu.current_memory + task.memory_required > gpu.memory_capacity * 1.2:
                memory_fits.append(float('inf'))
                continue
            
            # Calculate memory waste (unused memory after assignment)
            available_memory = gpu.memory_capacity - gpu.current_memory
            memory_waste = available_memory - task.memory_required
            memory_fits.append(memory_waste)
        
        # Find GPU with minimum memory waste
        if all(fit == float('inf') for fit in memory_fits):
            return None  # No available GPUs
        
        best_fit_gpu = min(range(len(gpus)), key=lambda i: memory_fits[i])
        self.total_tasks_assigned += 1
        return best_fit_gpu
    
    def get_description(self):
        return "Best Fit: Selects GPU with tightest memory fit (minimum waste)"

class PriorityScheduler(TraditionalScheduler):
    """Priority Scheduler - considers task priority and GPU load"""
    
    def __init__(self):
        super().__init__("Priority")
    
    def select_gpu(self, gpus: List, task: EnhancedTask) -> Optional[int]:
        """Select GPU based on task priority and GPU load"""
        self.total_decisions += 1
        
        # Calculate priority score for each GPU
        priority_scores = []
        for i, gpu in enumerate(gpus):
            if gpu.crashed or gpu.is_in_cooldown():
                priority_scores.append(float('-inf'))
                continue
            
            # Check if GPU can handle task
            if gpu.current_memory + task.memory_required > gpu.memory_capacity * 1.2:
                priority_scores.append(float('-inf'))
                continue
            
            # Calculate priority score
            task_priority = 3 if task.priority == "high" else 1
            gpu_load = len(gpu.task_queue) + len(gpu.running_tasks)
            temp_penalty = max(0, gpu.temperature - 70) * 0.1
            
            # Priority score: higher priority tasks get better GPUs
            score = task_priority * 10 - gpu_load - temp_penalty
            priority_scores.append(score)
        
        # Find GPU with maximum priority score
        if all(score == float('-inf') for score in priority_scores):
            return None  # No available GPUs
        
        best_priority_gpu = max(range(len(gpus)), key=lambda i: priority_scores[i])
        self.total_tasks_assigned += 1
        return best_priority_gpu
    
    def get_description(self):
        return "Priority: Considers task priority and GPU load for assignment"

class BaselineComparison:
    """Baseline comparison system for traditional schedulers"""
    
    def __init__(self):
        self.schedulers = {
            'FCFS': FCFSScheduler(),
            'Round Robin': RoundRobinScheduler(),
            'Least Loaded': LeastLoadedScheduler(),
            'Random': RandomScheduler(),
            'Best Fit': BestFitScheduler(),
            'Priority': PriorityScheduler()
        }
    
    def get_scheduler(self, name: str) -> TraditionalScheduler:
        """Get scheduler by name"""
        return self.schedulers.get(name, RandomScheduler())
    
    def get_all_schedulers(self):
        """Get all available schedulers"""
        return list(self.schedulers.values())
    
    def get_scheduler_names(self):
        """Get all scheduler names"""
        return list(self.schedulers.keys())
    
    def compare_all(self, gpus: List, tasks: List) -> dict:
        """Compare all schedulers with same workload"""
        results = {}
        
        for name, scheduler in self.schedulers.items():
            # Reset scheduler state
            scheduler.total_tasks_assigned = 0
            scheduler.total_decisions = 0
            scheduler.execution_history = []
            
            # Simulate task assignment
            successful_assignments = 0
            for task in tasks:
                gpu_index = scheduler.select_gpu(gpus, task)
                if gpu_index is not None:
                    successful_assignments += 1
            
            # Store results
            results[name] = {
                'successful_assignments': successful_assignments,
                'total_tasks': len(tasks),
                'success_rate': successful_assignments / len(tasks),
                'scheduler_stats': scheduler.get_stats()
            }
        
        return results
    
    def print_comparison(self, results: dict):
        """Print comparison results"""
        print("\n" + "=" * 80)
        print("BASELINE SCHEDULER COMPARISON")
        print("=" * 80)
        print(f"{'Scheduler':<15} {'Success Rate':<12} {'Assignments':<12} {'Total Tasks':<12}")
        print("-" * 80)
        
        for name, result in results.items():
            success_rate = result['success_rate'] * 100
            assignments = result['successful_assignments']
            total_tasks = result['total_tasks']
            print(f"{name:<15} {success_rate:>10.1f}% {assignments:>12d} {total_tasks:>12d}")
        
        # Find best performer
        best_scheduler = max(results.keys(), key=lambda k: results[k]['success_rate'])
        print(f"\nBest Performer: {best_scheduler} ({results[best_scheduler]['success_rate']*100:.1f}%)")
        print("=" * 80)
