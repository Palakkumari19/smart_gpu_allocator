import random

class Task:
    def __init__(self):
        # LLM requests only - adjusted for 8GB GPUs
        self.memory_required = random.randint(1, 4)   # 1-4GB for 8GB GPUs
        self.execution_time = random.randint(2, 5)     # moderate time
        self.task_type = "llm"
        self.execution_progress = 0  # Track progress for parallel execution
        
        # Advanced features - Step 2
        self.preempted = False  # Track if task was preempted
        self.preemption_count = 0  # Number of times preempted
        self.original_execution_time = None  # Store original time
        
    def preempt(self):
        """Mark task as preempted"""
        self.preempted = True
        self.preemption_count += 1
        # Reduce remaining time after preemption (resume boost)
        if self.original_execution_time is None:
            self.original_execution_time = self.execution_time
        self.execution_time = max(1, self.execution_time - 1)  # Resume boost
        
    def resume(self):
        """Resume task after preemption"""
        self.preempted = False
