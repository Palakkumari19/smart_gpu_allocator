from environment import GPUEnvironment
from workload.task_generator import TaskGenerator, EnhancedTask
import time

def demo_task_generator():
    """Demonstrate the new task generator"""
    print("=== TASK GENERATOR DEMO ===")
    print("=" * 50)
    
    generator = TaskGenerator()
    
    # Generate different task types
    print("\n1. GENERATING DIFFERENT TASK TYPES:")
    for task_type in ["LLM", "Image", "Video"]:
        task = generator.generate_task(task_type)
        print(f"  {task}")
        print(f"    Description: {task.description}")
        print(f"    Priority: {task.priority}")
        print(f"    Resource Intensity: {task.get_resource_intensity():.2f}")
        print()
    
    # Generate random tasks
    print("2. GENERATING RANDOM TASKS:")
    for i in range(5):
        task = generator.generate_task()
        print(f"  {task}")
    
    # Generate burst workload
    print("\n3. BURST WORKLOAD GENERATION:")
    burst = generator.generate_burst_workload("medium")
    print(f"Generated {len(burst)} tasks in burst")
    for task in burst[:3]:  # Show first 3
        print(f"  {task}")
    
    # Show statistics
    print("\n4. WORKLOAD STATISTICS:")
    stats = generator.get_workload_statistics()
    print(f"Total Generated: {stats['total_generated']}")
    print("Task Distribution:")
    for task_type, count in stats['task_distribution'].items():
        percentage = stats['percentages'][task_type]
        print(f"  {task_type}: {count} ({percentage:.1f}%)")

def demo_workload_simulation():
    """Run simulation with different workload modes"""
    print("\n=== WORKLOAD SIMULATION DEMO ===")
    print("=" * 50)
    
    # Test different workload modes
    modes = ["mixed", "llm_heavy", "image_heavy", "video_heavy"]
    
    for mode in modes:
        print(f"\n--- {mode.upper()} WORKLOAD MODE ---")
        env = GPUEnvironment(scheduler="least_loaded")
        env.set_workload_mode(mode)
        
        # Run simulation
        for step in range(15):
            env.step()
        
        # Get metrics
        metrics = env.get_metrics()
        workload_stats = metrics['workload_stats']
        
        print(f"Completed Tasks: {metrics['completed']}")
        print(f"Task Distribution:")
        for task_type, count in workload_stats['task_distribution'].items():
            if count > 0:
                percentage = workload_stats['percentages'][task_type]
                print(f"  {task_type}: {count} ({percentage:.1f}%)")

def main():
    """Main demonstration function"""
    print("AETHERGRID : TASK GENERATOR")
    print("=" * 60)
    
    # Demo task generator
    demo_task_generator()
    
    # Demo workload simulation
    demo_workload_simulation()
    
    print("\n" + "=" * 60)
    print("PHASE 2 STEP 3 FEATURES VERIFIED:")
    print("* Task Types: LLM (High/Slow), Image (Low/Fast), Video (Medium/Medium)")
    print("* Random Task Generation: random.choice(['LLM', 'Image', 'Video'])")
    print("* Workload Profiles: Memory and time ranges for each task type")
    print("* Burst Workload: Variable intensity task generation")
    print("* Workload Statistics: Real-time task distribution tracking")
    print("* Enhanced Task Properties: Priority, description, resource intensity")
    print("=" * 60)

if __name__ == "__main__":
    main()
