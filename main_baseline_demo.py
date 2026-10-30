from environment import GPUEnvironment
from schedulers.traditional import BaselineComparison
import time

def demo_traditional_schedulers():
    """Demonstrate traditional scheduling algorithms"""
    print("=== TRADITIONAL SCHEDULERS DEMO ===")
    print("Phase 3 Step 5: Implement Traditional Algorithms")
    print("=" * 60)
    
    # Test each traditional scheduler
    traditional_schedulers = [
        ("traditional_fcfs", "FCFS"),
        ("traditional_round_robin", "Round Robin"),
        ("traditional_least_loaded", "Least Loaded")
    ]
    
    for scheduler_name, display_name in traditional_schedulers:
        print(f"\n--- {display_name} Scheduler ---")
        env = GPUEnvironment(scheduler=scheduler_name)
        env.set_traffic_pattern("mixed", duration=15)
        
        # Run simulation
        for step in range(15):
            env.step()
        
        # Get metrics
        metrics = env.get_metrics()
        traditional_stats = metrics['traditional_stats']
        
        print(f"Completed Tasks: {metrics['completed']}")
        print(f"Assignment Rate: {traditional_stats.get(scheduler_name.split('_')[1], {}).get('assignment_rate', 0):.2%}")
        print(f"Memory Usage: {metrics['avg_memory_usage']:.1f}%")
        print(f"Latency: {metrics['latency']}")
        
        # Show scheduler-specific stats
        if scheduler_name in traditional_stats:
            stats = traditional_stats[scheduler_name]
            print(f"Decisions Made: {stats.get('total_decisions', 0)}")
            print(f"Tasks Assigned: {stats.get('total_tasks_assigned', 0)}")

def baseline_comparison():
    """Compare all traditional schedulers as baseline"""
    print("\n=== BASELINE COMPARISON ===")
    print("Comparing traditional algorithms for baseline performance")
    print("=" * 60)
    
    # Create environment for comparison
    env = GPUEnvironment(scheduler="traditional_round_robin")
    
    # Generate test workload
    test_tasks = []
    for _ in range(25):
        test_tasks.append(env.task_generator.generate_task())
    
    print(f"Testing with {len(test_tasks)} tasks:")
    task_types = {}
    for task in test_tasks:
        task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
    
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count}")
    
    # Compare all traditional schedulers
    comparison_results = env.compare_traditional_schedulers(test_tasks)
    
    # Print comparison
    print(f"\nBaseline Comparison Results:")
    print("-" * 60)
    
    for scheduler_name, result in comparison_results.items():
        success_rate = result['success_rate'] * 100
        assignments = result['successful_assignments']
        total = result['total_tasks']
        
        print(f"{scheduler_name:<15}: {success_rate:>6.1f}% success rate "
              f"({assignments}/{total} tasks assigned)")
    
    # Find best performer
    best_scheduler = max(comparison_results.keys(), 
                       key=lambda k: comparison_results[k]['success_rate'])
    best_rate = comparison_results[best_scheduler]['success_rate'] * 100
    
    print(f"\nBaseline Best Performer: {best_scheduler}")
    print(f"Success Rate: {best_rate:.1f}%")
    
    return comparison_results

def compare_all_scheduler_types():
    """Compare traditional vs RL vs MARL schedulers"""
    print("\n=== COMPREHENSIVE SCHEDULER COMPARISON ===")
    print("Traditional vs RL vs MARL - Complete Comparison")
    print("=" * 60)
    
    # All scheduler types
    all_schedulers = [
        ("random", "Random"),
        ("fcfs", "FCFS (Original)"),
        ("traditional_fcfs", "FCFS (Traditional)"),
        ("round_robin", "Round Robin (Original)"),
        ("traditional_round_robin", "Round Robin (Traditional)"),
        ("least_loaded", "Least Loaded (Original)"),
        ("traditional_least_loaded", "Least Loaded (Traditional)"),
        ("rl", "RL (Single Agent)"),
        ("marl", "MARL (Multi Agent)")
    ]
    
    results = {}
    
    for scheduler_name, display_name in all_schedulers:
        print(f"\nTesting {display_name}...")
        env = GPUEnvironment(scheduler=scheduler_name)
        env.set_traffic_pattern("mixed", duration=20)
        
        # Run simulation
        start_time = time.time()
        for step in range(20):
            env.step()
        end_time = time.time()
        
        # Get metrics
        metrics = env.get_metrics()
        
        results[display_name] = {
            'completed': metrics['completed'],
            'crashes': metrics['crashes'],
            'memory_usage': metrics['avg_memory_usage'],
            'latency': metrics['latency'],
            'execution_time': end_time - start_time,
            'throughput': metrics['completed'] / (end_time - start_time)
        }
        
        # Add scheduler-specific metrics
        if 'marl_success_rate' in metrics:
            results[display_name]['marl_success_rate'] = metrics['marl_success_rate']
        if 'traditional_stats' in metrics:
            traditional_stats = metrics['traditional_stats']
            scheduler_key = scheduler_name.split('_')[-1] if '_' in scheduler_name else scheduler_name
            if scheduler_key in traditional_stats:
                results[display_name]['assignment_rate'] = traditional_stats[scheduler_key].get('assignment_rate', 0)
    
    # Print comprehensive comparison table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SCHEDULER COMPARISON")
    print("=" * 80)
    print(f"{'Scheduler':<20} {'Completed':<10} {'Memory%':<8} {'Latency':<8} {'Throughput':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['completed']:<10} "
              f"{result['memory_usage']:<8.1f} {result['latency']:<8} "
              f"{result['throughput']:<12.2f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Best performers
    best_completed = max(results.keys(), key=lambda k: results[k]['completed'])
    best_memory = min(results.keys(), key=lambda k: results[k]['memory_usage'])
    best_latency = min(results.keys(), key=lambda k: results[k]['latency'])
    best_throughput = max(results.keys(), key=lambda k: results[k]['throughput'])
    
    print(f"Most Tasks Completed: {best_completed} ({results[best_completed]['completed']} tasks)")
    print(f"Best Memory Usage: {best_memory} ({results[best_memory]['memory_usage']:.1f}%)")
    print(f"Lowest Latency: {best_latency} ({results[best_latency]['latency']})")
    print(f"Highest Throughput: {best_throughput} ({results[best_throughput]['throughput']:.2f} tasks/sec)")
    
    # Traditional vs Advanced comparison
    traditional_schedulers = [k for k in results.keys() if 'Traditional' in k]
    advanced_schedulers = [k for k in results.keys() if 'Traditional' not in k and k not in ['Random']]
    
    if traditional_schedulers and advanced_schedulers:
        trad_avg = sum(results[k]['completed'] for k in traditional_schedulers) / len(traditional_schedulers)
        adv_avg = sum(results[k]['completed'] for k in advanced_schedulers) / len(advanced_schedulers)
        
        print(f"\nTraditional vs Advanced:")
        print(f"Traditional Average: {trad_avg:.1f} tasks completed")
        print(f"Advanced Average: {adv_avg:.1f} tasks completed")
        print(f"Improvement: {((adv_avg - trad_avg) / trad_avg * 100):+.1f}%")
    
    return results

def demonstrate_scheduler_algorithms():
    """Demonstrate the logic of each traditional algorithm"""
    print("\n=== SCHEDULER ALGORITHM DEMONSTRATION ===")
    print("Showing how each traditional algorithm makes decisions")
    print("=" * 60)
    
    from schedulers.traditional import FCFSScheduler, RoundRobinScheduler, LeastLoadedScheduler
    
    # Create schedulers
    fcfs = FCFSScheduler()
    rr = RoundRobinScheduler()
    ll = LeastLoadedScheduler()
    
    print("Algorithm Descriptions:")
    print(f"\nFCFS: {fcfs.get_description()}")
    print(f"Round Robin: {rr.get_description()}")
    print(f"Least Loaded: {ll.get_description()}")
    
    # Create mock GPU states for demonstration
    print(f"\nMock GPU States for Decision Making:")
    print("GPU 0: Available, 50% memory, 2 tasks in queue")
    print("GPU 1: Available, 80% memory, 0 tasks in queue")
    print("GPU 2: Crashed, 30% memory, 1 task in queue")
    print("GPU 3: Cooldown, 60% memory, 3 tasks in queue")
    
    # Create mock task
    from workload.task_generator import TaskGenerator
    generator = TaskGenerator()
    mock_task = generator.generate_task("LLM")
    
    print(f"\nMock Task: {mock_task.task_type}, {mock_task.memory_required}GB, {mock_task.execution_time} steps")
    
    # Show decisions (would need actual GPU objects for real implementation)
    print(f"\nAlgorithm Decisions:")
    print("FCFS: Would select GPU 0 (first available)")
    print("Round Robin: Would select next GPU in rotation")
    print("Least Loaded: Would select GPU 1 (lowest memory + queue load)")

def main():
    """Main demonstration function"""
    print("AETHERGRID : BASELINE SCHEDULERS")
    print("Phase 3 Step 5: Implement Traditional Algorithms")
    print("=" * 60)
    
    # Demonstrate traditional schedulers
    demo_traditional_schedulers()
    
    # Baseline comparison
    baseline_comparison()
    
    # Compare all scheduler types
    compare_all_scheduler_types()
    
    # Demonstrate algorithms
    demonstrate_scheduler_algorithms()
    
    print("\n" + "=" * 60)
    print("PHASE 3 STEP 5 COMPLETE")
    print("=" * 60)
    print("Traditional Scheduling Algorithms Implemented:")
    print("* FCFS: First-Come, First-Served")
    print("* Round Robin: Cyclic GPU assignment")
    print("* Least Loaded: Minimum load selection")
    print("* Baseline Comparison: Performance metrics")
    print("* Integration: Works with all existing features")
    print("=" * 60)

if __name__ == "__main__":
    main()
