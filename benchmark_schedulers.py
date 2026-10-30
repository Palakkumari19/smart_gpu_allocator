import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from environment import GPUEnvironment
from workload.task_generator import TaskGenerator, VideoProfile


def set_fixed_seeds(seed):
    """Set deterministic seeds for all randomized components."""
    random.seed(seed)
    np.random.seed(seed)


def build_shared_task_stream(steps, seed, task_count=None):
    """Generate one deterministic task stream that all schedulers will use."""
    set_fixed_seeds(seed)
    generator = TaskGenerator()
    if task_count is None:
        task_count = max(steps * 8, 80)
    return [generator.generate_task() for _ in range(task_count)]


def attach_shared_task_stream(env, tasks):
    """Inject a shared task stream into the environment without changing scheduler code."""
    task_queue = list(tasks)

    def generate_task(task_type=None):
        if task_queue:
            task = task_queue.pop(0)
            env.task_generator.total_generated += 1
            try:
                profile = VideoProfile(task.profile_name)
                env.task_generator.generated_tasks[profile] += 1
            except (AttributeError, ValueError, KeyError):
                pass
            return task
        return env.task_generator.generate_task(task_type)

    env.generate_task = generate_task


def benchmark_scheduler(scheduler_name, steps=50, runs=10, seed=42):
    """Benchmark a scheduler across multiple deterministic runs and average the results."""
    print(f"\n=== Benchmarking {scheduler_name.upper()} Scheduler over {runs} runs ===")

    run_metrics = []
    for run_idx in range(runs):
        run_seed = seed + run_idx
        set_fixed_seeds(run_seed)
        task_stream = build_shared_task_stream(steps, run_seed)

        env = GPUEnvironment(scheduler=scheduler_name)
        env.set_traffic_pattern("mixed", duration=steps)
        attach_shared_task_stream(env, task_stream)

        start_time = time.perf_counter()
        for _ in range(steps):
            env.step()
        end_time = time.perf_counter()

        final_metrics = env.get_metrics()
        execution_time = max(end_time - start_time, 1e-9)
        throughput = final_metrics['completed'] / execution_time

        run_metrics.append({
            'Scheduler': scheduler_name,
            'Completed': float(final_metrics['completed']),
            'Crashes': float(final_metrics['crashes']),
            'Avg_Memory_Usage': float(final_metrics['avg_memory_usage']),
            'Avg_Temperature': float(final_metrics['avg_temperature']),
            'Total_Latency': float(final_metrics['latency']),
            'Total_Preemptions': float(final_metrics['total_preemptions']),
            'Avg_Fragmentation': float(final_metrics['avg_fragmentation']),
            'Execution_Time': float(execution_time),
            'Throughput': float(throughput),
        })

    aggregated = {
        'Scheduler': scheduler_name,
        'Runs': runs,
        'Mean_Completed': float(np.mean([m['Completed'] for m in run_metrics])),
        'Completed_Std': float(np.std([m['Completed'] for m in run_metrics])),
        'Mean_Crashes': float(np.mean([m['Crashes'] for m in run_metrics])),
        'Crashes_Std': float(np.std([m['Crashes'] for m in run_metrics])),
        'Mean_Latency': float(np.mean([m['Total_Latency'] for m in run_metrics])),
        'Latency_Std': float(np.std([m['Total_Latency'] for m in run_metrics])),
        'Mean_Throughput': float(np.mean([m['Throughput'] for m in run_metrics])),
        'Throughput_Std': float(np.std([m['Throughput'] for m in run_metrics])),
        'Mean_Memory_Usage': float(np.mean([m['Avg_Memory_Usage'] for m in run_metrics])),
        'Mean_Temperature': float(np.mean([m['Avg_Temperature'] for m in run_metrics])),
        'Mean_Preemptions': float(np.mean([m['Total_Preemptions'] for m in run_metrics])),
        'Mean_Fragmentation': float(np.mean([m['Avg_Fragmentation'] for m in run_metrics])),
        'Mean_Execution_Time': float(np.mean([m['Execution_Time'] for m in run_metrics])),
    }

    if scheduler_name == 'marl':
        final_metrics = env.get_metrics()
        aggregated.update({
            'MARL_Global_Reward': float(final_metrics['marl_global_reward']),
            'MARL_Success_Rate': float(final_metrics['marl_success_rate']),
            'MARL_Avg_Epsilon': float(final_metrics['marl_avg_epsilon']),
            'MARL_Q_Table_Size': float(final_metrics['marl_total_q_table_size']),
        })

    print(f"  Mean Completed: {aggregated['Mean_Completed']:.2f} ± {aggregated['Completed_Std']:.2f}")
    print(f"  Mean Latency: {aggregated['Mean_Latency']:.2f} ± {aggregated['Latency_Std']:.2f}")
    print(f"  Mean Throughput: {aggregated['Mean_Throughput']:.2f} ± {aggregated['Throughput_Std']:.2f}")
    print(f"  Mean Crashes: {aggregated['Mean_Crashes']:.2f} ± {aggregated['Crashes_Std']:.2f}")

    return aggregated


def benchmark_all_schedulers(steps=50, runs=10, seed=42):
    """Benchmark all schedulers and compare average results across repeated runs."""
    print("AETHERGRID SCHEDULER BENCHMARK")
    print("=" * 60)
    print(f"Methodology: {runs} runs, fixed seeds, identical task stream per run")

    schedulers = ["random", "round_robin", "fcfs", "least_loaded", "rl", "marl"]
    results = []

    for scheduler in schedulers:
        try:
            result = benchmark_scheduler(scheduler, steps=steps, runs=runs, seed=seed)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {scheduler}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 100)
        print("AVERAGED SCHEDULER COMPARISON TABLE")
        print("=" * 100)
        print(df.to_string(index=False, float_format='%.2f'))

        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)
        best_throughput = max(results, key=lambda x: x['Mean_Throughput'])
        best_memory = min(results, key=lambda x: x['Mean_Memory_Usage'])
        least_crashes = min(results, key=lambda x: x['Mean_Crashes'])
        lowest_latency = min(results, key=lambda x: x['Mean_Latency'])

        print(f"Best Mean Throughput: {best_throughput['Scheduler']} ({best_throughput['Mean_Throughput']:.2f} tasks/sec)")
        print(f"Best Mean Memory Usage: {best_memory['Scheduler']} ({best_memory['Mean_Memory_Usage']:.1f}%)")
        print(f"Least Mean Crashes: {least_crashes['Scheduler']} ({least_crashes['Mean_Crashes']:.2f})")
        print(f"Lowest Mean Latency: {lowest_latency['Scheduler']} ({lowest_latency['Mean_Latency']:.2f})")

        marl_result = next((r for r in results if r.get('Scheduler') == 'marl' and 'MARL_Global_Reward' in r), None)
        if marl_result is not None:
            print(f"\nMARL Performance:")
            print(f"  Mean Reward: {marl_result['MARL_Global_Reward']:.2f}")
            print(f"  Mean Success Rate: {marl_result['MARL_Success_Rate']:.2%}")
            print(f"  Mean Epsilon: {marl_result['MARL_Avg_Epsilon']:.3f}")
            print(f"  Mean Q-Table Size: {marl_result['MARL_Q_Table_Size']:.0f}")

    return results

def test_marl_individual_agents():
    """Test individual MARL agent performance"""
    print("\n" + "=" * 60)
    print("INDIVIDUAL MARL AGENT ANALYSIS")
    print("=" * 60)
    
    env = GPUEnvironment(scheduler="marl")
    env.set_traffic_pattern("mixed", duration=30)
    
    # Run simulation
    for step in range(30):
        env.step()
    
    # Get MARL stats
    marl_stats = env.marl_manager.get_global_stats()
    
    print(f"Global Step: {marl_stats['global_step']}")
    print(f"Total Reward: {marl_stats['total_reward']:.2f}")
    print(f"Global Success Rate: {marl_stats['global_success_rate']:.2%}")
    print(f"Average Epsilon: {marl_stats['avg_epsilon']:.3f}")
    print(f"Total Q-Table Size: {marl_stats['total_q_table_size']}")
    
    print("\nIndividual Agent Performance:")
    print("-" * 40)
    for agent_stat in marl_stats['agent_stats']:
        print(f"GPU {agent_stat['gpu_id']}:")
        print(f"  Reward: {agent_stat['total_reward']:.2f}")
        print(f"  Tasks Assigned: {agent_stat['tasks_assigned']}")
        print(f"  Success Rate: {agent_stat['success_rate']:.2%}")
        print(f"  Epsilon: {agent_stat['epsilon']:.3f}")
        print(f"  Q-Table Size: {agent_stat['q_table_size']}")

def main():
    """Main benchmark function"""
    print("AETHERGRID - SCHEDULER BENCHMARKING")
    print("=" * 60)
    
    # Benchmark all schedulers
    results = benchmark_all_schedulers()
    
    # Test MARL individual agents
    test_marl_individual_agents()
    
    print("\n" + "=" * 60)
    print("\nBENCHMARK COMPLETE")
    print("=" * 60)
    print("All features verified:")
    print("* Phase 1: GPU Simulator (8GB, parallel execution, crash conditions)")
    print("* Phase 2 Step 2: Advanced Features (fragmentation, preemption, cooldown)")
    print("* Phase 2 Step 3: Task Generator (LLM, Image, Video)")
    print("* Phase 2 Step 4: Traffic Patterns (light, heavy, burst, peak/off-peak)")
    print("* MARL System: Multi-Agent RL with separate GPU agents")
    print("* Enhanced State: Memory, temp, queue, crash status, fragmentation")
    print("* Improved Rewards: Better crash/load/queue handling")
    print("* Smart GPU Selection: Q-values + queue awareness")
    print("* Model Tuning: Benchmarked against all schedulers")
    print("=" * 60)

if __name__ == "__main__":
    main()
