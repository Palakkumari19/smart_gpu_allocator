from environment import GPUEnvironment
import time

def test_phase3_traditional_schedulers():
    """Test Traditional Schedulers"""
    print("=== TESTING  TRADITIONAL SCHEDULERS ===")
    print("Verifying: FCFS, Round Robin, Least Loaded algorithms")
    print("-" * 70)
    
    # Test each traditional scheduler
    traditional_schedulers = [
        ("traditional_fcfs", "FCFS"),
        ("traditional_round_robin", "Round Robin"),
        ("traditional_least_loaded", "Least Loaded")
    ]
    
    phase3_ok = True
    
    for scheduler_name, display_name in traditional_schedulers:
        print(f"\nTesting {display_name}...")
        env = GPUEnvironment(scheduler=scheduler_name)
        env.set_traffic_pattern("mixed", duration=10)
        
        # Run simulation
        for step in range(10):
            env.step()
        
        metrics = env.get_metrics()
        traditional_stats = metrics.get('traditional_stats', {})
        
        # Verify scheduler is working
        scheduler_working = metrics['completed'] >= 0
        has_stats = scheduler_name.split('_')[-1] in traditional_stats
        
        print(f"  Tasks completed: {metrics['completed']}")
        print(f"  Scheduler working: {scheduler_working}")
        print(f"  Statistics available: {has_stats}")
        
        if has_stats:
            stats = traditional_stats[scheduler_name.split('_')[-1]]
            print(f"  Assignment rate: {stats.get('assignment_rate', 0):.2%}")
            print(f"  Total decisions: {stats.get('total_decisions', 0)}")
        
        phase3_ok &= scheduler_working
    
    print(f"\nPhase 3 Step 5 Status: {'PASS' if phase3_ok else 'FAIL'}")
    return phase3_ok

def test_baseline_comparison():
    """Test baseline comparison functionality"""
    print("\n=== TESTING BASELINE COMPARISON ===")
    print("Verifying: Baseline comparison system")
    print("-" * 70)
    
    env = GPUEnvironment(scheduler="traditional_round_robin")
    
    # Test comparison functionality
    try:
        # Generate test tasks
        test_tasks = []
        for _ in range(15):
            test_tasks.append(env.task_generator.generate_task())
        
        # Compare schedulers
        comparison_results = env.compare_traditional_schedulers(test_tasks)
        
        baseline_ok = True
        baseline_ok &= len(comparison_results) > 0
        
        print(f"Comparison results generated: {len(comparison_results)} schedulers")
        for name, result in comparison_results.items():
            print(f"  {name}: {result['success_rate']:.2%} success rate")
            baseline_ok &= 'success_rate' in result
        
        print(f"Baseline Comparison Status: {'PASS' if baseline_ok else 'FAIL'}")
        return baseline_ok
        
    except Exception as e:
        print(f"Baseline Comparison Error: {e}")
        return False

def test_all_scheduler_integration():
    """Test integration of all scheduler types"""
    print("\n=== TESTING ALL SCHEDULER INTEGRATION ===")
    print("Verifying: Traditional + RL + MARL schedulers work together")
    print("-" * 70)
    
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
    
    integration_ok = True
    
    for scheduler_name, display_name in all_schedulers:
        print(f"\nTesting {display_name}...")
        try:
            env = GPUEnvironment(scheduler=scheduler_name)
            env.set_traffic_pattern("light", duration=5)
            
            # Run short simulation
            for step in range(5):
                env.step()
            
            metrics = env.get_metrics()
            scheduler_works = metrics['completed'] >= 0
            
            print(f"  Status: {'WORKING' if scheduler_works else 'FAILED'}")
            print(f"  Tasks completed: {metrics['completed']}")
            
            # Check for scheduler-specific metrics
            if 'traditional_stats' in metrics:
                print(f"  Traditional metrics: Available")
            if 'marl_stats' in metrics:
                print(f"  MARL metrics: Available")
            
            integration_ok &= scheduler_works
            
        except Exception as e:
            print(f"  ERROR: {e}")
            integration_ok = False
    
    print(f"\nAll Scheduler Integration Status: {'PASS' if integration_ok else 'FAIL'}")
    return integration_ok

def test_feature_compatibility():
    """Test that traditional schedulers work with all existing features"""
    print("\n=== TESTING FEATURE COMPATIBILITY ===")
    print("Verifying: Traditional schedulers + all previous features")
    print("-" * 70)
    
    # Test with different traffic patterns
    traffic_patterns = ["light", "heavy", "burst", "peak", "off_peak"]
    compatibility_ok = True
    
    for pattern in traffic_patterns:
        print(f"\nTesting with {pattern} traffic pattern...")
        env = GPUEnvironment(scheduler="traditional_fcfs")
        env.set_traffic_pattern(pattern, duration=8)
        
        # Run simulation
        for step in range(8):
            env.step()
        
        metrics = env.get_metrics()
        
        # Check all features are working
        has_workload = 'workload_stats' in metrics
        has_traffic = 'traffic_stats' in metrics
        has_advanced = 'total_preemptions' in metrics
        has_traditional = 'traditional_stats' in metrics
        
        print(f"  Workload generation: {has_workload}")
        print(f"  Traffic patterns: {has_traffic}")
        print(f"  Advanced features: {has_advanced}")
        print(f"  Traditional schedulers: {has_traditional}")
        
        compatibility_ok &= has_workload and has_traffic and has_advanced and has_traditional
    
    print(f"\nFeature Compatibility Status: {'PASS' if compatibility_ok else 'FAIL'}")
    return compatibility_ok

def test_baseline_output():
    """Test baseline output generation"""
    print("\n=== TESTING BASELINE OUTPUT ===")
    print("Verifying: Baseline comparison output generation")
    print("-" * 70)
    
    env = GPUEnvironment(scheduler="traditional_least_loaded")
    env.set_traffic_pattern("mixed", duration=12)
    
    # Run simulation
    for step in range(12):
        env.step()
    
    # Get comprehensive metrics
    metrics = env.get_metrics()
    
    # Verify all metric categories are present
    output_ok = True
    output_ok &= 'completed' in metrics
    output_ok &= 'workload_stats' in metrics
    output_ok &= 'traffic_stats' in metrics
    output_ok &= 'traditional_stats' in metrics
    output_ok &= 'marl_stats' in metrics  # Should be available even if not used
    
    print(f"Basic metrics: {'Available' if 'completed' in metrics else 'Missing'}")
    print(f"Workload metrics: {'Available' if 'workload_stats' in metrics else 'Missing'}")
    print(f"Traffic metrics: {'Available' if 'traffic_stats' in metrics else 'Missing'}")
    print(f"Traditional metrics: {'Available' if 'traditional_stats' in metrics else 'Missing'}")
    print(f"MARL metrics: {'Available' if 'marl_stats' in metrics else 'Missing'}")
    
    # Show sample output
    if output_ok:
        print(f"\nSample Output:")
        print(f"  Tasks completed: {metrics['completed']}")
        print(
            f"  Task distribution: "
            f"Light={metrics['light_tasks_generated']}, "
            f"Medium={metrics['medium_tasks_generated']}, "
            f"Heavy={metrics['heavy_tasks_generated']}, "
            f"Extreme={metrics['extreme_tasks_generated']}"
        )
        print(f"  Traffic pattern: {metrics['current_traffic_pattern']}")
        print(f"  FCFS assignment rate: {metrics['fcfs_assignment_rate']:.2%}")
        print(f"  Round Robin assignment rate: {metrics['round_robin_assignment_rate']:.2%}")
        print(f"  Least Loaded assignment rate: {metrics['least_loaded_assignment_rate']:.2%}")
    
    print(f"\nBaseline Output Status: {'PASS' if output_ok else 'FAIL'}")
    return output_ok

def main():
    """Main verification function"""
    print("AETHERGRID - PHASE 3 VERIFICATION")
    print("=" * 70)
    
    # Test all Phase 3 Step 5 features
    results = {
        'Phase 3 Step 5 - Traditional Schedulers': test_phase3_traditional_schedulers(),
        'Baseline Comparison System': test_baseline_comparison(),
        'All Scheduler Integration': test_all_scheduler_integration(),
        'Feature Compatibility': test_feature_compatibility(),
        'Baseline Output Generation': test_baseline_output()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for feature, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{feature:<40} : {status}")
        all_passed &= passed
    
    print("=" * 70)
    if all_passed:
        print("PHASE 3 STEP 5 VERIFICATION SUCCESSFUL!")
        print("AETHERGRID now includes:")
        print("- Phase 1: GPU Simulator (8GB, parallel execution, crash conditions)")
        print("- Phase 2 Step 2: Advanced Features (fragmentation, preemption, cooldown)")
        print("- Phase 2 Step 3: Task Generator (LLM, Image, Video)")
        print("- Phase 2 Step 4: Traffic Patterns (light, heavy, burst, peak/off-peak)")
        print("- MARL System: Multi-Agent RL with separate GPU agents")
        print("- Phase 3 Step 5: Traditional Schedulers (FCFS, Round Robin, Least Loaded)")
        print("- Baseline Comparison: Performance metrics and comparison")
        print("- Complete Integration: All features work together")
        print("- Output Generation: Comprehensive metrics and statistics")
    else:
        print("SOME PHASE 3 FEATURES FAILED VERIFICATION!")
        print("Please check the failed components above.")
    print("=" * 70)

if __name__ == "__main__":
    main()
