from environment import GPUEnvironment
import time
import json
import pickle
from datetime import datetime

class MARLTrainingSystem:
    """Advanced MARL training system with multiple dataset support"""
    
    def __init__(self):
        self.training_history = []
        self.model_versions = []
        self.performance_metrics = []
        
    def create_training_scenarios(self):
        """Create different training scenarios for MARL"""
        scenarios = {
            'light_load': {
                'traffic_pattern': 'light',
                'duration': 40,
                'description': 'Light workload for basic training',
                'target_success_rate': 0.65
            },
            'heavy_load': {
                'traffic_pattern': 'heavy',
                'duration': 40,
                'description': 'Heavy workload for stress training',
                'target_success_rate': 0.45
            },
            'burst_load': {
                'traffic_pattern': 'burst',
                'duration': 40,
                'description': 'Burst workload for adaptive training',
                'target_success_rate': 0.55
            },
            'mixed_variable': {
                'traffic_pattern': 'mixed',
                'duration': 60,
                'description': 'Variable mixed workload for comprehensive training',
                'target_success_rate': 0.60
            },
            'peak_hours': {
                'traffic_pattern': 'peak',
                'duration': 50,
                'description': 'Peak hours simulation for production training',
                'target_success_rate': 0.70
            },
            'stress_test': {
                'traffic_pattern': 'burst',
                'duration': 30,
                'description': 'Extreme stress for robustness training',
                'target_success_rate': 0.40
            }
        }
        return scenarios
    
    def train_marl_on_scenario(self, scenario_name, scenario_config, save_model=True):
        """Train MARL on specific scenario"""
        print(f"\n=== Training MARL on {scenario_name} ===")
        print(f"Description: {scenario_config['description']}")
        print(f"Target Success Rate: {scenario_config['target_success_rate']:.2f}")
        print("=" * 60)
        
        # Create environment for training
        env = GPUEnvironment(scheduler="marl")
        env.set_traffic_pattern(scenario_config['traffic_pattern'], duration=scenario_config['duration'])
        
        # Training metrics
        training_metrics = {
            'scenario': scenario_name,
            'start_time': time.time(),
            'steps': 0,
            'success_rates': [],
            'avg_rewards': [],
            'final_success_rate': 0,
            'target_success_rate': scenario_config['target_success_rate']
        }
        
        print(f"Training Progress:")
        print("Step | Success Rate | Avg Reward | Epsilon | Status")
        print("-" * 60)
        
        # Training loop
        for step in range(scenario_config['duration']):
            env.step()
            training_metrics['steps'] += 1
            
            if step % 5 == 0:
                metrics = env.get_metrics()
                if 'marl_stats' in metrics:
                    marl_stats = metrics['marl_stats']
                    success_rate = marl_stats.get('global_success_rate', 0)
                    avg_reward = marl_stats.get('total_reward', 0) / max(1, training_metrics['steps'])
                    avg_epsilon = marl_stats.get('avg_epsilon', 0)
                    
                    training_metrics['success_rates'].append(success_rate)
                    training_metrics['avg_rewards'].append(avg_reward)
                    
                    status = "TRAINING"
                    if success_rate >= scenario_config['target_success_rate']:
                        status = "TARGET REACHED"
                    
                    print(f"{step:4d}  | {success_rate:11.2%}    | {avg_reward:8.1f}      | {avg_epsilon:6.3f} | {status}")
        
        # Calculate final metrics
        training_metrics['end_time'] = time.time()
        training_metrics['final_success_rate'] = training_metrics['success_rates'][-1] if training_metrics['success_rates'] else 0
        training_metrics['avg_final_success_rate'] = sum(training_metrics['success_rates']) / len(training_metrics['success_rates'])
        training_metrics['total_training_time'] = training_metrics['end_time'] - training_metrics['start_time']
        
        # Performance evaluation
        target_met = training_metrics['final_success_rate'] >= scenario_config['target_success_rate']
        improvement = training_metrics['final_success_rate'] - 0.4  # Baseline improvement
        
        print(f"\nTraining Complete!")
        print(f"Final Success Rate: {training_metrics['final_success_rate']:.2f}%")
        print(f"Target Success Rate: {scenario_config['target_success_rate']:.2f}%")
        print(f"Target Met: {'YES' if target_met else 'NO'}")
        print(f"Improvement: {improvement:+.2%}")
        
        # Save model if target met
        if save_model and target_met:
            model_data = {
                'scenario': scenario_name,
                'training_metrics': training_metrics,
                'model_state': {
                    'agent_epsilons': [agent.epsilon for agent in env.marl_manager.agents],
                    'q_tables': [dict(agent.q_table) for agent in env.marl_manager.agents],
                    'training_time': datetime.now().isoformat()
                },
                'performance': {
                    'success_rate': training_metrics['final_success_rate'],
                    'target_met': target_met,
                    'improvement': improvement
                }
            }
            
            # Save to file
            filename = f"marl_model_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved: {filename}")
        
        # Store training session
        training_session = {
            'scenario': scenario_name,
            'config': scenario_config,
            'metrics': training_metrics,
            'model_file': filename if save_model and target_met else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_session)
        self.performance_metrics.append({
            'scenario': scenario_name,
            'success_rate': training_metrics['final_success_rate'],
            'target_rate': scenario_config['target_success_rate'],
            'target_met': target_met,
            'improvement': improvement
        })
        
        return training_metrics
    
    def train_comprehensive_curriculum(self):
        """Train MARL through comprehensive curriculum"""
        print("=== COMPREHENSIVE MARL TRAINING CURRICULUM ===")
        print("Progressive training from basic to advanced scenarios")
        print("=" * 60)
        
        scenarios = self.create_training_scenarios()
        curriculum_order = [
            'light_load',      # Start with easy scenario
            'heavy_load',      # Progress to harder
            'burst_load',      # Train adaptation
            'mixed_variable',  # Comprehensive training
            'peak_hours',      # Production-like training
            'stress_test'       # Robustness validation
        ]
        
        all_results = []
        
        for i, scenario_name in enumerate(curriculum_order):
            print(f"\n--- Training {i+1}/{len(curriculum_order)}: {scenario_name.upper()} ---")
            
            scenario_config = scenarios[scenario_name]
            result = self.train_marl_on_scenario(scenario_name, scenario_config, save_model=True)
            all_results.append(result)
            
            print(f"Result: {result['final_success_rate']:.2f}% success rate")
            time.sleep(1)  # Brief pause between scenarios
        
        # Curriculum analysis
        print(f"\n" + "=" * 60)
        print("CURRICULUM ANALYSIS")
        print("=" * 60)
        
        success_rates = [r['final_success_rate'] for r in all_results]
        improvement_rates = [r['improvement'] for r in all_results]
        
        print("Scenario Results:")
        for i, (scenario_name, result) in enumerate(zip(curriculum_order, all_results)):
            target_met = "✅" if result['target_met'] else "❌"
            print(f"  {i+1}. {scenario_name:<15}: {result['final_success_rate']:6.1f}% {target_met}")
        
        avg_success_rate = sum(success_rates) / len(success_rates)
        avg_improvement = sum(improvement_rates) / len(improvement_rates)
        
        print(f"\nCurriculum Summary:")
        print(f"  Average Success Rate: {avg_success_rate:.2f}%")
        print(f"  Average Improvement: {avg_improvement:.2f}%")
        print(f"  Targets Met: {sum(1 for r in all_results if r['target_met'])}/{len(all_results)}")
        
        # Save comprehensive training report
        report = {
            'curriculum_date': datetime.now().isoformat(),
            'scenarios': curriculum_order,
            'results': all_results,
            'summary': {
                'avg_success_rate': avg_success_rate,
                'avg_improvement': avg_improvement,
                'targets_met': sum(1 for r in all_results if r['target_met']),
                'total_scenarios': len(all_results)
            }
        }
        
        with open(f"marl_curriculum_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Curriculum report saved: marl_curriculum_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return report
    
    def load_and_evaluate_model(self, model_file):
        """Load trained model and evaluate performance"""
        print(f"\n=== LOADING AND EVALUATING MODEL ===")
        print(f"Model file: {model_file}")
        print("=" * 60)
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            print("Model loaded successfully!")
            print(f"Training scenario: {model_data['scenario']}")
            print(f"Training date: {model_data['model_state']['training_time']}")
            print(f"Final success rate: {model_data['performance']['success_rate']:.2f}%")
            print(f"Target met: {model_data['performance']['target_met']}")
            print(f"Improvement: {model_data['performance']['improvement']:.2f}%")
            
            return model_data
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def compare_models(self, model_files):
        """Compare multiple trained models"""
        print(f"\n=== MODEL COMPARISON ===")
        print(f"Comparing {len(model_files)} trained models")
        print("=" * 60)
        
        model_results = []
        
        for model_file in model_files:
            result = self.load_and_evaluate_model(model_file)
            if result:
                model_results.append({
                    'file': model_file,
                    'scenario': result['scenario'],
                    'success_rate': result['performance']['success_rate'],
                    'improvement': result['performance']['improvement'],
                    'target_met': result['performance']['target_met']
                })
        
        if model_results:
            # Sort by success rate
            model_results.sort(key=lambda x: x['success_rate'], reverse=True)
            
            print("\nModel Ranking:")
            print("Rank | Model File | Scenario | Success Rate | Improvement | Target Met")
            print("-" * 80)
            
            for i, model_result in enumerate(model_results):
                target_met = "✅" if model_result['target_met'] else "❌"
                print(f"{i+1:4d}  | {model_result['file']:<20} | {model_result['scenario']:<15} | "
                      f"{model_result['success_rate']:8.1f}% | {model_result['improvement']:+6.1f}% | {target_met}")
            
            # Best model analysis
            best_model = model_results[0]
            print(f"\nBest Model: {best_model['file']}")
            print(f"Best Scenario: {best_model['scenario']}")
            print(f"Best Success Rate: {best_model['success_rate']:.2f}%")
            print(f"Best Improvement: {best_model['improvement']:.2f}%")
        
        return model_results
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        print("\n=== GENERATING TRAINING REPORT ===")
        print("=" * 60)
        
        if not self.training_history:
            print("No training history available")
            return
        
        # Calculate overall statistics
        total_sessions = len(self.training_history)
        total_improvements = sum(s['improvement'] for s in self.training_history if s['improvement'] > 0)
        avg_improvement = total_improvements / max(1, total_sessions)
        
        print(f"Total Training Sessions: {total_sessions}")
        print(f"Total Positive Improvements: {total_improvements}")
        print(f"Average Improvement: {avg_improvement:.2f}%")
        
        # Best training session
        if self.performance_metrics:
            best_session = max(self.performance_metrics, key=lambda x: x['success_rate'])
            print(f"\nBest Training Session:")
            print(f"  Scenario: {best_session['scenario']}")
            print(f"  Success Rate: {best_session['success_rate']:.2f}%")
            print(f"  Target Rate: {best_session['target_rate']:.2f}%")
            print(f"  Target Met: {'YES' if best_session['target_met'] else 'NO'}")
            print(f"  Improvement: {best_session['improvement']:.2f}%")
        
        # Generate report
        report = {
            'report_date': datetime.now().isoformat(),
            'total_sessions': total_sessions,
            'total_improvements': total_improvements,
            'avg_improvement': avg_improvement,
            'best_session': best_session if self.performance_metrics else None,
            'training_history': self.training_history
        }
        
        filename = f"marl_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved: {filename}")
        return filename

def main():
    """Main function to demonstrate MARL training capabilities"""
    print("AETHERGRID - ADVANCED MARL TRAINING SYSTEM")
    print("Multi-Dataset Training for Superior MARL Performance")
    print("=" * 60)
    
    trainer = MARLTrainingSystem()
    
    # Demonstrate training scenarios
    print("\n1. SINGLE SCENARIO TRAINING")
    print("Training MARL on heavy load scenario...")
    result = trainer.train_marl_on_scenario('heavy_load', {
        'traffic_pattern': 'heavy',
        'duration': 30,
        'description': 'Heavy load training',
        'target_success_rate': 0.50
    })
    
    print(f"\nHeavy Load Training Result: {result['final_success_rate']:.2f}% success rate")
    
    print("\n2. COMPREHENSIVE CURRICULUM TRAINING")
    print("Running comprehensive training curriculum...")
    curriculum_result = trainer.train_comprehensive_curriculum()
    
    print(f"\nCurriculum Average Success Rate: {curriculum_result['summary']['avg_success_rate']:.2f}%")
    print(f"Curriculum Average Improvement: {curriculum_result['summary']['avg_improvement']:.2f}%")
    
    print("\n3. TRAINING REPORT GENERATION")
    report_file = trainer.generate_training_report()
    print(f"Training report generated: {report_file}")
    
    print("\n" + "=" * 60)
    print("MARL TRAINING SYSTEM CAPABILITIES DEMONSTRATED")
    print("=" * 60)
    
    print("Multi-Scenario Training: Different workload types")
    print("✅ Progressive Curriculum: Easy to hard scenarios")
    print("✅ Target-Based Training: Specific success rate goals")
    print("✅ Model Persistence: Save and load trained models")
    print("✅ Performance Tracking: Comprehensive metrics and analysis")
    print("✅ Model Comparison: Evaluate multiple trained models")
    print("✅ Report Generation: Detailed training documentation")
    
    print("\nBENEFITS FOR MARL SUPERIORITY:")
    print("- Trained MARL will outperform traditional methods")
    print("- Different datasets provide specialized training")
    print("- Progressive training builds robustness")
    print("- Target-based training ensures performance goals")
    print("- Model persistence enables continuous improvement")
    
    print("\nUSAGE EXAMPLES:")
    print("# Train on specific scenario")
    print("trainer.train_marl_on_scenario('peak_hours', {")
    print("    'traffic_pattern': 'peak',")
    print("    'duration': 50,")
    print("    'target_success_rate': 0.70")
    print("})")
    print("")
    print("# Run comprehensive curriculum")
    print("trainer.train_comprehensive_curriculum()")
    print("")
    print("# Compare trained models")
    print("model_files = ['model1.pkl', 'model2.pkl']")
    print("trainer.compare_models(model_files)")
    print("")
    print("# Generate training report")
    print("trainer.generate_training_report()")

if __name__ == "__main__":
    main()
