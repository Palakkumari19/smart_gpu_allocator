import json
from execution.real_gpu_executor import RealGPUVideoExecutor

def main():
    print("Initializing RealGPUVideoExecutor on cuda:0...")
    executor = RealGPUVideoExecutor(
        device="cuda:0",
        batch_size=8,
    )
    
    print("Starting video processing workload...")
    result = executor.process_video(
        input_path="data/sample.mp4",
        output_path="outputs/sample_processed.mp4",
        task_id="video-test-001",
    )
    
    print("\nExecution Complete!")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
