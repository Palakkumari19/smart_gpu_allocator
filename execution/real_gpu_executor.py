"""
Real Physical GPU Video Execution Pipeline using PyTorch and OpenCV.
Decodes video frames, batches them, executes frame-filtering tensor operations on cuda:0,
and encodes processed frames into an output video file.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Callable, List
import numpy as np
import cv2
import torch
import torch.nn.functional as F

logger = logging.getLogger("aethergrid.real_gpu_executor")


class RealGPUVideoExecutor:
    """
    Executes actual hardware-accelerated video transformation workloads on a physical GPU.
    Decodes video via OpenCV, batches frames, transfers them to CUDA VRAM as PyTorch tensors,
    applies convolutional filtering and tensor operations, and encodes the output back to disk.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        default_batch_size: int = 16,
        batch_size: Optional[int] = None,
        filter_type: str = "gaussian_blur",
        fallback_to_cpu: bool = False,
    ):
        """
        Initialize the Real GPU Video Executor.

        Args:
            device: Target execution device (e.g. 'cuda:0', 'cuda', 'cpu')
            default_batch_size: Default number of frames processed concurrently in VRAM
            batch_size: Optional alias for default_batch_size
            filter_type: Tensor operation ('gaussian_blur', 'sharpen', 'edge_detect', 'color_enhance')
            fallback_to_cpu: If True, falls back to CPU when CUDA is requested but unavailable
        """
        self.default_batch_size = batch_size if batch_size is not None else default_batch_size
        self.batch_size = self.default_batch_size
        self.filter_type = filter_type
        self.fallback_to_cpu = fallback_to_cpu

        if self.default_batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {self.default_batch_size}")

        # CUDA availability check and device resolution
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                if self.fallback_to_cpu:
                    logger.warning("CUDA is requested but unavailable. Falling back to CPU.")
                    self.device = torch.device("cpu")
                else:
                    raise RuntimeError(
                        f"CUDA is requested ({device}) but CUDA is not available on this system. "
                        "Please verify your NVIDIA drivers and PyTorch CUDA build."
                    )
            else:
                try:
                    self.device = torch.device(device)
                    # Test device allocation
                    _test_tensor = torch.zeros((1,), device=self.device)
                    del _test_tensor
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize CUDA device '{device}': {e}")
        else:
            self.device = torch.device(device)

        # Precompute convolutional filtering kernels on target device
        self._init_kernels()

    def _init_kernels(self):
        """Construct filter kernels for 3-channel (RGB/BGR) tensor convolutions."""
        if self.filter_type == "gaussian_blur":
            # 5x5 2D Gaussian Kernel
            sigma = 1.5
            ksize = 5
            ax = torch.arange(-(ksize // 2), ksize // 2 + 1, dtype=torch.float32)
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel_2d = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
            kernel_2d = kernel_2d / kernel_2d.sum()
            # Shape for depthwise conv2d: (channels, 1, ksize, ksize)
            self.kernel = kernel_2d.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1).to(self.device)
            self.padding = ksize // 2
        elif self.filter_type == "sharpen":
            # 3x3 unsharp mask kernel
            kernel_2d = torch.tensor(
                [[0.0, -1.0, 0.0],
                 [-1.0, 5.0, -1.0],
                 [0.0, -1.0, 0.0]],
                dtype=torch.float32
            )
            self.kernel = kernel_2d.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)
            self.padding = 1
        elif self.filter_type == "edge_detect":
            # 3x3 Laplacian edge detector
            kernel_2d = torch.tensor(
                [[-1.0, -1.0, -1.0],
                 [-1.0,  8.0, -1.0],
                 [-1.0, -1.0, -1.0]],
                dtype=torch.float32
            )
            self.kernel = kernel_2d.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)
            self.padding = 1
        else:
            # Identity / brightness kernel
            kernel_2d = torch.tensor(
                [[0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0]],
                dtype=torch.float32
            )
            self.kernel = kernel_2d.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)
            self.padding = 1

    def _process_frame_batch(self, frames_np: np.ndarray) -> np.ndarray:
        """
        Process a batch of BGR frames on the GPU.

        Args:
            frames_np: Numpy array of shape (B, H, W, 3) and dtype uint8

        Returns:
            Numpy array of shape (B, H, W, 3) and dtype uint8
        """
        # Convert to tensor and permute to NCHW: (B, 3, H, W), normalized to [0, 1]
        tensor = torch.from_numpy(frames_np).to(self.device, non_blocking=True)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0

        # Perform depthwise 2D convolution on GPU
        filtered = F.conv2d(tensor, self.kernel, padding=self.padding, groups=3)

        # Tone mapping / contrast / saturation tensor adjustment
        filtered = torch.clamp(filtered * 1.05 + 0.01, 0.0, 1.0)

        # Convert back to uint8 NHWC: (B, H, W, 3)
        out_tensor = (filtered.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).to(torch.uint8)

        # Move to host CPU memory
        return out_tensor.cpu().numpy()

    def execute(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Decode input video, execute batched PyTorch GPU tensor filtering,
        and encode to output file.

        Args:
            input_path: Absolute or relative path to the input video file
            output_path: Destination path for processed video. If None, auto-generated.
            batch_size: Number of frames per CUDA batch. Uses default_batch_size if None.
            progress_callback: Optional callable receiving (frames_processed, total_frames, elapsed_sec)

        Returns:
            Dictionary containing execution statistics and metadata
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        bs = batch_size or self.default_batch_size
        if bs <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {bs}")

        # Set default output path if not specified
        if not output_path:
            dir_name = os.path.dirname(input_path) or "."
            base_name = os.path.basename(input_path)
            stem, ext = os.path.splitext(base_name)
            output_path = os.path.join(dir_name, f"{stem}_processed{ext or '.mp4'}")

        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Reset GPU peak memory statistics if using CUDA
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()

        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0
        if total_frames <= 0:
            total_frames = 1

        # Configure VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback to alternate fourcc
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"OpenCV failed to create video writer for output: {output_path}")

        frames_processed = 0
        current_batch: List[np.ndarray] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_batch.append(frame)

                if len(current_batch) >= bs:
                    # Stack into batch array
                    batch_np = np.stack(current_batch, axis=0)
                    out_np = self._process_frame_batch(batch_np)

                    for out_frame in out_np:
                        writer.write(out_frame)

                    frames_processed += len(current_batch)
                    current_batch.clear()

                    if progress_callback:
                        progress_callback(
                            frames_processed,
                            total_frames,
                            time.perf_counter() - start_time,
                        )

            # Process any remaining frames in the final partial batch
            if current_batch:
                batch_np = np.stack(current_batch, axis=0)
                out_np = self._process_frame_batch(batch_np)
                for out_frame in out_np:
                    writer.write(out_frame)
                frames_processed += len(current_batch)
                current_batch.clear()

                if progress_callback:
                    progress_callback(
                        frames_processed,
                        total_frames,
                        time.perf_counter() - start_time,
                    )

        except torch.cuda.OutOfMemoryError as oom:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA Out of Memory during video execution at frame {frames_processed}. "
                f"Try reducing batch_size (current: {bs}). Details: {oom}"
            ) from oom
        finally:
            cap.release()
            writer.release()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.perf_counter() - start_time
        effective_fps = frames_processed / max(0.001, elapsed)

        peak_vram_bytes = 0
        if self.device.type == "cuda":
            peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)

        return {
            "status": "COMPLETED",
            "input_path": input_path,
            "output_path": output_path,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "device": str(self.device),
            "cuda_used": self.device.type == "cuda",
            "batch_size": bs,
            "filter_type": self.filter_type,
            "elapsed_seconds": round(elapsed, 4),
            "processing_fps": round(effective_fps, 2),
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_mb": round(peak_vram_bytes / (1024 * 1024), 2),
        }

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        task_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a video pipeline workload. Convenience alias for execute() with task tracking.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            task_id: Optional identifier for tracking execution tasks
            batch_size: Optional frame batch size override
            progress_callback: Optional progress notification callback

        Returns:
            Dictionary with processing metrics and status
        """
        result = self.execute(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            progress_callback=progress_callback,
            **kwargs,
        )
        if task_id is not None:
            result["task_id"] = task_id
        return result

