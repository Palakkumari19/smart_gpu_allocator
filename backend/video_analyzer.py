import cv2
import os
import math
from typing import Dict, Optional, Tuple


class VideoAnalyzer:
    """Extract metadata from video files using OpenCV"""
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    @staticmethod
    def get_video_info(file_path: str) -> Dict:
        """
        Extract comprehensive metadata from a video file
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in VideoAnalyzer.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format: {file_ext}. Supported formats: {VideoAnalyzer.SUPPORTED_FORMATS}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 500:  # 500MB limit
            raise ValueError(f"Video file too large: {file_size_mb:.1f}MB. Maximum allowed: 500MB")
        
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file. File may be corrupted.")
        
        try:
            # Extract basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = 0
            
            # Get resolution string
            resolution = VideoAnalyzer._get_resolution_string(width, height)
            
            # Get codec info
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = VideoAnalyzer._decode_fourcc(fourcc)
            
            # Estimate bitrate (rough calculation)
            if duration > 0:
                bitrate_bps = (file_size_mb * 8 * 1024 * 1024) / duration
                bitrate_mbps = bitrate_bps / (1024 * 1024)
                bitrate_str = f"{bitrate_mbps:.1f} Mbps"
            else:
                bitrate_str = "Unknown"
            
            filename = os.path.basename(file_path)
            
            return {
                "filename": filename,
                "file_size_mb": round(file_size_mb, 2),
                "width": width,
                "height": height,
                "resolution": resolution,
                "fps": round(fps, 2) if fps > 0 else 0,
                "duration": round(duration, 2),
                "frames": frame_count,
                "codec": codec,
                "bitrate": bitrate_str,
                "file_path": file_path
            }
            
        finally:
            cap.release()
    
    @staticmethod
    def _get_resolution_string(width: int, height: int) -> str:
        """Convert width/height to standard resolution string"""
        common_resolutions = {
            (3840, 2160): "4K",
            (2560, 1440): "1440p",
            (1920, 1080): "1080p",
            (1280, 720): "720p",
            (854, 480): "480p",
            (640, 360): "360p"
        }
        
        # Check exact match
        if (width, height) in common_resolutions:
            return common_resolutions[(width, height)]
        
        # Check close matches (within 5%)
        for (std_w, std_h), name in common_resolutions.items():
            if abs(width - std_w) / std_w < 0.05 and abs(height - std_h) / std_h < 0.05:
                return name
        
        # Return custom resolution
        return f"{width}x{height}"
    
    @staticmethod
    def _decode_fourcc(fourcc: int) -> str:
        """Decode fourcc code to codec name"""
        try:
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "Unknown"
    
    @staticmethod
    def classify_workload(video_info: Dict) -> str:
        """
        Classify video workload based on actual metadata
        
        Classification rules:
        - LIGHT: 720p or lower, low FPS, short duration
        - MEDIUM: 1080p, moderate FPS, moderate duration
        - HEAVY: 1440p, high FPS, long duration
        - EXTREME: 4K or higher, 60 FPS, long duration
        """
        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]
        duration = video_info["duration"]
        
        # Calculate workload score
        total_pixels = width * height * fps * duration
        
        # Classification thresholds (based on total pixels processed)
        # These thresholds are calibrated for the 8GB GPU capacity
        
        if width >= 3840 or height >= 2160:  # 4K or higher
            if fps >= 60 or duration >= 60:
                return "EXTREME"
            return "HEAVY"
        elif width >= 2560 or height >= 1440:  # 1440p
            if fps >= 60 or duration >= 60:
                return "EXTREME"
            return "HEAVY"
        elif width >= 1920 or height >= 1080:  # 1080p
            if fps >= 60 and duration >= 60:
                return "HEAVY"
            elif fps >= 30 or duration >= 30:
                return "MEDIUM"
            return "LIGHT"
        elif width >= 1280 or height >= 720:  # 720p
            if fps >= 60 and duration >= 60:
                return "MEDIUM"
            return "LIGHT"
        else:  # Lower resolutions
            return "LIGHT"
    
    @staticmethod
    def estimate_resources(video_info: Dict) -> Tuple[float, int]:
        """
        Estimate VRAM requirement and execution time based on video properties
        
        Returns:
            (memory_required_gb, execution_time_steps)
        """
        width = video_info["width"]
        height = video_info["fps"]
        fps = video_info["fps"]
        duration = video_info["duration"]
        frames = video_info["frames"]
        
        # Calculate total pixels to process
        pixels_per_frame = width * height
        total_pixels = pixels_per_frame * frames
        
        # Maximum reference: 4K @ 60fps @ 180s
        max_pixels = 3840 * 2160 * 60 * 180
        
        # Memory estimation (VRAM scales with resolution, less with duration)
        # 4K requires ~7GB, 720p requires ~1GB
        memory_factor = pixels_per_frame / (3840 * 2160)
        memory_required = max(1.0, min(7.5, memory_factor * 7.5))
        memory_required = round(memory_required, 1)
        
        # Execution time estimation (scales with total pixels)
        # Normalize to simulation steps (1-20 steps)
        time_factor = total_pixels / max_pixels
        execution_time = max(1, min(20, int(time_factor * 20)))
        
        return memory_required, execution_time
