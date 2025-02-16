import pyrealsense2 as rs
import numpy as np

class CameraInterface:
    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start the pipeline
        self.pipeline.start(self.config)
    
    def get_color_frame(self):
        """
        Get the latest color frame from the camera
        Returns:
            numpy.ndarray: Color image as a numpy array, or None if frame is not available
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
            
        return np.asanyarray(color_frame.get_data())
    
    def get_depth_frame(self):
        """
        Get the latest depth frame from the camera
        Returns:
            numpy.ndarray: Depth image as a numpy array, or None if frame is not available
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            return None
            
        return np.asanyarray(depth_frame.get_data())
    
    def stop(self):
        """
        Stop the camera pipeline
        """
        self.pipeline.stop()

if __name__ == "__main__":
    import cv2

    # Initialize camera
    camera = CameraInterface()

    try:
        while True:
            # Get color and depth frames
            color_frame = camera.get_color_frame()
            depth_frame = camera.get_depth_frame()

            if color_frame is not None and depth_frame is not None:
                # Normalize depth frame for visualization (convert to 8-bit grayscale)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                    cv2.COLORMAP_JET
                )

                # Show frames
                cv2.imshow('Color Frame', color_frame)
                cv2.imshow('Depth Frame', depth_colormap)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()