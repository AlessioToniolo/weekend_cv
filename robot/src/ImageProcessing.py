import CameraInterface
import cv2
import numpy as np

class ImageProcessing:
    def __init__(self):
        self.camera = CameraInterface.CameraInterface()
        self.camera_height = 500  # Fixed camera height in mm
        self.width = 1280
        self.height = 720
        
    def get_contours(self):
        """
        Get the contours in the latest color frame including holes
        Returns:
            tuple: (contours, hierarchy) - List of contours and their hierarchical relationship
        """
        color_frame = self.camera.get_color_frame()
        # Convert color image to grayscale
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        # Find contours including holes
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    def get_part_thickness(self, contour):
        """
        Calculate average thickness of part within given contour
        Args:
            contour: Contour of the part
        Returns:
            float: Average thickness of the part in millimeters
        """
        # Get depth frame
        depth_frame = self.camera.get_depth_frame()
        if depth_frame is None:
            return None

        # Create mask from contour
        mask = np.zeros_like(depth_frame, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), -1)

        # Get depth values within contour
        part_depths = depth_frame[mask == 255]
        
        if len(part_depths) == 0:
            return None

        # Calculate average depth of part surface
        avg_part_depth = np.mean(part_depths)
        
        # Calculate thickness by subtracting from camera height
        thickness_mm = self.camera_height - avg_part_depth
        
        return thickness_mm

    def visualize_part(self):
        """
        Open CV stream showing binary mask of the part
        """
        try:
            while True:
                # Get contours
                contours, _ = self.get_contours()
                
                # Create black background
                frame = np.zeros((self.height, self.width), dtype=np.uint8)
                
                # Draw all contours in white (including holes)
                cv2.drawContours(frame, contours, -1, (255), -1)
                
                # Show frame
                cv2.imshow('Part Visualization', frame)
                
                # Break loop with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cv2.destroyAllWindows()
            self.camera.stop()

def main():
    image_processor = ImageProcessing()
    image_processor.visualize_part()

if __name__ == "__main__":
    main()