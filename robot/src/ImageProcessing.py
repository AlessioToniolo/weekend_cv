import CameraInterface
import cv2

class ImageProcessing:
    def __init__(self):
        self.camera = CameraInterface.CameraInterface()

    def get_contours(self):
        """
        Get the contours in the latest color frame given numpy.ndarray
        Returns:
            list: List of contours in the image
        """
        color_frame = self.camera.get_color_frame()
        
        # Convert color image to grayscale
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours