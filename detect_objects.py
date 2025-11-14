"""
Object Detection Module
Supports Traditional CV (contour-based) and CRAFT (text detection)
"""

import cv2
import numpy as np
from typing import List, Tuple

# ============================================================================
# Traditional CV Detector
# ============================================================================

class TraditionalDetector:
    """
    Contour-based object detector for clean backgrounds.
    Best for: whiteboard, paper, simple digital images.
    """
    
    def __init__(self, min_area=200, max_area=30000,
                 aspect_ratio_range=(0.2, 5.0)):
        """
        Args:
            min_area: Minimum object area in pixels²
            max_area: Maximum object area in pixels²
            aspect_ratio_range: (min, max) aspect ratio to keep
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects using contour detection.
        
        Args:
            image: numpy array (H, W) or (H, W, 3)
            
        Returns:
            bboxes: list of (x, y, w, h)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and extract bounding boxes
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = w / float(h) if h > 0 else 0
                if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                    bboxes.append((x, y, w, h))
        
        return bboxes
    
    def visualize(self, image: np.ndarray, bboxes: List[Tuple],
                  labels=None, confidences=None,
                  color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on image.
        
        Args:
            image: numpy array
            bboxes: list of (x, y, w, h)
            labels: list of label strings (optional)
            confidences: list of confidence scores (optional)
            color: BGR color tuple
            thickness: line thickness
            
        Returns:
            result: image with drawn boxes
        """
        result = image.copy()
        
        # Convert to BGR if grayscale
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for i, (x, y, w, h) in enumerate(bboxes):
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
            
            # Draw label if provided
            if labels is not None:
                label_text = str(labels[i])
                if confidences is not None:
                    label_text += f" ({confidences[i]:.2f})"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(result, (x, y-text_h-10), (x+text_w, y), color, -1)
                
                # Text
                cv2.putText(result, label_text, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result

# ============================================================================
# CRAFT Detector (Advanced)
# ============================================================================

class CRAFTDetector:
    """
    CRAFT-based text/character detector.
    Requires: CRAFT model weights (craft_mlt_25k.pth)
    
    Better for: complex backgrounds, rotated text, scene text
    """
    
    def __init__(self, model_path='craft_mlt_25k.pth',
                 text_threshold=0.7, link_threshold=0.4, low_text=0.4):
        """
        Args:
            model_path: Path to CRAFT pretrained weights
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold
            low_text: Low text threshold
        """
        try:
            import sys
            sys.path.append('./craft_repo')
            from craft_repo import craft
            from craft_repo import craft_utils
            from craft_repo import imgproc
            
            self.craft_net = craft.CRAFT()
            self.craft_net.load_state_dict(torch.load(model_path))
            self.craft_net.eval()
            
            self.craft_utils = craft_utils
            self.imgproc = imgproc
            
            self.text_threshold = text_threshold
            self.link_threshold = link_threshold
            self.low_text = low_text
            
            self.available = True
            print("✅ CRAFT detector initialized")
            
        except Exception as e:
            self.available = False
            print(f"⚠️  CRAFT not available: {e}")
            print("   Falling back to Traditional CV detector")
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using CRAFT.
        
        Args:
            image: numpy array (H, W, 3) BGR
            
        Returns:
            bboxes: list of (x, y, w, h)
        """
        if not self.available:
            raise RuntimeError("CRAFT not available")
        
        # Preprocess image
        img_resized, target_ratio, size_heatmap = self.imgproc.resize_aspect_ratio(
            image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        # Convert to tensor
        x = self.imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            y, feature = self.craft_net(x)
        
        # Post-process
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        
        boxes, polys = self.craft_utils.getDetBoxes(
            score_text, score_link,
            self.text_threshold, self.link_threshold, self.low_text
        )
        
        # Adjust coordinates
        boxes = self.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        # Convert to (x, y, w, h) format
        bboxes = []
        for box in boxes:
            x_min = int(min(box[:, 0]))
            y_min = int(min(box[:, 1]))
            x_max = int(max(box[:, 0]))
            y_max = int(max(box[:, 1]))
            
            w = x_max - x_min
            h = y_max - y_min
            
            bboxes.append((x_min, y_min, w, h))
        
        return bboxes

# ============================================================================
# Hybrid Detector
# ============================================================================

class HybridDetector:
    """
    Combines CRAFT (for text/digits) and Traditional CV (for shapes).
    Best of both worlds!
    """
    
    def __init__(self, craft_path=None):
        """
        Args:
            craft_path: Path to CRAFT weights (optional)
        """
        self.cv_detector = TraditionalDetector()
        
        if craft_path and os.path.exists(craft_path):
            try:
                self.craft_detector = CRAFTDetector(craft_path)
                self.craft_available = self.craft_detector.available
            except:
                self.craft_available = False
        else:
            self.craft_available = False
        
        if self.craft_available:
            print("✅ Hybrid detector: CRAFT + Traditional CV")
        else:
            print("✅ Hybrid detector: Traditional CV only")
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect using hybrid approach.
        
        Strategy:
        1. Use CRAFT for text/digits (if available)
        2. Use Traditional CV for shapes
        3. Merge and deduplicate bboxes
        """
        if self.craft_available:
            # CRAFT for text
            text_bboxes = self.craft_detector.detect(image)
            
            # Traditional CV for shapes (mask out text regions first)
            masked_image = self._mask_regions(image, text_bboxes)
            shape_bboxes = self.cv_detector.detect(masked_image)
            
            # Merge
            all_bboxes = self._merge_bboxes(text_bboxes, shape_bboxes)
        else:
            # Fall back to Traditional CV only
            all_bboxes = self.cv_detector.detect(image)
        
        return all_bboxes
    
    def _mask_regions(self, image, bboxes):
        """Mask out specified regions."""
        masked = image.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(masked, (x, y), (x+w, y+h), (255, 255, 255), -1)
        return masked
    
    def _merge_bboxes(self, bboxes1, bboxes2, iou_threshold=0.5):
        """Merge two bbox lists and remove duplicates."""
        all_bboxes = list(bboxes1) + list(bboxes2)
        
        # Simple NMS (Non-Maximum Suppression)
        # TODO: Implement proper NMS if needed
        
        return all_bboxes

# ============================================================================
# Utility Functions
# ============================================================================

def non_max_suppression(bboxes: List[Tuple], iou_threshold=0.5) -> List[Tuple]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        bboxes: list of (x, y, w, h)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        filtered_bboxes: list of (x, y, w, h)
    """
    if len(bboxes) == 0:
        return []
    
    # Convert to (x1, y1, x2, y2) format
    boxes = np.array([
        [x, y, x+w, y+h] for (x, y, w, h) in bboxes
    ])
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by bottom-right y coordinate
    idxs = np.argsort(y2)
    
    keep = []
    while len(idxs) > 0:
        # Pick last index
        last = len(idxs) - 1
        i = idxs[last]
        keep.append(i)
        
        # Find overlap
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / areas[idxs[:last]]
        
        # Delete indexes with high overlap
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > iou_threshold)[0])
        ))
    
    # Convert back to (x, y, w, h)
    filtered_bboxes = [
        (int(boxes[i, 0]), int(boxes[i, 1]),
         int(boxes[i, 2] - boxes[i, 0]), int(boxes[i, 3] - boxes[i, 1]))
        for i in keep
    ]
    
    return filtered_bboxes

# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect_objects.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)
    
    # Detect
    detector = TraditionalDetector()
    bboxes = detector.detect(image)
    
    print(f"Detected {len(bboxes)} objects")
    
    # Visualize
    result = detector.visualize(image, bboxes)
    
    # Save
    output_path = 'detection_result.png'
    cv2.imwrite(output_path, result)
    print(f"Saved result to {output_path}")

