import cv2
import numpy as np
import math
from collections import defaultdict, deque
from ultralytics import YOLO
import time
from yt_dlp import YoutubeDL  

class VehicleTracker:
    def __init__(self, max_disappeared=50, max_distance=150):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox, confidence):
        self.objects[self.next_id] = {
            'centroid': centroid, 
            'bbox': bbox, 
            'confidence': confidence,
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
            
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_bboxes = []
        input_confidences = []
        
        for (i, (startX, startY, endX, endY, conf)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
            input_bboxes.append((startX, startY, endX, endY))
            input_confidences.append(conf)
            
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_confidences[i])
        else:
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = input_bboxes[col]
                self.objects[object_id]['confidence'] = input_confidences[col]
                self.objects[object_id]['area'] = (input_bboxes[col][2] - input_bboxes[col][0]) * (input_bboxes[col][3] - input_bboxes[col][1])
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col], input_confidences[col])
                    
        return self.objects

class CollisionDetector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.tracker = VehicleTracker()
        
        # Vehicle trajectory history
        self.trajectories = defaultdict(lambda: deque(maxlen=50))
        self.speeds = defaultdict(lambda: deque(maxlen=15))
        self.accelerations = defaultdict(lambda: deque(maxlen=10))
        self.last_positions = {}
        self.last_times = {}
        self.vehicle_sizes = {}
        
        # Enhanced collision detection parameters
        self.near_collision_distance = 80  # Orange warning distance
        self.collision_distance = 30  # Red alert distance
        self.bbox_overlap_threshold = 0.2  # Minimum IoU for bbox overlap
        self.speed_drop_threshold = 0.5  # 50% speed drop
        self.trajectory_change_threshold = 45  # degrees
        self.min_speed_for_collision = 5  # minimum speed to consider collision
        self.min_frames_for_collision = 8  # minimum frames to track before collision detection
        self.acceleration_threshold = -15  # sudden deceleration threshold
        
        # Collision state tracking
        self.collision_alerts = {}
        self.near_collision_alerts = {}
        self.collision_history = []
        self.collision_cooldown = {}
        self.frame_count = 0
        
        # Vehicle classes in COCO dataset (more comprehensive)
        self.vehicle_classes = [1, 2, 3, 5, 7, 14]  # person, car, motorcycle, bus, truck, bird
        self.primary_vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_bbox_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_bbox_distance(self, box1, box2):
        """Calculate minimum distance between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # If boxes overlap, distance is 0
        if not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
            return 0.0
        
        # Calculate minimum distance between box edges
        dx = max(0, max(x1_1 - x2_2, x1_2 - x2_1))
        dy = max(0, max(y1_1 - y2_2, y1_2 - y2_1))
        
        return math.sqrt(dx*dx + dy*dy)
    
    def calculate_speed(self, prev_pos, curr_pos, time_diff):
        if time_diff == 0:
            return 0
        distance = self.calculate_distance(prev_pos, curr_pos)
        return distance / time_diff  # pixels per second
    
    def calculate_acceleration(self, speeds):
        """Calculate acceleration from speed history"""
        if len(speeds) < 2:
            return 0
        recent_speeds = list(speeds)[-2:]
        return recent_speeds[-1] - recent_speeds[0]  # simple acceleration approximation
    
    def detect_trajectory_convergence(self, traj1, traj2):
        """Enhanced trajectory convergence detection"""
        if len(traj1) < 5 or len(traj2) < 5:
            return False
        
        # Get recent trajectory points
        recent1 = list(traj1)[-5:]
        recent2 = list(traj2)[-5:]
        
        # Calculate movement vectors (more robust)
        vec1 = np.array([recent1[-1][0] - recent1[0][0], recent1[-1][1] - recent1[0][1]])
        vec2 = np.array([recent2[-1][0] - recent2[0][0], recent2[-1][1] - recent2[0][1]])
        
        # Normalize vectors
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return False
            
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Vector between current positions
        between_vec = np.array([recent2[-1][0] - recent1[-1][0], recent2[-1][1] - recent1[-1][1]])
        if np.linalg.norm(between_vec) == 0:
            return True  # Already overlapping
        
        between_norm = between_vec / np.linalg.norm(between_vec)
        
        # Check if vehicles are moving towards each other
        dot1 = np.dot(vec1_norm, between_norm)
        dot2 = np.dot(vec2_norm, -between_norm)
        
        # More lenient threshold for convergence
        return dot1 > 0.3 and dot2 > 0.3
    
    def detect_sudden_changes(self, speeds, accelerations):
        """Detect sudden speed or acceleration changes"""
        if len(speeds) < 5:
            return False, False
            
        recent_speeds = list(speeds)[-3:]
        earlier_speeds = list(speeds)[-8:-3] if len(speeds) >= 8 else list(speeds)[:-3]
        
        speed_drop = False
        sudden_decel = False
        
        if earlier_speeds:
            recent_avg = np.mean(recent_speeds)
            earlier_avg = np.mean(earlier_speeds)
            
            if earlier_avg > self.min_speed_for_collision:
                speed_ratio = recent_avg / earlier_avg if earlier_avg > 0 else 0
                speed_drop = speed_ratio < self.speed_drop_threshold
        
        # Check for sudden deceleration
        if len(accelerations) >= 3:
            recent_accel = list(accelerations)[-3:]
            sudden_decel = any(a < self.acceleration_threshold for a in recent_accel)
        
        return speed_drop, sudden_decel
    
    def is_collision_cooldown_active(self, id1, id2):
        """Check if collision pair is in cooldown period"""
        pair_key = tuple(sorted([id1, id2]))
        current_time = time.time()
        
        if pair_key in self.collision_cooldown:
            if current_time - self.collision_cooldown[pair_key] < 2.0:  # 2 second cooldown
                return True
        
        return False
    
    def set_collision_cooldown(self, id1, id2):
        """Set cooldown for collision pair"""
        pair_key = tuple(sorted([id1, id2]))
        self.collision_cooldown[pair_key] = time.time()
    
    def analyze_vehicle_proximity(self, vehicle_data):
        """Enhanced proximity analysis with multiple criteria"""
        near_collisions = []
        collisions = []
        vehicle_ids = list(vehicle_data.keys())
        
        for i, id1 in enumerate(vehicle_ids):
            for j, id2 in enumerate(vehicle_ids[i+1:], i+1):
                
                # Skip if not enough tracking history
                if (len(self.trajectories[id1]) < self.min_frames_for_collision or 
                    len(self.trajectories[id2]) < self.min_frames_for_collision):
                    continue
                
                # Skip if in cooldown
                if self.is_collision_cooldown_active(id1, id2):
                    continue
                
                obj1 = vehicle_data[id1]
                obj2 = vehicle_data[id2]
                
                bbox1 = obj1['bbox']
                bbox2 = obj2['bbox']
                centroid1 = obj1['centroid']
                centroid2 = obj2['centroid']
                
                # Calculate distances and overlaps
                centroid_distance = self.calculate_distance(centroid1, centroid2)
                bbox_distance = self.calculate_bbox_distance(bbox1, bbox2)
                bbox_iou = self.calculate_bbox_iou(bbox1, bbox2)
                
                # Get current speeds and accelerations
                speed1 = list(self.speeds[id1])[-1] if self.speeds[id1] else 0
                speed2 = list(self.speeds[id2])[-1] if self.speeds[id2] else 0
                
                # Calculate relative size factor (larger vehicles need more space)
                area1 = obj1.get('area', 1000)
                area2 = obj2.get('area', 1000)
                size_factor = max(math.sqrt(area1), math.sqrt(area2)) / 100
                
                # Adaptive thresholds based on vehicle size
                adaptive_collision_dist = self.collision_distance * (1 + size_factor * 0.5)
                adaptive_near_collision_dist = self.near_collision_distance * (1 + size_factor * 0.3)
                
                # Enhanced collision criteria
                collision_detected = False
                collision_reasons = []
                collision_confidence = 0
                
                # Critical proximity or overlap (highest priority)
                if bbox_iou > self.bbox_overlap_threshold:
                    collision_detected = True
                    collision_reasons.append("Bbox overlap")
                    collision_confidence += 3
                
                if bbox_distance < 15 * size_factor:
                    collision_detected = True
                    collision_reasons.append("Critical proximity")
                    collision_confidence += 2
                
                # Movement-based detection
                if (centroid_distance < adaptive_collision_dist and 
                    (speed1 > self.min_speed_for_collision or speed2 > self.min_speed_for_collision)):
                    collision_detected = True
                    collision_reasons.append("Close proximity with movement")
                    collision_confidence += 2
                
                # Trajectory convergence
                if (centroid_distance < adaptive_collision_dist * 1.8 and
                    self.detect_trajectory_convergence(
                        list(self.trajectories[id1]), 
                        list(self.trajectories[id2])
                    )):
                    collision_detected = True
                    collision_reasons.append("Converging paths")
                    collision_confidence += 2
                
                # Sudden behavioral changes
                speed_drop1, decel1 = self.detect_sudden_changes(self.speeds[id1], self.accelerations[id1])
                speed_drop2, decel2 = self.detect_sudden_changes(self.speeds[id2], self.accelerations[id2])
                
                if (centroid_distance < adaptive_collision_dist * 2.5 and 
                    (speed_drop1 or speed_drop2 or decel1 or decel2)):
                    collision_detected = True
                    collision_reasons.append("Sudden stop/deceleration")
                    collision_confidence += 1
                
                # Only trigger collision if confidence is high enough
                if collision_detected and collision_confidence >= 2:
                    self.set_collision_cooldown(id1, id2)
                    collision_info = {
                        'vehicle_ids': [id1, id2],
                        'positions': [centroid1, centroid2],
                        'bboxes': [bbox1, bbox2],
                        'distance': centroid_distance,
                        'bbox_distance': bbox_distance,
                        'bbox_iou': bbox_iou,
                        'reasons': collision_reasons,
                        'confidence': collision_confidence,
                        'timestamp': time.time(),
                        'speeds': [speed1, speed2]
                    }
                    collisions.append(collision_info)
                
                # Near collision criteria (more lenient)
                elif (centroid_distance < adaptive_near_collision_dist and 
                      bbox_distance < 60 * size_factor and
                      (speed1 > 2 or speed2 > 2)):
                    
                    near_collision_info = {
                        'vehicle_ids': [id1, id2],
                        'positions': [centroid1, centroid2],
                        'bboxes': [bbox1, bbox2],
                        'distance': centroid_distance,
                        'bbox_distance': bbox_distance,
                        'speeds': [speed1, speed2]
                    }
                    near_collisions.append(near_collision_info)
        
        return near_collisions, collisions
    
    def process_frame(self, frame):
        self.frame_count += 1
        current_time = time.time()
        
        # Enhanced YOLO detection with multiple scales
        results = self.model(frame, verbose=False, conf=0.4, iou=0.5)
        
        # Extract vehicle detections with enhanced filtering
        vehicle_boxes = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Primary vehicle classes with higher confidence
                    if cls in self.primary_vehicle_classes and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Filter out very small detections (likely false positives)
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        if area > 500 and width > 20 and height > 20:  # Minimum size filter
                            vehicle_boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
                    
                    # Secondary vehicle classes (lower confidence acceptable)
                    elif cls in self.vehicle_classes and conf > 0.6 and cls not in self.primary_vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        if area > 800:  # Larger minimum size for secondary classes
                            vehicle_boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
        
        # Update tracker
        tracked_objects = self.tracker.update(vehicle_boxes)
        
        # Update trajectories, speeds, and accelerations
        vehicle_data = {}
        for object_id, obj_data in tracked_objects.items():
            centroid = obj_data['centroid']
            bbox = obj_data['bbox']
            confidence = obj_data['confidence']
            area = obj_data['area']
            
            vehicle_data[object_id] = {
                'centroid': centroid, 
                'bbox': bbox, 
                'confidence': confidence,
                'area': area
            }
            
            # Update trajectory
            self.trajectories[object_id].append(centroid)
            
            # Calculate speed and acceleration
            if object_id in self.last_positions and object_id in self.last_times:
                time_diff = current_time - self.last_times[object_id]
                if time_diff > 0.01:  # Avoid division by very small numbers
                    speed = self.calculate_speed(
                        self.last_positions[object_id], 
                        centroid, 
                        time_diff
                    )
                    self.speeds[object_id].append(speed)
                    
                    # Calculate acceleration
                    if len(self.speeds[object_id]) >= 2:
                        acceleration = self.calculate_acceleration(self.speeds[object_id])
                        self.accelerations[object_id].append(acceleration)
            
            self.last_positions[object_id] = centroid
            self.last_times[object_id] = current_time
        
        # Analyze proximity and detect collisions
        near_collisions, collisions = self.analyze_vehicle_proximity(vehicle_data)
        
        # Draw results
        result_frame = self.draw_results(frame, vehicle_boxes, tracked_objects, 
                                       near_collisions, collisions)
        
        return result_frame, near_collisions, collisions
    
    def draw_results(self, frame, vehicle_boxes, tracked_objects, near_collisions, collisions):
        result_frame = frame.copy()
        
        # Get collision and near-collision vehicle IDs for coloring
        collision_ids = set()
        near_collision_ids = set()
        
        for collision in collisions:
            collision_ids.update(collision['vehicle_ids'])
        
        for near_collision in near_collisions:
            near_collision_ids.update(near_collision['vehicle_ids'])
        
        # Draw tracked vehicles with colored rectangles only
        for object_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            x1, y1, x2, y2 = bbox
            
            # Determine color and thickness based on collision status
            if object_id in collision_ids:
                color = (0, 0, 255)  # Red for collision
                thickness = 4
            elif object_id in near_collision_ids:
                color = (0, 165, 255)  # Orange for near collision
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            # Draw only the bounding box rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw simple statistics in corner
        stats_text = [
            f"Vehicles: {len(tracked_objects)}",
            f"Warnings: {len(near_collisions)}",
            f"Alerts: {len(collisions)}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(result_frame, text, (10, 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame

def main():
    # Initialize collision detector
    detector = CollisionDetector()

    # Step 1: Get YouTube Live Stream URL
    url = 'https://www.youtube.com/live/up3rJmxI1Fo?si=d2QFZ3KMcOWUkVV9'  # Replace with actual live URL

    ydl_opts = {'quiet': True, 'format': 'best[height<=720]'}  # Limit to 720p or lower
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        stream_url = info_dict['url']
    
    # Open video capture
    cap = cv2.VideoCapture(stream_url)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Video writer for saving results (optional)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    
    # Display window size settings
    display_width = 960
    display_height = 540
    
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        print(f"Stream resolution: {width}x{height}, FPS: {fps}")
        out = cv2.VideoWriter('collision_detection_output.avi', fourcc, fps, (display_width, display_height))
        
        cv2.namedWindow('Vehicle Collision Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vehicle Collision Detection', display_width, display_height)
    else:
        print("Error: Could not open video stream")
        return
    
    print("Starting collision detection system... Press 'q' to quit")
    print("🟢 Green: Normal | 🟠 Orange: Near Collision | 🔴 Red: Critical Alert")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or connection lost")
            break
        
        frame_count += 1
        
        # Resize frame for display and processing efficiency
        frame_resized = cv2.resize(frame, (display_width, display_height))
        
        # Process frame
        result_frame, near_collisions, collisions = detector.process_frame(frame_resized)
        
        # Log alerts (reduced frequency)
        if frame_count % 30 == 0:  # Log every 30 frames
            if near_collisions:
                print(f"⚠️ {len(near_collisions)} Near collision warning(s)")
            
        for collision in collisions:
            print(f"🚨 COLLISION ALERT! Vehicles: {collision['vehicle_ids']}, Confidence: {collision['confidence']}, Reasons: {', '.join(collision['reasons'][:2])}")
            detector.collision_history.append(collision)
        
        # Display result
        cv2.imshow('Vehicle Collision Detection', result_frame)
        
        # Save frame
        if out:
            out.write(result_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    if detector.collision_history:
        print(f"\n📊 Session Summary: {len(detector.collision_history)} collision alert(s) detected")

if __name__ == "__main__":
    main()