import cv2
import numpy as np
import math
from collections import defaultdict, deque
from ultralytics import YOLO
import time
from yt_dlp import YoutubeDL  

class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        self.objects[self.next_id] = centroid
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
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
            
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
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
                self.objects[object_id] = input_centroids[col]
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
                    self.register(input_centroids[col])
                    
        return self.objects

class CollisionDetector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')  # You can use yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt for better accuracy
        self.tracker = VehicleTracker()
        
        # Vehicle trajectory history
        self.trajectories = defaultdict(lambda: deque(maxlen=30))
        self.speeds = defaultdict(lambda: deque(maxlen=10))
        self.last_positions = {}
        self.last_times = {}
        
        # Collision detection parameters
        self.collision_threshold_distance = 50  # pixels
        self.speed_drop_threshold = 0.7  # 70% speed drop
        self.trajectory_change_threshold = 45  # degrees
        self.min_speed_for_collision = 5  # minimum speed to consider collision
        
        # Collision tracking
        self.collision_alerts = {}
        self.collision_history = []
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_speed(self, prev_pos, curr_pos, time_diff):
        if time_diff == 0:
            return 0
        distance = self.calculate_distance(prev_pos, curr_pos)
        return distance / time_diff  # pixels per second
    
    def calculate_trajectory_angle(self, trajectory):
        if len(trajectory) < 2:
            return None
        
        # Calculate angle between last two points
        p1 = trajectory[-2]
        p2 = trajectory[-1]
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    
    def detect_trajectory_change(self, trajectory):
        if len(trajectory) < 5:
            return False
            
        # Compare recent trajectory with earlier trajectory
        recent_angles = []
        early_angles = []
        
        # Calculate angles for recent points
        for i in range(len(trajectory) - 3, len(trajectory) - 1):
            if i > 0:
                angle = math.degrees(math.atan2(
                    trajectory[i][1] - trajectory[i-1][1],
                    trajectory[i][0] - trajectory[i-1][0]
                ))
                recent_angles.append(angle)
        
        # Calculate angles for earlier points
        for i in range(2, min(5, len(trajectory) - 3)):
            if i > 0:
                angle = math.degrees(math.atan2(
                    trajectory[i][1] - trajectory[i-1][1],
                    trajectory[i][0] - trajectory[i-1][0]
                ))
                early_angles.append(angle)
        
        if not recent_angles or not early_angles:
            return False
            
        avg_recent = np.mean(recent_angles)
        avg_early = np.mean(early_angles)
        
        angle_diff = abs(avg_recent - avg_early)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        return angle_diff > self.trajectory_change_threshold
    
    def check_trajectory_intersection(self, traj1, traj2):
        if len(traj1) < 2 or len(traj2) < 2:
            return False
            
        # Check if recent trajectories intersect
        for i in range(max(0, len(traj1) - 5), len(traj1) - 1):
            for j in range(max(0, len(traj2) - 5), len(traj2) - 1):
                if self.line_intersection(traj1[i], traj1[i+1], traj2[j], traj2[j+1]):
                    return True
        return False
    
    def line_intersection(self, p1, p2, p3, p4):
        # Check if two line segments intersect
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def detect_sudden_stop(self, speeds):
        if len(speeds) < 3:
            return False
            
        recent_speed = np.mean(list(speeds)[-2:])
        earlier_speed = np.mean(list(speeds)[-5:-2]) if len(speeds) >= 5 else np.mean(list(speeds)[:-2])
        
        if earlier_speed < self.min_speed_for_collision:
            return False
            
        speed_drop_ratio = recent_speed / earlier_speed if earlier_speed > 0 else 0
        return speed_drop_ratio < self.speed_drop_threshold
    
    def detect_collision(self, vehicle_data):
        collisions = []
        vehicle_ids = list(vehicle_data.keys())
        
        for i, id1 in enumerate(vehicle_ids):
            for j, id2 in enumerate(vehicle_ids[i+1:], i+1):
                collision_detected = False
                collision_reasons = []
                
                pos1 = vehicle_data[id1]['position']
                pos2 = vehicle_data[id2]['position']
                
                # Check proximity
                distance = self.calculate_distance(pos1, pos2)
                if distance > self.collision_threshold_distance:
                    continue
                
                # Check trajectory intersection
                if self.check_trajectory_intersection(
                    list(self.trajectories[id1]), 
                    list(self.trajectories[id2])
                ):
                    collision_detected = True
                    collision_reasons.append("Trajectory intersection")
                
                # Check sudden trajectory changes
                if (self.detect_trajectory_change(list(self.trajectories[id1])) or 
                    self.detect_trajectory_change(list(self.trajectories[id2]))):
                    collision_detected = True
                    collision_reasons.append("Sudden trajectory change")
                
                # Check sudden stops
                if (self.detect_sudden_stop(self.speeds[id1]) or 
                    self.detect_sudden_stop(self.speeds[id2])):
                    collision_detected = True
                    collision_reasons.append("Sudden speed drop")
                
                if collision_detected:
                    collision_info = {
                        'vehicle_ids': [id1, id2],
                        'positions': [pos1, pos2],
                        'distance': distance,
                        'reasons': collision_reasons,
                        'timestamp': time.time(),
                        'speeds': [
                            list(self.speeds[id1])[-1] if self.speeds[id1] else 0,
                            list(self.speeds[id2])[-1] if self.speeds[id2] else 0
                        ]
                    }
                    collisions.append(collision_info)
        
        return collisions
    
    def process_frame(self, frame):
        current_time = time.time()
        
        # YOLO detection
        results = self.model(frame, verbose=False)   

        # Extract vehicle detections
        vehicle_boxes = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:  # Vehicle classes
                        conf = float(box.conf[0])
                        if conf > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicle_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        # Update tracker
        tracked_objects = self.tracker.update(vehicle_boxes)
        
        # Update trajectories and speeds
        vehicle_data = {}
        for object_id, centroid in tracked_objects.items():
            vehicle_data[object_id] = {'position': centroid}
            
            # Update trajectory
            self.trajectories[object_id].append(centroid)
            
            # Calculate speed
            if object_id in self.last_positions and object_id in self.last_times:
                time_diff = current_time - self.last_times[object_id]
                if time_diff > 0:
                    speed = self.calculate_speed(
                        self.last_positions[object_id], 
                        centroid, 
                        time_diff
                    )
                    self.speeds[object_id].append(speed)
            
            self.last_positions[object_id] = centroid
            self.last_times[object_id] = current_time
        
        # Detect collisions
        collisions = self.detect_collision(vehicle_data)
        
        # Draw results
        result_frame = self.draw_results(frame, vehicle_boxes, tracked_objects, collisions)
        
        return result_frame, collisions
    
    def draw_results(self, frame, vehicle_boxes, tracked_objects, collisions):
        result_frame = frame.copy()
        
        # Draw vehicle detections
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw tracked vehicles with trajectories
        for object_id, centroid in tracked_objects.items():
            # Draw centroid
            cv2.circle(result_frame, tuple(centroid), 5, (255, 0, 0), -1)
            
            # Draw ID
            cv2.putText(result_frame, f"ID: {object_id}", 
                       (centroid[0] - 20, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw trajectory
            trajectory = list(self.trajectories[object_id])
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(result_frame, tuple(trajectory[i-1]), 
                            tuple(trajectory[i]), (0, 0, 255), 2)
            
            # Draw speed
            if self.speeds[object_id]:
                speed = self.speeds[object_id][-1]
                cv2.putText(result_frame, f"Speed: {speed:.1f}", 
                           (centroid[0] - 20, centroid[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw collision alerts
        for collision in collisions:
            for pos in collision['positions']:
                cv2.circle(result_frame, tuple(pos), 30, (0, 0, 255), 3)
                cv2.putText(result_frame, "COLLISION!", 
                           (pos[0] - 40, pos[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw collision info
            info_text = f"Collision: {', '.join(collision['reasons'])}"
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw statistics
        stats_text = [
            f"Vehicles tracked: {len(tracked_objects)}",
            f"Collisions detected: {len(collisions)}",
            f"Total alerts: {len(self.collision_history)}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(result_frame, text, (10, result_frame.shape[0] - 60 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
    display_width = 960  # Reduced window size
    display_height = 540
    
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # Default to 25 if fps is 0
        print(f"Stream resolution: {width}x{height}, FPS: {fps}")
        out = cv2.VideoWriter('collision_detection_output.avi', fourcc, fps, (display_width, display_height))
        
        # Create named window with specific size
        cv2.namedWindow('Vehicle Collision Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vehicle Collision Detection', display_width, display_height)
    else:
        print("Error: Could not open video stream")
        return
    
    print("Starting collision detection system... Press 'q' to quit")
    
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
        result_frame, collisions = detector.process_frame(frame_resized)
        
        # Only log collisions (reduced terminal output)
        for collision in collisions:
            print(f"🚨 COLLISION DETECTED! Vehicles: {collision['vehicle_ids']}, Distance: {collision['distance']:.1f}px, Reasons: {', '.join(collision['reasons'])}")
            detector.collision_history.append(collision)
        
        # Display result
        cv2.imshow('Vehicle Collision Detection', result_frame)
        
        # Save frame
        if out:
            out.write(result_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Skip frames to reduce processing load (process every 2nd frame)
        if frame_count % 2 == 0:
            continue
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    if detector.collision_history:
        print(f"\n📊 Session Summary: {len(detector.collision_history)} total collision(s) detected")

if __name__ == "__main__":
    main()