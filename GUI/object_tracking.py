import os
import logging
import argparse
import warnings
import traceback
from timeit import time
import csv
import glob

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from deep_sort import preprocessing
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from tools import generate_detections as gdet
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("Warning: DeepSORT not available. Using simple tracking.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tracking')
warnings.filterwarnings('ignore')

BASE_FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(description='Object Tracking using Ultralytics YOLO (Person & Ball).')
    parser.add_argument('--model', type=str, default='yolo11x.pt',
                        help='YOLO model path (e.g., yolo11x.pt, yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detection')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Is GPU acceleration available')
    parser.add_argument('--fps', type=int, default=0,
                        help=f'Frame per sec to process based {BASE_FPS} fps,' +
                        ' 0 to consume with no skipped frame')
    parser.add_argument('--src', type=str, default='./output/cylindricalFrames',
                        help='Video filename or directory with frames for tracking person and ball')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='Location for tracked frame images will be stored.')
    parser.add_argument('--field-corners', type=str, default=None,
                        help='Field corner coordinates as JSON string')

    args = parser.parse_args()
    return args


def transform_to_field_coordinates(x, y, perspective_matrix):
    """Transform pixel coordinates to real field coordinates using perspective transform"""
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, perspective_matrix)
    return transformed[0][0][0], transformed[0][0][1]


def create_perspective_matrix(field_corners):
    """
    Create perspective transformation matrix from field corners to real coordinates
    Field corners order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    Real field size: 9m (width) × 16m (height)
    """
    # Source points (from video/image)
    src_points = np.float32(field_corners)
    
    # Destination points (real field coordinates in meters)
    # Top-Left (0, 0), Top-Right (9, 0), Bottom-Right (9, 16), Bottom-Left (0, 16)
    dst_points = np.float32([
        [0, 0],      # Top-Left
        [9, 0],      # Top-Right
        [9, 16],     # Bottom-Right
        [0, 16]      # Bottom-Left
    ])
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix


def main(args):
    logger.info('Start Tracking...')

    # Load YOLO model - Force CPU mode to avoid DLL issues
    device = 'cpu'  # Always use CPU
    model = YOLO(args.model)
    
    fps = max(0, min(BASE_FPS, args.fps))
    
    # Setup perspective transformation if field corners are provided
    perspective_matrix = None
    if hasattr(args, 'field_corners') and args.field_corners is not None:
        perspective_matrix = create_perspective_matrix(args.field_corners)
        logger.info('Perspective transformation enabled for field coordinates')

    # Setup DeepSORT if available, otherwise use YOLO's built-in tracker
    if DEEPSORT_AVAILABLE:
        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap = 1.0

        # feature extractor for deepsort re-id
        encoder = gdet.BoxEncoder()

        metric = nn_matching.NearestNeighborDistanceMetric(
            'cosine', max_cosine_distance, nn_budget
        )
        tracker = Tracker(metric)
        use_deepsort = True
    else:
        use_deepsort = False
        logger.info("Using YOLO built-in tracking")

    # Check if source is a directory of frames or a video file
    is_frame_sequence = os.path.isdir(args.src)
    
    if is_frame_sequence:
        # Load frames from directory
        frame_files = sorted(glob.glob(os.path.join(args.src, "*.jpg")) + 
                           glob.glob(os.path.join(args.src, "*.png")))
        if not frame_files:
            logger.error(f"No image frames found in directory: {args.src}")
            return
        logger.info(f"Found {len(frame_files)} frames in {args.src}")
        total_frames = len(frame_files)
        frame_iterator = iter(frame_files)
    else:
        # Open video file
        capture = cv2.VideoCapture(args.src)
        if not capture.isOpened():
            logger.error(f"Cannot open video: {args.src}")
            return
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video loaded: {total_frames} frames")
    
    # Create CSV files for tracking results
    person_csv_path = os.path.join(args.out_dir, 'person_tracking.csv')
    ball_csv_path = os.path.join(args.out_dir, 'ball_tracking.csv')
    person_field_csv_path = os.path.join(args.out_dir, 'person_tracking_field_coords.csv')
    ball_field_csv_path = os.path.join(args.out_dir, 'ball_tracking_field_coords.csv')
    
    person_csv = open(person_csv_path, 'w', newline='')
    ball_csv = open(ball_csv_path, 'w', newline='')
    
    person_writer = csv.writer(person_csv)
    ball_writer = csv.writer(ball_csv)
    
    # Write CSV headers
    person_writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    ball_writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    
    # Create field coordinate CSV files if perspective matrix exists
    if perspective_matrix is not None:
        person_field_csv = open(person_field_csv_path, 'w', newline='')
        ball_field_csv = open(ball_field_csv_path, 'w', newline='')
        
        person_field_writer = csv.writer(person_field_csv)
        ball_field_writer = csv.writer(ball_field_csv)
        
        # Write CSV headers for field coordinates
        person_field_writer.writerow(['frame', 'track_id', 'center_x_m', 'center_y_m', 'confidence'])
        ball_field_writer.writerow(['frame', 'track_id', 'center_x_m', 'center_y_m', 'confidence'])
    
    frame_interval = BASE_FPS // fps if fps > 0 else 0
    frame_index = 0
    
    while True:
        # Read frame from video or frame sequence
        if is_frame_sequence:
            try:
                frame_path = next(frame_iterator)
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    continue
                ret = True
            except StopIteration:
                ret = False
        else:
            ret, frame = capture.read()
            
        if not ret:
            break

        if 0 < fps and frame_index % frame_interval != 0:
            frame_index += 1
            continue

        # YOLO detection - filter for person (class 0) and sports ball (class 32) in COCO
        results = model(frame, conf=args.conf, classes=[0, 32], device=device, verbose=False)
        
        if use_deepsort and DEEPSORT_AVAILABLE:
            # Extract boxes for DeepSORT
            boxs = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes_xyxy:
                    # Convert xyxy to xywh for DeepSORT
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    boxs.append([x1, y1, w, h])

            if boxs:
                features = encoder(frame, boxs)
            else:
                features = np.array([])

            # Create detections
            detections = [Detection(bbox, 1.0, feature)
                          for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression
            boxes_arr = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes_arr, nms_max_overlap, scores
            )
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            frame_index += 1

            # store original scene
            cv2.imwrite(os.path.join(args.out_dir, f'{frame_index}.jpg'), frame)

            show_img = frame.copy()
            # Draw tracked objects
            for track in tracker.tracks:
                bbox = [max(0, int(x)) for x in track.to_tlbr()]
                if not track.is_confirmed() or track.time_since_update > 1:
                    if 2 <= track.time_since_update < 10:
                        try:
                            cv2.imwrite(
                                os.path.join(
                                    args.out_dir,
                                    f'missed-{frame_index}-{track.track_id}.jpg'
                                ),
                                frame,
                            )
                        except:
                            traceback.print_exc()
                    logger.info('Skipped by time_since_update')
                    continue

                logger.info(f'Frame #{frame_index} - Id: {track.track_id}')
                cv2.rectangle(show_img,
                              (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (255,255,255), 2)
                cv2.putText(show_img,
                            str(track.track_id),
                            (bbox[0], bbox[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2)

            # show image
            cv2.imwrite(os.path.join(args.out_dir, f'anno-{frame_index}.jpg'), show_img)
            cv2.imshow('Tracking', show_img)
            
        else:
            # Use YOLO's built-in tracking
            results = model.track(frame, persist=True, conf=args.conf, classes=[0, 32], device=device, verbose=False)
            
            frame_index += 1
            
            # store original scene
            cv2.imwrite(os.path.join(args.out_dir, f'{frame_index}.jpg'), frame)
            
            # Draw results
            show_img = frame.copy()
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else i
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = 'Person' if cls == 0 else 'Ball'
                    
                    logger.info(f'Frame #{frame_index} - Id: {track_id}, Class: {cls_name}, Conf: {conf:.2f}')
                    
                    # Calculate center point of bounding box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Save tracking data to CSV (pixel coordinates)
                    if cls == 0:  # Person
                        person_writer.writerow([frame_index, track_id, int(x1), int(y1), int(x2), int(y2), f'{conf:.4f}'])
                    else:  # Ball (cls == 32)
                        ball_writer.writerow([frame_index, track_id, int(x1), int(y1), int(x2), int(y2), f'{conf:.4f}'])
                    
                    # Save field coordinates if perspective matrix exists
                    if perspective_matrix is not None:
                        field_x, field_y = transform_to_field_coordinates(center_x, center_y, perspective_matrix)
                        if cls == 0:  # Person
                            person_field_writer.writerow([frame_index, track_id, f'{field_x:.4f}', f'{field_y:.4f}', f'{conf:.4f}'])
                        else:  # Ball
                            ball_field_writer.writerow([frame_index, track_id, f'{field_x:.4f}', f'{field_y:.4f}', f'{conf:.4f}'])
                    
                    # Different colors for different classes
                    color = (255, 255, 255) if cls == 0 else (0, 165, 255)  # White for person, Orange for ball
                    
                    cv2.rectangle(show_img,
                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                  color, 2)
                    cv2.putText(show_img,
                                f'{cls_name} ID:{track_id}',
                                (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2)
            
            # show image
            cv2.imwrite(os.path.join(args.out_dir, f'anno-{frame_index}.jpg'), show_img)
            cv2.imshow('Tracking', show_img)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if use_deepsort and DEEPSORT_AVAILABLE:
        logger.info(f'Missed obj: {tracker.missed_obj}, Missed frame: {tracker.missed_frame}')
    
    logger.info(f'Tracking complete. Total frames: {frame_index}')
    logger.info(f'Results saved to:')
    logger.info(f'  - Person tracking (pixels): {person_csv_path}')
    logger.info(f'  - Ball tracking (pixels): {ball_csv_path}')
    if perspective_matrix is not None:
        logger.info(f'  - Person tracking (field coords): {person_field_csv_path}')
        logger.info(f'  - Ball tracking (field coords): {ball_field_csv_path}')

    # Close CSV files
    person_csv.close()
    ball_csv.close()
    if perspective_matrix is not None:
        person_field_csv.close()
        ball_field_csv.close()
    
    if not is_frame_sequence:
        capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    main(args)
