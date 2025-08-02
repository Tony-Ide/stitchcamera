from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import base64

app = Flask(__name__)

# Global variables to store camera captures
camera_0 = None
camera_1 = None

# Stitching variables
stitch_homography = None
stitch_locked = False
stitch_enabled = False

def get_camera_feed(camera_index):
    """Get camera feed for the specified camera index"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    # Set camera properties for 60 FPS
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
    
    # Try to get the actual FPS to verify
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera {camera_index} configured for {actual_fps} FPS")
    
    return cap

def generate_frames(camera_index):
    """Generate frames from camera feed"""
    global camera_0, camera_1
    
    # Get the appropriate camera
    if camera_index == 0:
        if camera_0 is None:
            camera_0 = get_camera_feed(0)
        cap = camera_0
    elif camera_index == 1:
        if camera_1 is None:
            camera_1 = get_camera_feed(1)
        cap = camera_1
    else:
        return
    
    if cap is None:
        return
    
    # Optimize JPEG encoding for speed
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Good quality, fast encoding
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"Error reading from camera {camera_index}")
            break
        
        # Encode the frame as JPEG with optimized settings
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue
        
        # Yield the frame in bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def compute_homography(img1, img2):
    """Compute homography between two images using ORB features"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(500)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return None
    
    # Create BF matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        return None
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    return H

def stitch_images(img1, img2, homography):
    """Stitch two images using homography"""
    if homography is None:
        return np.hstack([img1, img2])  # Side by side if no homography
    
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    
    # Warp the second image
    warped_img2 = cv2.warpPerspective(img2, homography, (width1 + width2, height1))
    
    # Create panorama
    panorama = warped_img2.copy()
    panorama[0:height1, 0:width1] = img1
    
    return panorama

def generate_stitched_frames():
    """Generate stitched video frames"""
    global stitch_homography, stitch_locked, stitch_enabled
    
    while True:
        if not stitch_enabled or camera_0 is None or camera_1 is None:
            time.sleep(0.1)
            continue
        
        # Read frames from both cameras
        ret1, frame1 = camera_0.read()
        ret2, frame2 = camera_1.read()
        
        if not ret1 or not ret2:
            continue
        
        # Resize frames for consistent processing
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        
        # Compute homography once if not locked
        if not stitch_locked and stitch_homography is None:
            print("🔍 Computing homography for static cameras...")
            stitch_homography = compute_homography(frame1, frame2)
            if stitch_homography is not None:
                stitch_locked = True
                print("🔒 Homography locked for static cameras!")
            else:
                print("❌ Failed to compute homography")
        
        # Stitch images
        if stitch_homography is not None:
            panorama = stitch_images(frame1, frame2, stitch_homography)
        else:
            panorama = np.hstack([frame1, frame2])
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', panorama, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # time.sleep(0.016)  # ~60 FPS

@app.route('/')
def index():
    """Main page with links to camera streams"""
    return render_template('index.html')

@app.route('/camera0')
def camera0_page():
    """Page for camera 0 stream"""
    return render_template('camera.html', camera_id=0, camera_name="Camera 0")

@app.route('/camera1')
def camera1_page():
    """Page for camera 1 stream"""
    return render_template('camera.html', camera_id=1, camera_name="Camera 1")

@app.route('/video_feed0')
def video_feed0():
    """Video streaming route for camera 0"""
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    """Video streaming route for camera 1"""
    return Response(generate_frames(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera0/stream')
def api_camera0_stream():
    """API endpoint for camera 0 stream"""
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera1/stream')
def api_camera1_stream():
    """API endpoint for camera 1 stream"""
    return Response(generate_frames(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stitching')
def stitching_page():
    """Page for camera stitching with Python API access"""
    return render_template('stitching.html')

@app.route('/api/stitched/stream')
def api_stitched_stream():
    """API endpoint for stitched camera stream"""
    return Response(generate_stitched_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stitch/start', methods=['POST'])
def start_stitching():
    """Start camera stitching"""
    global stitch_enabled, stitch_locked, stitch_homography
    
    if camera_0 is None or camera_1 is None:
        return jsonify({'error': 'Both cameras must be available'}), 400
    
    stitch_enabled = True
    stitch_locked = False
    stitch_homography = None
    
    print("🎬 Starting camera stitching...")
    return jsonify({'status': 'started', 'message': 'Stitching started'})

@app.route('/api/stitch/stop', methods=['POST'])
def stop_stitching():
    """Stop camera stitching"""
    global stitch_enabled
    
    stitch_enabled = False
    print("⏹️ Stopping camera stitching...")
    return jsonify({'status': 'stopped', 'message': 'Stitching stopped'})

@app.route('/api/stitch/reset', methods=['POST'])
def reset_stitching():
    """Reset homography and recompute"""
    global stitch_locked, stitch_homography
    
    stitch_locked = False
    stitch_homography = None
    print("🔄 Resetting homography for recomputation...")
    return jsonify({'status': 'reset', 'message': 'Homography reset'})

@app.route('/api/stitch/status')
def get_stitch_status():
    """Get current stitching status"""
    return jsonify({
        'enabled': stitch_enabled,
        'locked': stitch_locked,
        'homography_computed': stitch_homography is not None,
        'camera_0_available': camera_0 is not None,
        'camera_1_available': camera_1 is not None
    })

@app.route('/api/frame/stitched')
def get_stitched_frame():
    """Get a single stitched frame as base64 for Python code"""
    global stitch_homography, stitch_locked
    
    if not stitch_enabled or camera_0 is None or camera_1 is None:
        return jsonify({'error': 'Stitching not available'}), 400
    
    # Read frames from both cameras
    ret1, frame1 = camera_0.read()
    ret2, frame2 = camera_1.read()
    
    if not ret1 or not ret2:
        return jsonify({'error': 'Failed to read camera frames'}), 400
    
    # Resize frames for consistent processing
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))
    
    # Compute homography once if not locked
    if not stitch_locked and stitch_homography is None:
        print("🔍 Computing homography for static cameras...")
        stitch_homography = compute_homography(frame1, frame2)
        if stitch_homography is not None:
            stitch_locked = True
            print("🔒 Homography locked for static cameras!")
    
    # Stitch images
    if stitch_homography is not None:
        panorama = stitch_images(frame1, frame2, stitch_homography)
    else:
        panorama = np.hstack([frame1, frame2])
    
    # Encode to base64
    ret, buffer = cv2.imencode('.jpg', panorama, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ret:
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'frame': f"data:image/jpeg;base64,{frame_base64}",
            'stitched': stitch_homography is not None,
            'homography_locked': stitch_locked,
            'timestamp': time.time()
        })
    else:
        return jsonify({'error': 'Failed to encode frame'}), 500

if __name__ == '__main__':
    # Initialize cameras
    camera_0 = get_camera_feed(0)
    camera_1 = get_camera_feed(1)
    
    if camera_0 is None:
        print("Warning: Camera 0 not available")
    if camera_1 is None:
        print("Warning: Camera 1 not available")
    
    print("Starting Flask server...")
    print("Available routes:")
    print("- /camera0 - Camera 0 stream page")
    print("- /camera1 - Camera 1 stream page")
    print("- /stitching - Camera stitching page with Python API")
    print("- /api/camera0/stream - Camera 0 API endpoint")
    print("- /api/camera1/stream - Camera 1 API endpoint")
    print("- /api/stitched/stream - Stitched camera stream")
    print("- /api/frame/stitched - Single stitched frame (for Python code)")
    print("\nOptimized for 60 FPS streaming with minimal latency")
    
    # Run with threaded=True for better performance with multiple camera streams
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True) 