# Camera Stream Web Application

A Flask web application that streams camera feeds from Camera 0 and Camera 1 on different API routes, similar to how they appear in OBS Studio.

## Features

- **Dual Camera Support**: Stream from Camera 0 and Camera 1 simultaneously
- **Web Interface**: Beautiful web interface to view camera streams
- **API Endpoints**: Direct API access to camera streams for integration with other applications
- **OBS Studio Compatible**: Streams can be used as sources in OBS Studio
- **Fullscreen Support**: Fullscreen viewing mode for each camera
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Cameras are Connected**:
   - Make sure Camera 0 and Camera 1 are connected to your computer
   - Verify cameras are recognized by your system

## Usage

### Starting the Application

Run the application using the Python executable:

```bash
C:\Users\tonyi\anaconda3\envs\py312\python.exe app.py
```

The server will start on `http://localhost:5000`

### Available Routes

#### Web Interface
- **Home Page**: `http://localhost:5000/` - Main page with links to both cameras
- **Camera 0**: `http://localhost:5000/camera0` - Full-screen view of Camera 0
- **Camera 1**: `http://localhost:5000/camera1` - Full-screen view of Camera 1

#### API Endpoints
- **Camera 0 Stream**: `http://localhost:5000/api/camera0/stream` - MJPEG stream for Camera 0
- **Camera 1 Stream**: `http://localhost:5000/api/camera1/stream` - MJPEG stream for Camera 1

### Using with OBS Studio

1. In OBS Studio, add a new **Browser Source**
2. Set the URL to one of the API endpoints:
   - `http://localhost:5000/api/camera0/stream` for Camera 0
   - `http://localhost:5000/api/camera1/stream` for Camera 1
3. Set the width and height as needed
4. The camera feed will appear in OBS Studio

### Using with Other Applications

The API endpoints return MJPEG streams that can be used in:
- Video conferencing applications
- Recording software
- Custom applications
- Web browsers

## Troubleshooting

### Camera Not Detected
- Ensure cameras are properly connected
- Check if cameras are being used by other applications
- Try restarting the application

### Poor Performance
- Close other applications using the cameras
- Reduce camera resolution if needed
- Check system resources

### Browser Issues
- Use a modern browser (Chrome, Firefox, Safari, Edge)
- Ensure JavaScript is enabled
- Try refreshing the page

## Technical Details

- **Framework**: Flask (Python)
- **Video Processing**: OpenCV
- **Stream Format**: MJPEG (Motion JPEG)
- **Protocol**: HTTP multipart/x-mixed-replace
- **Port**: 5000 (configurable in app.py)

## File Structure

```
stitch/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/         # HTML templates
    ├── index.html     # Main page
    └── camera.html    # Individual camera page
```

## Customization

You can modify the application by:
- Changing the port number in `app.py`
- Adjusting camera settings in the `get_camera_feed()` function
- Modifying the HTML templates for different styling
- Adding more camera support by extending the code

## License

This project is open source and available under the MIT License. 