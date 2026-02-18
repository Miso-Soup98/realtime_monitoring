# Testing Guide - Screen Real-time Monitoring

## How to Test `run_windows_screen.bat`

### Method 1: Test with Video Player (Recommended)

1. **Prepare the video:**
   - Open the video file `da3fd50e6979f4b2cc582dfe0695445b.mp4` in a video player (Windows Media Player, VLC, etc.)
   - **Don't play it yet** - just have it ready

2. **Run the batch file:**
   - Double-click `run_windows_screen.bat`
   - Wait for Python environment setup (first time only)

3. **Select ROI region:**
   - A full-screen window will appear showing your desktop
   - Drag the red box to select the area where the waveform will be displayed
   - Click the bottom-right green checkmark (or press Enter) to confirm
   - Click the bottom-left red X (or press Esc) to cancel

4. **Start monitoring:**
   - After confirming ROI, the monitoring window will appear
   - **Now play the video** in fullscreen or windowed mode
   - Make sure the waveform area is visible within the ROI you selected

5. **Observe the results:**
   - The preview window shows a green rectangle around the ROI
   - The top-left shows `output=0` or `output=1`
   - The command window shows real-time detection results
   - Press `q` to quit

### Method 2: Test with Actual Oscilloscope Software

1. **Open your oscilloscope software** (the actual application you want to monitor)

2. **Run `run_windows_screen.bat`**

3. **Select ROI** to cover the waveform display area in your oscilloscope software

4. **Start monitoring** - the program will detect waveforms in real-time

### What to Look For

- **output=1**: Wide/thick waveform detected
- **output=0**: Narrow/bar waveform detected
- The detection updates in real-time as the waveform changes

### Troubleshooting

- **If ROI selection doesn't work**: Make sure you're clicking and dragging the red box
- **If no detection**: Check that the waveform is actually visible in the selected ROI area
- **If preview window doesn't show**: The program is still running, check the command window for output
