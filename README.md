# Slider Captcha Automation

A Python-based solution for automating slider captcha solving using YOLOv8 for puzzle piece detection. This project demonstrates how to use computer vision and machine learning to solve slider captchas automatically.

![Python Version](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3.99-red)

## Features

- Automated slider captcha solving
- Real-time puzzle piece detection using YOLOv8
- Customizable screen positions for different resolutions
- Interactive keyboard controls
- Support for both manual and automatic solving modes
- GPU acceleration support for faster detection
- Training notebook for custom model training

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for better performance)
- Windows OS (tested on Windows 10)
- Screen resolution: 1920x1080 (configurable for other resolutions)

## Installation

### Option 1: Using Anaconda (Recommended)

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create a new conda environment:
```bash
conda create -n slider-captcha python=3.9
conda activate slider-captcha
```

3. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install other dependencies:
```bash
pip install -r requirement.txt
```

### Option 2: Using Python venv

1. Create a new Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirement.txt
```

## Model Training

This project includes a Jupyter notebook (`train_puzzle_detector.ipynb`) for training your own YOLOv8 model. The notebook provides:

- Step-by-step guide for dataset preparation
- Model training configuration
- Validation and testing procedures
- Best practices for data collection and labeling
- Tips for model optimization

To use the training notebook:

1. Install Jupyter:
```bash
pip install jupyter
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `train_puzzle_detector.ipynb`

### Dataset Preparation
1. Collect 500+ slider captcha images
2. Label puzzle pieces using [CVAT](https://www.cvat.ai/) or [LabelImg](https://github.com/tzutalin/labelImg)
3. Split dataset into train/val/test sets (70/20/10)

### Training Process
1. Used YOLOv8n (nano) model as the base
2. Training configuration:
   - Epochs: 100
   - Batch size: 16
   - Image size: 640x640
   - Learning rate: 0.01
   - Optimizer: SGD

### Training Command
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640 batch=16
```

The trained model is saved in `runs/detect/puzzle_detector/weights/best.pt`

## Configuration

The script uses predefined positions for the captcha elements. Update these positions based on your screen resolution:

```python
refresh_button_position = (2137, 712)      # Position of refresh button
slider_button_position = (1734, 997)       # Position of slider button
source_puzzle_position = [1681, 751, 1786, 775]  # Source puzzle area [x1, y1, x2, y2]
destination_puzzle_position = [1903, 750, 2108, 773]  # Destination area [x1, y1, x2, y2]
puzzle_position = [1677, 678, 490, 275]    # Full puzzle area [x1, y1, width, height]
```

To calibrate positions:
1. Run the script
2. Press 'p' to print current mouse position
3. Move your mouse to each element and note the coordinates
4. Update the positions in the code

## Usage

1. Ensure the trained model is in the correct location:
   ```
   runs/detect/puzzle_detector/weights/best.pt
   ```

2. Run the script:
   ```bash
   python slider_captcha_solver.py
   ```

3. Available commands:
   - 's' - Solve captcha manually
   - 'r' - Refresh captcha
   - 'p' - Print current mouse position (for calibration)
   - 't' - Auto-detect and solve using YOLOv8
   - 'q' - Quit the program

## Project Structure

```
slider_captcha_fix/
├── .gitignore          # Git ignore file
├── LICENSE             # MIT License
├── README.md          # This file
├── requirement.txt    # Python dependencies
├── slider_captcha_solver.py  # Main script
├── app.py             # Web interface
├── train_puzzle_detector.ipynb  # Training notebook
└── runs/              # Training outputs and model weights
    └── detect/
        └── puzzle_detector/
            └── weights/
                └── best.pt  # Trained model
```

## Dependencies

- pyautogui==0.9.54 - For mouse control and screen capture
- keyboard==0.13.5 - For keyboard input handling
- ultralytics==8.3.99 - YOLOv8 implementation
- opencv-python==4.8.1.78 - Computer vision operations
- numpy==1.24.3 - Numerical operations
- Pillow==10.0.0 - Image processing
- torch>=2.0.0 - PyTorch for deep learning
- torchvision>=0.15.0 - Computer vision utilities
- jupyter - For running the training notebook

## Troubleshooting

1. Model not found:
   - Verify the model path in the code
   - Ensure the model file exists in the correct location
   - Check file permissions

2. Position calibration issues:
   - Use the 'p' key to get current mouse positions
   - Update position variables in the code
   - Verify screen resolution matches configured positions

3. Performance issues:
   - Ensure CUDA is properly installed if using GPU
   - Check GPU memory usage
   - Consider using a smaller YOLO model if needed

4. Installation problems:
   - Verify Python version (3.9+)
   - Check CUDA compatibility
   - Ensure all dependencies are installed correctly

5. Training issues:
   - Check dataset format and structure
   - Verify labeling consistency
   - Monitor GPU memory during training
   - Adjust batch size if needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- PyAutoGUI for automation
- OpenCV community for computer vision tools

## Disclaimer

This tool is for educational purposes only. Use responsibly and in accordance with the terms of service of the websites you interact with.
