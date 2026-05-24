Helmet-less Motorcyclist Detection System
A pipeline architecture designed to detect motorcyclists and automatically identify helmet-use violations. This system integrates a Faster R-CNN model (to locate motorcyclists) and a YOLOv3 model (to analyze the cropped rider image for 'Helmet' or 'No Helmet' compliance).
🚀 System Architecture & Workflow
    1. Motorcyclist Detection: The framework utilizes a Faster R-CNN model to pinpoint and crop bike riders from an overall frame or video stream.
    2. Helmet Verification: The cropped image is passed to a custom-trained YOLOv3 network.
    3. Violation Logging: If a "No Helmet" condition is triggered, the frame snapshot is automatically isolated and exported into an explicit output directory for evidence logging.
🛠️ Tech Stack & Requirements
    • Runtime Environment: Google Colab (GPU Accelerator Recommended)
    • Core Frameworks: TensorFlow v1.x/v2.x (via compatibility layers), OpenCV (DNN Module)
    • Supporting Libraries: NumPy, tf_slim, Protobuf Compiler
📂 Project Directory Structure
Ensure your Google Drive folder (/MyDrive/HelmetDetection/) matches this layout before running the notebook:
Plaintext
HelmetDetection/
├── rcnn/
│   ├── frozen_inference_graph.pb     # Pre-trained Faster R-CNN weights
│   └── label_map.pbtxt                # Label mappings for R-CNN
├── yolo/
│   ├── yolov3_custom.cfg              # YOLOv3 network configurations
│   ├── yolov3_custom_4000.weights     # Custom trained YOLOv3 weights
│   └── obj.names                      # Class targets ('Helmet', 'No Helmet')
├── input/
│   ├── images/                        # Base input directory for static images
│   └── videos/                        # Base input directory for video samples
└── output/                            # Target location for logged infractions
📖 Step-by-Step Execution Guide (Google Colab)
Follow these clear milestones to successfully run the notebook pipeline:
Step 1: Open and Initialize the Notebook
    • Copy the project code into a clean Google Colab Notebook.
    • Navigate to Runtime ➔ Change runtime type and set the Hardware Accelerator to GPU.
Step 2: Mount Google Drive
    • Execute the initialization cells to authenticate and mount your personal storage layer.
Python
from google.colab import drive
drive.mount('/content/drive')
    • Ensure that your models, weights, and configurations are uploaded to your Drive directory under /MyDrive/HelmetDetection/.
Step 3: Compile Dependencies & Protobufs
    • Run the environmental cells to clone the TensorFlow models repository and run the protoc compiler. This registers the Object Detection API utilities required for processing raw images.
Step 4: Configure Data Processing Paths
    • Before executing the detection cells, modify the input targets in the script configuration block to match your specific file names:
Python
# Update these strings with your exact filenames before processing
IMAGE_NAME = 'input/images/your_sample_photo.jpeg' 
VIDEO_NAME = 'input/videos/your_traffic_clip.mp4'
Step 5: Execute and Retrieve Logs
    • Run the pipeline processing cells sequentially.
    • Static Images: The system will display the processed frame in real-time inline.
    • Video Streams: The finalized, annotated .mp4 video alongside cropped images tracking every non-compliant rider will generate directly inside the designated /output/ folder in your Google Drive.
🤝 Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create.
    1. Fork the Project
    2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
    3. Commit your Changes (git commit -m 'Add some AmazingFeature')
    4. Push to the Branch (git push origin feature/AmazingFeature)
    5. Open a Pull Request
📄 License
Distributed under the MIT License. See LICENSE for more information.
© 2026 Helmet Detection Project
