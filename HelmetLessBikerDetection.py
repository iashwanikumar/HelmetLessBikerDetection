{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# **Mount the Google drive**\n",
        "The trained models are uploaded to the google drive and inorder to load them the drive is mounted."
      ],
      "metadata": {
        "id": "OuaUXUm4-W_3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "id": "-bL6OThMU16s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nvuncz1Au4HI"
      },
      "source": [
        "%cd /content/drive/MyDrive/HelmetDetection"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Import the required packages**"
      ],
      "metadata": {
        "id": "595wggyR-nno"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7F0ljq2CuoiL"
      },
      "source": [
        "# clone Tensorflow object detection api\n",
        "!git clone https://github.com/tensorflow/models"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uGhZgUMVwR59"
      },
      "source": [
        "# Run to install proto buffers for object detection api\n",
        "!apt-get update\n",
        "!apt-get install -y -qq protobuf-compiler python-pil python-lxml"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LO2dJ31-v4az"
      },
      "source": [
        "import sys\n",
        "sys.path.append('/content/drive/MyDrive/HelmetDetection/models/research/slim')\n",
        "!pip install tf_slim"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3DvKZZ-evrHZ"
      },
      "source": [
        "%cd /content/drive/MyDrive/HelmetDetection/models/research\n",
        "!protoc object_detection/protos/*.proto --python_out=."
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ozRSk4-Mt5Oh"
      },
      "source": [
        "# Import packages\n",
        "import os\n",
        "import cv2\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import sys\n",
        "from google.colab.patches import cv2_imshow\n",
        "\n",
        "# Import utilites\n",
        "from object_detection.utils import label_map_util\n",
        "from object_detection.utils import visualization_utils as vis_util"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ODMSeymU7jnL"
      },
      "source": [
        "pwd"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Configure the paths to trained models**\n",
        "A Faster RCNN model is trained to detect motorcyclists in a given image.\n",
        "\n",
        "A YOLO model is trained to detect helmet in the given image of motorcyclist.\n",
        "\n"
      ],
      "metadata": {
        "id": "ra_zb3-p--hA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "CWD_PATH = '/content/drive/MyDrive/HelmetDetection/'\n",
        "\n",
        "MODEL_RCNN = 'rcnn'\n",
        "\n",
        "MODEL_YOLO='yolo'\n",
        "\n",
        "# Path to frozen detection graph .pb file, which contains the model that is used\n",
        "# for rcnn object detection.\n",
        "PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_RCNN,'frozen_inference_graph.pb')\n",
        "\n",
        "# Path to label map file\n",
        "PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_RCNN,'label_map.pbtxt')\n",
        "\n",
        "# Path to yolo files\n",
        "configPath=os.path.join(CWD_PATH,MODEL_YOLO,'yolov3_custom.cfg')\n",
        "weightsPath=os.path.join(CWD_PATH,MODEL_YOLO,'yolov3_custom_4000.weights')\n",
        "labelsPath=os.path.join(CWD_PATH,MODEL_YOLO,'obj.names')\n"
      ],
      "metadata": {
        "id": "ZwmRo4QJFE7V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Configure the input and output paths**\n",
        "\n",
        "Images of Motorcyclists without helmet are stored in the Output folder.\n"
      ],
      "metadata": {
        "id": "3WV7pc28FYzP"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "scoBh0iIyS4h"
      },
      "source": [
        "# Path to input image\n",
        "IMAGE_NAME = 'input/images/sample.jpeg'\n",
        "\n",
        "# Path to input video\n",
        "VIDEO_NAME = 'input/videos/clip.mp4'\n",
        "\n",
        "OUTPUT_FOLDER='output/'\n",
        "\n",
        "# Path to output\n",
        "PATH_TO_OUTPUT = os.path.join(CWD_PATH, OUTPUT_FOLDER,'/images')\n",
        "VIDEO_OUTPUT = os.path.join(CWD_PATH, OUTPUT_FOLDER,'output_clip.mp4')\n",
        "\n",
        "# Path to image\n",
        "PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)\n",
        "\n",
        "# Path to video\n",
        "PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Configure Faster RCNN Model**"
      ],
      "metadata": {
        "id": "Ft467aHO_Vs3"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dAlwI92lxwaQ"
      },
      "source": [
        "\n",
        "# Load the label map.\n",
        "# Label maps map indices to category names\n",
        "# Here we use internal utility functions\n",
        "label_map = label_map_util.load_labelmap(PATH_TO_LABELS)\n",
        "categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)\n",
        "category_index = label_map_util.create_category_index(categories)\n",
        "\n",
        "# Load the Tensorflow model into memory.\n",
        "detection_graph = tf.Graph()\n",
        "with detection_graph.as_default():\n",
        "    od_graph_def = tf.compat.v1.GraphDef()\n",
        "    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:\n",
        "        serialized_graph = fid.read()\n",
        "        od_graph_def.ParseFromString(serialized_graph)\n",
        "        tf.import_graph_def(od_graph_def, name='')\n",
        "    sess = tf.compat.v1.Session(graph=detection_graph)\n",
        "\n",
        "# Define input and output tensors (i.e. data) for the object detection classifier\n",
        "\n",
        "# Input tensor is the image\n",
        "image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')\n",
        "\n",
        "# Output tensors are the detection boxes, scores, and classes\n",
        "# Each box represents a part of the image where a particular object was detected\n",
        "detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')\n",
        "\n",
        "# Each score represents level of confidence for each of the objects.\n",
        "# The score is shown on the result image, together with the class label.\n",
        "detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')\n",
        "detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')\n",
        "\n",
        "# Number of objects detected\n",
        "num_detections = detection_graph.get_tensor_by_name('num_detections:0')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Configure YOLO Model**"
      ],
      "metadata": {
        "id": "rRfKWMum_gx5"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m-1_dgv6B_7f"
      },
      "source": [
        "net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)\n",
        "ln = net.getLayerNames()\n",
        "ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]\n",
        "LABELS = open(labelsPath).read().strip().split(\"\\n\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Detection Of Motorcyclists Without Helmet From Input Image**\n",
        "The two models are integrated to detect the motorcyclists without helmet from the input image and the images of riders without helmet are stored in the output folder"
      ],
      "metadata": {
        "id": "J5GmKj5d_mdl"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cAo3ttxnyFMa"
      },
      "source": [
        "image = cv2.imread(PATH_TO_IMAGE)\n",
        "image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)\n",
        "image_expanded = np.expand_dims(image, axis=0)\n",
        "\n",
        "# Perform the actual detection by running the model with the image as input\n",
        "(boxes, scores, classes, num) = sess.run(\n",
        "    [detection_boxes, detection_scores, detection_classes, num_detections],\n",
        "    feed_dict={image_tensor: image_expanded})\n",
        "\n",
        "#getting the normalized coordinates of boxes\n",
        "normalizedBoxes = np.squeeze(boxes)\n",
        "normalizedScores = np.squeeze(scores)\n",
        "normalizedClasses = np.squeeze(classes)\n",
        "\n",
        "#set a min thresh score, say 0.8\n",
        "min_score_thresh = 0.8\n",
        "detectedBoxes = normalizedBoxes[normalizedScores > min_score_thresh]\n",
        "detectedClasses = normalizedClasses[normalizedScores > min_score_thresh]\n",
        "\n",
        "#get image size\n",
        "im_height,im_width, _=image.shape\n",
        "size=(im_width, im_height)\n",
        "\n",
        "#get  original coordinates of bike riders in image\n",
        "final_boxes = []\n",
        "for i in range(len(detectedBoxes)):\n",
        "    ymin, xmin, ymax, xmax = detectedBoxes[i]\n",
        "    final_boxes.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])\n",
        "\n",
        "j=0\n",
        "for [left,right,top,bottom] in final_boxes:\n",
        "\n",
        "    l=int(round(left))\n",
        "    r=int(round(right))\n",
        "    t=int(round(top))\n",
        "    b=int(round(bottom))\n",
        "    croppedImage=image[t:b,l:r]  # Extract each bike rider\n",
        "\n",
        "    cv2.rectangle(image, (l,t), (r,b), (255,0,0), 2)\n",
        "\n",
        "    # perform padding to cropped image\n",
        "    rows = croppedImage.shape[0]\n",
        "    cols = croppedImage.shape[1]\n",
        "    padding=0\n",
        "    if rows > cols:\n",
        "        padding = int((rows-cols) / 2)\n",
        "        paddedImg=cv2.copyMakeBorder(croppedImage, 0, 0, padding, padding,  cv2.BORDER_CONSTANT, (0, 0, 0))\n",
        "    else:\n",
        "        paddedImg=croppedImage\n",
        "\n",
        "    # image preprocessing for yolo model\n",
        "    (H, W) = paddedImg.shape[:2]\n",
        "    blob = cv2.dnn.blobFromImage(paddedImg, 1 / 255.0, (416, 416), swapRB=True, crop=False)\n",
        "\n",
        "    # run yolo model\n",
        "    net.setInput(blob)\n",
        "    layerOutputs = net.forward(ln)\n",
        "\n",
        "    # Initializing for getting box coordinates, confidences, classid\n",
        "    boxesH = []\n",
        "    confidencesH = []\n",
        "    classIDsH = []\n",
        "    thresholdH = 0.15\n",
        "\n",
        "    # getting coordinates of output predictions and performing NMS\n",
        "    for output in layerOutputs:\n",
        "        for detection in output:\n",
        "            confidenceOfEachClass = detection[5:]\n",
        "            classIDH = np.argmax(confidenceOfEachClass)    # get class with max confidence\n",
        "            confidenceH = confidenceOfEachClass[classIDH]  # get the max confidence\n",
        "            if confidenceH > thresholdH:\n",
        "                boxH = detection[0:4] * np.array([W, H, W, H])\n",
        "                (centerX, centerY, widthH, heightH) = boxH.astype(\"int\")\n",
        "                x = int(centerX - (widthH / 2))\n",
        "                y = int(centerY - (heightH / 2))\n",
        "                boxesH.append([x, y, int(widthH), int(heightH)])\n",
        "                confidencesH.append(float(confidenceH))\n",
        "                classIDsH.append(classIDH)\n",
        "    idxs = cv2.dnn.NMSBoxes(boxesH, confidencesH, thresholdH, 0.1)\n",
        "\n",
        "    # bounding boxes for helmet- no helmet\n",
        "    if len(idxs) > 0:\n",
        "        for i in idxs.flatten():\n",
        "            (x, y) = (boxesH[i][0], boxesH[i][1])\n",
        "            (w, h) = (boxesH[i][2], boxesH[i][3])\n",
        "            if LABELS[classIDsH[i]] == 'Helmet':\n",
        "                color = (0, 255, 0)\n",
        "                cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)\n",
        "                text = \"{}\".format(LABELS[classIDsH[i]])\n",
        "                cv2.putText(croppedImage, text, (x//2, y+ 4*h//3),\n",
        "                cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)\n",
        "            if (LABELS[classIDsH[i]] == 'No Helmet'):\n",
        "                # store images with no helmet\n",
        "                name=IMAGE_NAME.split(\".\")[0].split('/')[-1]\n",
        "                j+=1\n",
        "                cv2.imwrite( PATH_TO_OUTPUT + '{}.jpg'.format(name+str(j)),croppedImage)\n",
        "\n",
        "                # draw bounding box\n",
        "                color = (0, 0, 255)\n",
        "                cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)\n",
        "                text = \"{}\".format(LABELS[classIDsH[i]])\n",
        "                cv2.putText(croppedImage, text, (x//2, y + 4* h//3),\n",
        "                cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)\n",
        "\n",
        "\n",
        "cv2_imshow(image)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Detection Of Motorcyclists Without Helmet From Input Video Stream**\n",
        "The two models are integrated to detect the motorcyclists without helmet from the input video stream and the images of riders without helmet are stored in the output folder, along with the resulting video stream"
      ],
      "metadata": {
        "id": "8B3SEpWqAq5W"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6141ppv04fTR"
      },
      "source": [
        "\n",
        "video = cv2.VideoCapture(PATH_TO_VIDEO)\n",
        "\n",
        "images=[]\n",
        "f=0\n",
        "while(video.isOpened()):\n",
        "  if f%3!=0:\n",
        "    ret,image = video.read()\n",
        "    f+=1\n",
        "    continue\n",
        "  ret,image = video.read()\n",
        "  if ret == True:\n",
        "    image=cv2.resize(image,None,fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)\n",
        "    image_expanded = np.expand_dims(image, axis=0)\n",
        "\n",
        "    # Perform the actual detection by running the model with the image as input\n",
        "    (boxes, scores, classes, num) = sess.run(\n",
        "        [detection_boxes, detection_scores, detection_classes, num_detections],\n",
        "        feed_dict={image_tensor: image_expanded})\n",
        "\n",
        "\n",
        "    #getting the normalized coordinates of boxes\n",
        "    normalizedBoxes = np.squeeze(boxes)\n",
        "    normalizedScores = np.squeeze(scores)\n",
        "    normalizedClasses = np.squeeze(classes)\n",
        "\n",
        "    #set a min thresh score, say 0.8\n",
        "    min_score_thresh = 0.8\n",
        "    detectedBoxes = normalizedBoxes[normalizedScores > min_score_thresh]\n",
        "    detectedClasses= normalizedClasses[normalizedScores > min_score_thresh]\n",
        "\n",
        "    #get image size\n",
        "    im_height,im_width, _=image.shape\n",
        "    size=(im_width, im_height)\n",
        "\n",
        "    #get coordinates of bike riders in image\n",
        "    final_boxes = []\n",
        "    for i in range(len(detectedBoxes)):\n",
        "        ymin, xmin, ymax, xmax = detectedBoxes[i]\n",
        "        final_boxes.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])\n",
        "\n",
        "    j=0\n",
        "    for [left,right,top,bottom] in final_boxes:\n",
        "        l=int(round(left))\n",
        "        r=int(round(right))\n",
        "        t=int(round(top))\n",
        "        b=int(round(bottom))\n",
        "        croppedImage=image[t:b,l:r]  # Extract each bike rider\n",
        "\n",
        "        cv2.rectangle(image, (l,t), (r,b), (255,0,0), 2)\n",
        "\n",
        "        # perform padding to cropped image\n",
        "        rows = croppedImage.shape[0]\n",
        "        cols = croppedImage.shape[1]\n",
        "        padding=0\n",
        "        if rows > cols:\n",
        "            padding = int((rows-cols) / 2)\n",
        "            paddedImg=cv2.copyMakeBorder(croppedImage, 0,0,padding, padding,  cv2.BORDER_CONSTANT, (0, 0, 0))\n",
        "        else:\n",
        "            paddedImg=croppedImage\n",
        "\n",
        "        # image preprocessing for yolo model\n",
        "        (H, W) = paddedImg.shape[:2]\n",
        "        blob = cv2.dnn.blobFromImage(paddedImg, 1 / 255.0, (416, 416), swapRB=True, crop=False)\n",
        "\n",
        "        # run yolo model\n",
        "        net.setInput(blob)\n",
        "        layerOutputs = net.forward(ln)\n",
        "\n",
        "        # Initializing for getting box coordinates, confidences, classid\n",
        "        boxesH = []\n",
        "        confidencesH = []\n",
        "        classIDsH = []\n",
        "        thresholdH = 0.15\n",
        "\n",
        "        # getting coordinates of output predictions and performing NMS\n",
        "        for output in layerOutputs:\n",
        "            for detection in output:\n",
        "                confidenceOfEachClass = detection[5:]\n",
        "                classIDH = np.argmax(confidenceOfEachClass)    # get class with max confidence\n",
        "                confidenceH = confidenceOfEachClass[classIDH]  # get the max confidence\n",
        "                if confidenceH > thresholdH:\n",
        "                    boxH = detection[0:4] * np.array([W, H, W, H])\n",
        "                    (centerX, centerY, widthH, heightH) = boxH.astype(\"int\")\n",
        "                    x = int(centerX - (widthH / 2))\n",
        "                    y = int(centerY - (heightH / 2))\n",
        "                    boxesH.append([x, y, int(widthH), int(heightH)])\n",
        "                    confidencesH.append(float(confidenceH))\n",
        "                    classIDsH.append(classIDH)\n",
        "        idxs = cv2.dnn.NMSBoxes(boxesH, confidencesH, thresholdH, 0.1)\n",
        "\n",
        "        # bounding boxes for helmet- no helmet\n",
        "        if len(idxs) > 0:\n",
        "            for i in idxs.flatten():\n",
        "                (x, y) = (boxesH[i][0], boxesH[i][1])\n",
        "                (w, h) = (boxesH[i][2], boxesH[i][3])\n",
        "                if LABELS[classIDsH[i]] == 'Helmet':\n",
        "                    color = (0, 255, 0)\n",
        "                    cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)\n",
        "                    text = \"{}\".format(LABELS[classIDsH[i]])\n",
        "                    cv2.putText(croppedImage, text, (x//2, y+ 4*h//3),\n",
        "                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)\n",
        "                if (LABELS[classIDsH[i]] == 'No Helmet'):\n",
        "                    #store images with no helmet\n",
        "                    name=VIDEO_NAME.split(\".\")[0].split('/')[-1]+\"_\"+str(f)+\"_\"\n",
        "                    j+=1\n",
        "                    cv2.imwrite(PATH_TO_OUTPUT + '{}.jpg'.format(name+str(j)),croppedImage)\n",
        "                    # draw bounding boxes\n",
        "                    color = (0, 0, 255)\n",
        "                    cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)\n",
        "                    text = \"{}\".format(LABELS[classIDsH[i]])\n",
        "                    cv2.putText(croppedImage, text, (x//2, y + 4* h//3),\n",
        "                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)\n",
        "        images.append(image)\n",
        "    f+=1\n",
        "    #cv2_imshow(image)\n",
        "  else:\n",
        "    break\n",
        "video.release()\n",
        "\n",
        "\n",
        "out = cv2.VideoWriter(VIDEO_OUTPUT,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)\n",
        "for i in range(len(images)):\n",
        "    out.write(images[i])\n",
        "out.release()"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}