# object-detection-and-tracking-by-using-SORT

Here's a comprehensive guide for your project involving YOLOv8 and SORT tracker for object detection and tracking. This guide includes project details, system requirements, installation steps, dataset labeling, model training, monitoring, and more.

---

## **Project Details**

**Objective:**  
To develop a real-time object detection and tracking system using YOLOv8 for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking.

**Components:**
- YOLOv8 (You Only Look Once version 8) for object detection.
  
- SORT (Simple Online and Realtime Tracking) for tracking objects across frames.
  
- OpenCV for video capture and visualization.

---

## **System Requirements**

**Hardware:**

- A computer with a decent GPU (for training) and CPU.
  
- Webcam or video source for real-time testing.

**Software:**

- Operating System: Windows, Linux, or macOS.
  
- Python 3.8 or higher.
  
- Required Libraries: OpenCV, NumPy, Ultralytics YOLO, SORT, and other dependencies.

**Hardware Recommendations:**

- GPU: NVIDIA RTX 20 series or higher for efficient training.
  
- CPU: Multi-core processor (e.g., Intel i7 or AMD Ryzen).

---

## **Installation of Required Libraries**

```
pip install opencv-python numpy ultralytics sort
```

**Note:** Ensure you have the `sort` library and the appropriate YOLOv8 model files. You may need to install additional dependencies based on your environment.

---

## **Labeling Dataset with LabelImg**

1. **Download and Install LabelImg:**
   - Clone the repository:
     ```
     git clone https://github.com/tzutalin/labelImg.git
     ```
   - Navigate to the directory and install dependencies:
     ```
     cd labelImg
     pip install PyQt5 lxml
     python setup.py install
     ```
   - Run LabelImg:
     ```
     labelImg
     ```

2. **Label Images:**
   - Open your images in LabelImg.
     
   - Draw bounding boxes around objects and assign class labels.
     
   - Save annotations in YOLO format (`.txt` files with the same name as the image).

---

## **Annotation Formats**

**YOLO Format:**

- Each `.txt` file corresponds to one image.
  
- Each line in the file represents a bounding box in the format: `class_id x_center y_center width height`.
  
- Coordinates are normalized to the range [0, 1].

---

## **Training YOLOv8**

1. **Prepare Dataset:**
   
   - Organize images and annotations in directories.
     
   - Create a `.yaml` configuration file for YOLOv8 specifying paths to training and validation data.

2. **Train the Model:**
   
   ```
   from ultralytics import YOLO

   # Load YOLOv8 model
   
   model = YOLO('yolov8n.yaml')  # Load a pretrained model or specify your config

   # Train the model
   model.train(data='data.yaml', epochs=50, imgsz=640)
   ```

3. **Monitor Training:**
   
   - Use TensorBoard or built-in logging for visualizing training progress.

---

## **Monitoring Training**

- **TensorBoard:** Run TensorBoard to monitor training metrics.
  
  ```
  tensorboard --logdir runs/train
  ```

- **Logging:** Use the `logger` module or built-in logging features in YOLOv8 to check progress.

---

## **Confusion Matrix**

- After training, evaluate the model on a validation set and use metrics to generate a confusion matrix. This can be done using tools like Scikit-learn.
  
  ```python
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  
  y_true = [...]  # True labels
  y_pred = [...]  # Predicted labels
  
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()
  ```

---

## **Flowchart of Process**

1. **Capture Video Frames**
   
   - Use webcam or video file.

2. **Object Detection**
   
   - YOLOv8 model detects objects.

3. **Tracking**
   
   - Use SORT to track objects across frames.

4. **Visualization**
   
   - Draw bounding boxes and tracking IDs on frames.

5. **Display**
   
   - Show results in a window.

6. **Save/Log Results**
    
   - Optional: Save results or log data.

---

## **Disclaimer**

- **Accuracy:** The effectiveness of the object detection and tracking system is dependent on the quality and diversity of the dataset and the training parameters.
  
- **Model:** Ensure you use a correctly trained model and appropriate version of YOLOv8. Model performance may vary based on the specific use case and environment.
  
- **Ethical Use:** Use the technology responsibly and ensure compliance with privacy and data protection regulations.

---

Feel free to modify this guide based on your specific project requirements and environment.
