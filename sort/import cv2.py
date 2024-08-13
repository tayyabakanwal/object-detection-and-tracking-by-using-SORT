import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

def main():
    # Load the YOLOv8 model
    model = YOLO('C:\\Users\\hp\\Desktop\\sort\\train39\\tom.pt')  # Ensure this is the correct model file

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Initialize the SORT tracker
    tracker = Sort()

    # Define the class name to filter
    target_class_name = 'tom'

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Make predictions
        results = model(frame)

        detections = []

        # Extract detection details
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                label = result.names[class_id]  # Get the class name

                if label == target_class_name:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    detections.append([x1, y1, x2, y2, confidence])

        # Convert detections to numpy array for tracker
        detections_np = np.array(detections)

        # Ensure detections_np is not empty
        if detections_np.size == 0:
            tracks = np.empty((0, 5))  # Empty array to ensure no tracking is performed
        else:
            # Update tracker with YOLOv8 detections
            tracks = tracker.update(detections_np[:, :4])  # Pass only the bbox coordinates for tracking

        # Draw bounding boxes and tracking IDs
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frame, f"ID: {int(track_id)}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            
            # Log tracking information to the terminal
            print(f"Tracking ID: {int(track_id)} - BBox: ({x1}, {y1}), ({x2}, {y2})")

        # Display the frame with bounding boxes
        cv2.imshow('YOLOv8 Detection and SORT Tracking', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
