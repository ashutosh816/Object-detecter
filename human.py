import cv2
import numpy as np

# Load class names
with open("coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# Detect: person(0), car(2), dog(16), bottle(39)
TARGET_CLASS = [0, 2, 16, 39]

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    object_count = {}

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id in TARGET_CLASS:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Fix: handle tuple / None cases
    if indexes is not None and len(indexes) > 0:
        indexes = indexes.flatten()  # ✅ Now safe to flatten

        for i in indexes:
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]

            # Count objects
            object_count[label] = object_count.get(label, 0) + 1

            # Draw box & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display object count
    y_offset = 30
    for obj, count in object_count.items():
        cv2.putText(frame, f"{obj}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 25

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
