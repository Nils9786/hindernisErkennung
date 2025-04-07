import cv2
import numpy as np 


net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

target_class = "stop sign"

img = cv2.imread("testbild.jpg")
height, width = img.shape[:2]

# === 3. Bild für YOLO vorbereiten ===
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# === 4. Vorwärtsdurchlauf durchs Netz ===
output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

# === 5. Ergebnisse verarbeiten ===
conf_threshold = 0.5
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold and classes[class_id] == target_class:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Box & Label zeichnen
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, f"{target_class} {int(confidence*100)}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# === 6. Ergebnis anzeigen ===
cv2.imshow("Erkennung", img)
cv2.waitKey(0)
cv2.destroyAllWindows()