from ultralytics import YOLO
from sklearn.metrics import classification_report
import os

MODEL_PATH = "models/best0.827modele2.pt"
TEST_PATH = "data/dataset_cls/test"

model = YOLO(MODEL_PATH)

class_names = model.names
label_map = {v: k for k, v in class_names.items()}

y_true = []
y_pred = []

for class_dir in os.listdir(TEST_PATH):

    class_path = os.path.join(TEST_PATH, class_dir)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):

        if not img_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        img_path = os.path.join(class_path, img_name)

        result = model.predict(
            img_path,
            save=False,
            verbose=False
        )[0]

        true_idx = label_map[class_dir]

        y_true.append(true_idx)
        y_pred.append(result.probs.top1)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[class_names[i] for i in sorted(class_names)]
    )
)