import os
import cv2

# Dictionary for class names
class_names = {
    0: "Sleeve",
    1: "Shirt",
    2: "Neckline"
}
# Dictionary for colors 
class_colors = {
    0: (0, 255, 0),
    1: (255, 0, 0), 
    2: (0, 255, 255)
}

def get_pred(model, img):
    results = model.predict(img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()  
        classes = result.boxes.cls.cpu().numpy()  
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = class_names.get(class_id, 'Unknown')
            color = class_colors.get(class_id, (0, 255, 255)) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{class_name} {conf:.2f}', (x1, y1+abs(y1-y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

if __name__ == "__main__":
    from ultralytics import YOLO
    model_path = 'models/best.pt'
    img_path = 'dataset/valid/images/1019736_dataset 2025-01-21 17-17-04_MEN-Pants-id_00002898-09_7_additional.png'
    out_path = 'output'
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    pimg = get_pred(model, img)
    cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), pimg)