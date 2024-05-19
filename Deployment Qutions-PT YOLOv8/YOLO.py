from ultralytics import YOLO
import cv2
import math

# Define class descriptions
class_descriptions = {
    'layak-produksi-berbuah': "POHON BERBUAH: Pohon ini sedang berbuah dan layak untuk diproduksi.",
    'layak-produksi-tidak-berbuah': "POHON TIDAK BERBUAH: Pohon ini sehat namun belum berbuah.",
    'tidak-layak-produksi-berjamur': "POHON BERJAMUR: Pohon ini terkena jamur dan tidak layak untuk produksi.",
    'tidak-layak-produksi-kering': "POHON KERING: Pohon ini kering dan tidak layak untuk produksi.",
    'tidak-layak-produksi-keropos': "POHON CACAT/KEROPOS: Pohon ini keropos dan tidak layak untuk produksi."
}


def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("YOLO-Weights/qutions300.pt")
    classNames = ['layak-produksi-berbuah', 'layak-produksi-tidak-berbuah', 'tidak-layak-produksi-berjamur',
                  'tidak-layak-produksi-kering', 'tidak-layak-produksi-keropos']
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Get description for the class
                description = class_descriptions.get(class_name, "Deskripsi tidak tersedia")

                # Modify class_name to the desired labels
                if class_name.startswith('layak-produksi'):
                    class_name = 'Layak Produksi'
                elif class_name.startswith('tidak-layak-produksi'):
                    class_name = 'Tidak Layak Produksi'

                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'Layak Produksi':
                    color = (0, 128, 0)
                else:
                    color = (255, 0, 0)
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                    # Add description to the image
                    cv2.putText(img, description, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        yield img