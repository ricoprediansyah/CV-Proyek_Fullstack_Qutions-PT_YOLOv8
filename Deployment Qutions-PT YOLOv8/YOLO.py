from ultralytics import YOLO
import cv2
import math

class_descriptions = {
    'layak-produksi-berbuah': "POHON BERBUAH: Pohon ini sedang berbuah dan layak untuk diproduksi.",
    'layak-produksi-tidak-berbuah': "POHON TIDAK BERBUAH: Pohon ini sehat namun belum berbuah.",
    'tidak-layak-produksi-berjamur':  "MENANGGULANGI POHON BERJAMUR: Langkah-langkah yang bisa diambil adalah dengan mengidentifikasi jenis jamur yang menyerang pohon sawit, kemudian menggunakan fungisida yang sesuai dan melakukan aplikasinya secara berkala. Selain itu, sanitasi kebun juga penting dilakukan dengan membersihkan area sekitar pohon dari daun kering dan buah busuk serta melakukan pemangkasan daun yang terinfeksi. Pastikan juga untuk memberikan pemupukan yang seimbang dan memperhatikan drainase tanah yang baik. Pengaturan jarak tanam yang cukup antara pohon sawit juga diperlukan untuk menjaga sirkulasi udara yang baik. Selanjutnya, pertimbangkan penggunaan agen biokontrol dan lakukan rotasi tanaman. Terakhir, lakukan pemantauan rutin untuk mendeteksi tanda-tanda awal infeksi jamur dan ambil tindakan yang diperlukan. Dengan menerapkan langkah-langkah ini, diharapkan pohon sawit dapat terhindar dari serangan jamur dan produktivitasnya dapat ditingkatkan.",
    'tidak-layak-produksi-kering': "MENANGGULANGI POHON KERING: Langkah-langkah yang dapat dilakukan antara lain adalah memantau kondisi tanah secara teratur untuk mengukur kelembapan, menerapkan irigasi yang tepat sesuai dengan kebutuhan tanaman, memberikan pemupukan untuk meningkatkan daya tahan tanaman terhadap kekeringan, mengendalikan gulma di sekitar pohon, memastikan sistem drainase tanah berfungsi dengan baik, melakukan pemangkasan pada bagian pohon yang mengalami kekeringan, melakukan pemeliharaan rutin seperti penyiraman dan pemupukan, serta berkonsultasi dengan ahli pertanian jika kondisi pohon terus memburuk. Dengan langkah-langkah tersebut, diharapkan dapat membantu memulihkan kondisi pohon sawit dan mencegah kerugian lebih lanjut akibat kekeringan.",
    'tidak-layak-produksi-keropos': "MENANGGULANGI POHON CACAT/KEROPOS: langkah-langkah yang dapat diambil meliputi identifikasi penyebab cacat atau keropos pada pohon, seperti serangan hama atau penyakit, dan menerapkan langkah pengendalian yang sesuai, seperti penggunaan pestisida atau fungisida. Selain itu, pemangkasan bagian pohon yang terinfeksi, penerapan praktik sanitasi yang baik di sekitar kebun, dan pemupukan yang tepat juga dapat membantu mempercepat pemulihan pohon. Penting juga untuk melakukan pemantauan rutin terhadap kondisi pohon dan berkonsultasi dengan ahli pertanian untuk mendapatkan saran yang lebih spesifik dan efektif dalam menangani masalah tersebut."
}

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
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

                if class_name.startswith('layak-produksi'):
                    class_name = 'Layak Produksi'
                elif class_name.startswith('tidak-layak-produksi'):
                    class_name = 'Tidak Layak Produksi'

                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = (0, 128, 0) if class_name == 'Layak Produksi' else (255, 0, 0)
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img

def detect_descriptions(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    model = YOLO("YOLO-Weights/qutions300.pt")
    classNames = ['layak-produksi-berbuah', 'layak-produksi-tidak-berbuah', 'tidak-layak-produksi-berjamur',
                  'tidak-layak-produksi-kering', 'tidak-layak-produksi-keropos']
    descriptions = set()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = classNames[cls]
                description = class_descriptions.get(class_name, "Deskripsi tidak tersedia")
                descriptions.add(description)
    cap.release()
    return list(descriptions)
