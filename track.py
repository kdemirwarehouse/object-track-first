"""
Nesne Takibi - YOLO11 ile Gelişmiş Takip
YOLOv11 kullanarak videoda nesne tespiti, takip ve sayım.

Özellikler:
  - Sınıfa göre benzersiz nesne sayımı
  - Sanal sayım çizgisi (nesnelerin çizgiden geçişini sayar)
  - Takip izi çizimi
  - FPS göstergesi
  - Çıktı videosu kaydetme
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


TRAIL_LEN = 30

prev_centers = {}
line_cross_counts = defaultdict(int)
crossed_ids = set()

CLASS_COLORS = [
    (255, 100,   0), (  0, 200, 255), (100, 255,   0), (200,   0, 255),
    (  0, 255, 180), (255,  50, 150), ( 50, 200,  50), (200, 150,   0),
]


def get_color(class_id):
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def draw_text_box(frame, text, pos, font_scale=0.6, thickness=2, color=(0, 255, 0)):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def check_line_crossing(track_id, class_name, cy, line_y):
    if track_id in prev_centers:
        prev_cy = prev_centers[track_id]
        if (prev_cy < line_y <= cy or cy <= line_y < prev_cy) and track_id not in crossed_ids:
            crossed_ids.add(track_id)
            line_cross_counts[class_name] += 1
    prev_centers[track_id] = cy


def draw_trails(frame, trails):
    for track_id, (pts, color) in trails.items():
        pts_list = list(pts)
        for i in range(1, len(pts_list)):
            if pts_list[i - 1] is None or pts_list[i] is None:
                continue
            alpha = i / len(pts_list)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts_list[i - 1], pts_list[i], color, thickness)


def draw_panel(frame, class_counts, fps, line_cross_counts):
    panel_w = 270
    lines = ["=== NESNE SAYACI ==="]
    for cls, ids in sorted(class_counts.items()):
        gecen = line_cross_counts.get(cls, 0)
        lines.append(f"{cls}: {len(ids)} tespit | {gecen} gecti")
    lines.append(f"--- FPS: {fps:.1f} ---")

    panel_h = len(lines) * 26 + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (14, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def run(args):
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Hata: '{args.video}' videosu açılamadı.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Sanal sayım çizgisi: frame yüksekliğinin %60'ı
    line_y = int(height * 0.60)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, source_fps, (width, height))

    trails = defaultdict(lambda: [deque(maxlen=TRAIL_LEN), None])
    class_counts = defaultdict(set)

    fps = 0.0
    frame_count = 0
    t_start = time.time()

    print("Takip baslatiliyor... Cikmak icin 'q' tusuna bas.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes    = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes  = results[0].boxes.cls.int().cpu().tolist()
            confs    = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                if conf < args.conf:
                    continue

                class_name = results[0].names[cls]
                color = get_color(cls)

                class_counts[class_name].add(track_id)

                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                trails[track_id][0].append((cx, cy))
                trails[track_id][1] = color

                check_line_crossing(track_id, class_name, cy, line_y)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {class_name} {conf:.2f}"
                draw_text_box(frame, label, (x1, y1 - 4), color=color)

        draw_trails(frame, trails)

        # Sanal sayım çizgisi
        cv2.line(frame, (0, line_y), (width, line_y), (0, 120, 255), 2)
        draw_text_box(frame, "Sayim Cizgisi", (width // 2 - 60, line_y - 8),
                      color=(0, 180, 255), font_scale=0.5)

        frame_count += 1
        if time.time() - t_start > 0:
            fps = frame_count / (time.time() - t_start)

        draw_panel(frame, class_counts, fps, line_cross_counts)

        if writer:
            writer.write(frame)

        cv2.imshow("Nesne Takibi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n=== SONUCLAR ===")
    for cls, ids in sorted(class_counts.items()):
        gecen = line_cross_counts.get(cls, 0)
        print(f"  {cls}: {len(ids)} benzersiz nesne tespit edildi, {gecen} tanesi cizgiden gecti")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 ile Nesne Takibi")
    parser.add_argument("--video",  default="traffic.mp4",  help="Video dosyasi yolu")
    parser.add_argument("--model",  default="yolo11m.pt",   help="YOLO model dosyasi")
    parser.add_argument("--output", default=None,           help="Cikti video yolu (opsiyonel)")
    parser.add_argument("--conf",   type=float, default=0.3, help="Guven esigi (varsayilan: 0.3)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
