"""Takip motoru — YOLO çalıştırma, ID yönetimi ve çizgi geçiş mantığı."""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO

from tracker import draw


@dataclass
class TrackConfig:
    video: str = "traffic.mp4"
    model: str = "yolo11m.pt"
    output: str | None = None
    conf: float = 0.3
    line_ratio: float = 0.60  # sanal çizginin frame yüksekliğine oranı


@dataclass
class _State:
    class_counts: defaultdict = field(default_factory=lambda: defaultdict(set))
    cross_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    crossed_ids: set = field(default_factory=set)
    prev_cy: dict = field(default_factory=dict)
    trail_map: defaultdict = field(default_factory=draw.make_trail_map)


def _check_crossing(state: _State, track_id: int, class_name: str, cy: int, line_y: int) -> None:
    if track_id in state.prev_cy:
        prev = state.prev_cy[track_id]
        crossed = (prev < line_y <= cy) or (cy <= line_y < prev)
        if crossed and track_id not in state.crossed_ids:
            state.crossed_ids.add(track_id)
            state.cross_counts[class_name] += 1
    state.prev_cy[track_id] = cy


def run(cfg: TrackConfig) -> None:
    model = YOLO(cfg.model)
    cap = cv2.VideoCapture(cfg.video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: '{cfg.video}'")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    line_y = int(height * cfg.line_ratio)

    writer = None
    if cfg.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.output, fourcc, src_fps, (width, height))

    state = _State()
    fps = 0.0
    frame_count = 0
    t_start = time.time()

    print("Takip baslatiliyor... Cikmak icin 'q' tusuna basin.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes   = results[0].boxes.cls.int().cpu().tolist()
            confs     = results[0].boxes.conf.cpu().tolist()

            for box, tid, cls, conf in zip(boxes, track_ids, classes, confs):
                if conf < cfg.conf:
                    continue

                class_name = results[0].names[cls]
                color = draw.class_color(cls)

                state.class_counts[class_name].add(tid)

                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                state.trail_map[tid][0].append((cx, cy))
                state.trail_map[tid][1] = color

                _check_crossing(state, tid, class_name, cy, line_y)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw.text_box(frame, f"#{tid} {class_name} {conf:.2f}", (x1, y1 - 4), color=color)

        draw.trails(frame, state.trail_map)
        draw.counting_line(frame, line_y, width)

        frame_count += 1
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        draw.stats_panel(frame, state.class_counts, fps, state.cross_counts)

        if writer:
            writer.write(frame)

        cv2.imshow("Nesne Takibi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    _print_results(state)


def _print_results(state: _State) -> None:
    print("\n=== SONUCLAR ===")
    for cls, ids in sorted(state.class_counts.items()):
        gecen = state.cross_counts.get(cls, 0)
        print(f"  {cls}: {len(ids)} benzersiz nesne | {gecen} tanesi cizgiden gecti")
