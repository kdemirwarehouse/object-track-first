"""Görselleştirme yardımcıları — tüm OpenCV çizim işlemleri burada."""

from collections import defaultdict, deque

import cv2

TRAIL_LEN = 30

_CLASS_COLORS = [
    (255, 100,   0), (  0, 200, 255), (100, 255,   0), (200,   0, 255),
    (  0, 255, 180), (255,  50, 150), ( 50, 200,  50), (200, 150,   0),
]


def class_color(class_id: int) -> tuple[int, int, int]:
    return _CLASS_COLORS[class_id % len(_CLASS_COLORS)]


def text_box(frame, text: str, pos: tuple[int, int],
             font_scale: float = 0.6, thickness: int = 2,
             color: tuple = (0, 255, 0)) -> None:
    """Siyah arka planlı metin çizer."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def trails(frame, trail_map: dict) -> None:
    """Her nesnenin hareket izini çizer; iz sona doğru kalınlaşır."""
    for _tid, (pts, color) in trail_map.items():
        pts_list = list(pts)
        for i in range(1, len(pts_list)):
            if pts_list[i - 1] is None or pts_list[i] is None:
                continue
            alpha = i / len(pts_list)
            cv2.line(frame, pts_list[i - 1], pts_list[i], color, max(1, int(3 * alpha)))


def counting_line(frame, line_y: int, width: int) -> None:
    """Sanal sayım çizgisini çizer."""
    cv2.line(frame, (0, line_y), (width, line_y), (0, 120, 255), 2)
    text_box(frame, "Sayim Cizgisi", (width // 2 - 60, line_y - 8),
             color=(0, 180, 255), font_scale=0.5, thickness=1)


def stats_panel(frame, class_counts: dict, fps: float, cross_counts: dict) -> None:
    """Sol üst köşeye yarı saydam istatistik paneli çizer."""
    panel_w = 280
    lines = ["=== NESNE SAYACI ==="]
    for cls, ids in sorted(class_counts.items()):
        gecen = cross_counts.get(cls, 0)
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


def make_trail_map() -> defaultdict:
    return defaultdict(lambda: [deque(maxlen=TRAIL_LEN), None])
