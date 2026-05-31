"""Giriş noktası — CLI argümanlarını alıp takip motorunu başlatır."""

import argparse

from tracker import TrackConfig, run


def parse_args() -> TrackConfig:
    parser = argparse.ArgumentParser(
        description="YOLO11 ile gerçek zamanlı nesne takibi ve sayımı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video",       default="traffic.mp4", help="Video dosyası yolu")
    parser.add_argument("--model",       default="yolo11m.pt",  help="YOLO model dosyası")
    parser.add_argument("--output",      default=None,          help="Çıktı video yolu")
    parser.add_argument("--conf",        type=float, default=0.3, help="Güven eşiği")
    parser.add_argument("--line-ratio",  type=float, default=0.60,
                        help="Sanal çizginin frame yüksekliğine oranı (0.0–1.0)")
    args = parser.parse_args()
    return TrackConfig(
        video=args.video,
        model=args.model,
        output=args.output,
        conf=args.conf,
        line_ratio=args.line_ratio,
    )


if __name__ == "__main__":
    run(parse_args())
