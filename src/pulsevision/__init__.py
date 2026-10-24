
import cv2
import sys
import time
from pathlib import Path

# Allow imports from the src directory.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pulsevision.perception import FaceDetector


def main():
    detector = FaceDetector()

    # Use the faster backend we tested.
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Face detection running. Press Q to exit.")

    frames = 0
    start = time.perf_counter()
    report_time = start
    display_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Frame capture failed.")
                break

            result = detector.detect(frame)
            frames += 1

            if result is not None:
                x, y, w, h = result

                # Draw the detected face.
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "Face detected",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            now = time.perf_counter()

            if now - report_time >= 1.0:
                display_fps = frames / (now - start)
                report_time = now

            cv2.putText(
                frame,
                f"Processing FPS: {display_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("PulseVision - Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
