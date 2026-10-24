
import cv2
import time


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("DirectShow failed. Trying default camera backend...")
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Request a lighter video mode.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # May be unsupported by some cameras/backends.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Resolution: {width}x{height}")
    print(f"Camera-reported FPS: {reported_fps:.1f}")
    print("Press Q in the camera window to exit.")

    frames = 0
    interval_start = time.perf_counter()
    display_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame capture failed.")
                break

            frames += 1
            now = time.perf_counter()
            elapsed = now - interval_start

            # Update the rolling FPS once per second.
            if elapsed >= 1.0:
                display_fps = frames / elapsed
                frames = 0
                interval_start = now

            cv2.putText(
                frame,
                f"Capture FPS: {display_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("PulseVision - Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
