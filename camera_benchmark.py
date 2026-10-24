
import cv2
import time


def benchmark(camera_index=0, seconds=10):
    print("Starting camera benchmark...", flush=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("DirectShow failed. Trying default backend.", flush=True)
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Could not open camera.", flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Backend:", cap.getBackendName(), flush=True)
    print(
        "Resolution:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        flush=True,
    )
    print("Reported FPS:", cap.get(cv2.CAP_PROP_FPS), flush=True)
    print(f"Capturing for {seconds} seconds...", flush=True)

    frames = 0
    start = time.perf_counter()

    try:
        while time.perf_counter() - start < seconds:
            ok, frame = cap.read()

            if not ok:
                print("ERROR: Frame capture failed.", flush=True)
                break

            frames += 1

        elapsed = time.perf_counter() - start

        print(f"Elapsed: {elapsed:.2f} seconds", flush=True)
        print(f"Frames: {frames}", flush=True)
        print(
            f"Measured capture FPS: {frames / elapsed:.2f}",
            flush=True,
        )

    finally:
        cap.release()
        print("Camera released.", flush=True)


if __name__ == "__main__":
    benchmark()
