from ultralytics import YOLO
import cv2
import numpy as np

"""(video_path, model_path="yolo11l.pt", conf=0.1,
                            output_path="new_output.mp4", show_window=False,
                            use_colab=False, skip_frames=60, tile_size=1984)"""

def detect_persons_in_video(video_path, model_path, conf,
                            output_path, show_window,
                            use_colab, skip_frames, tile_size):
    """
    skip_frames = 1 → process every frame
    skip_frames = 2 → process every 2nd frame (50% faster)
    skip_frames = 3 → process every 3rd frame (≈66% faster)
    tile_size = tile ka size (default 640x640)
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    person_ids = set()
    frames_with_person = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            out.write(frame)
            continue

        current_frame_has_person = False

        # 🔹 Frame ko tiles me todna
        results_all = []
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = frame[y:y+tile_size, x:x+tile_size]
                if tile.shape[0] == 0 or tile.shape[1] == 0:
                    continue
                results = model.track(
                    tile,
                    persist=True,
                    conf=conf,
                    iou=0.7,
                    agnostic_nms=True,
                    max_det=1000,
                    classes=[0],
                    imgsz=tile_size
                )
                # bounding box ko global frame ke hisaab se shift karna
                if len(results) > 0:
                    b = results[0].boxes
                    if b.id is not None:
                        for i in range(len(b.id)):
                            results_all.append((
                                int(b.id[i]),
                                int(b.cls[i]),
                                float(b.conf[i]),
                                [int(b.xyxy[i][0])+x, int(b.xyxy[i][1])+y,
                                 int(b.xyxy[i][2])+x, int(b.xyxy[i][3])+y]
                            ))

        for track_id, cls, score, (x1, y1, x2, y2) in results_all:
            if cls != 0:  # only persons
                continue
            person_ids.add(track_id)
            current_frame_has_person = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if current_frame_has_person:
            frames_with_person += skip_frames

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 10), (400, 100), (0, 0, 0), -1)
        frame[10:100, 5:400] = cv2.addWeighted(overlay[10:100, 5:400], 0.5,
                                               frame[10:100, 5:400], 0.5, 0)

        """cv2.putText(frame, f"Unique Persons: {len(person_ids)}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Person Time: {frames_with_person / fps:.2f} s", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)"""

        out.write(frame)

        """if show_window:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break"""

    cap.release()
    out.release()
    """if show_window:
        cv2.destroyAllWindows()"""

    #print(f"✅ Total unique Persons detected: {len(person_ids)}")
    #print(f"✅ Total duration Persons appeared: {frames_with_person / fps:.2f} seconds")
    return len(person_ids)


# Example usage
"""""if __name__ == "__main__":
    video_path = "C10.webm"
    detect_persons_in_video(video_path,
                            model_path="yolo11l.pt",
                            conf=0.1,
                            output_path=f"{video_path.split('.')[0]}_1output.mp4",
                            show_window=True,
                            use_colab=False,
                            skip_frames=60,
                            tile_size=1984)  # 🔹 tiling enable"""""