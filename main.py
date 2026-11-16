import csv
import os
from datetime import datetime

import cv2

# ---- Settings ----
USE_WEBCAM = True                          # True = webcam, False = video file
VIDEO_PATH = 0 if USE_WEBCAM else "samples/sample.mp4"
FRAME_WIDTH = 900
MIN_NEIGHBORS = 5                          # higher = fewer false detections
MIN_FACE_SIZE = (30, 30)
MAX_DISTANCE = 60                          # max pixels a head moves between frames
MAX_MISSED = 20                            # frames before a lost head's ID is dropped
MIN_MOVING_HITS = 5                        # frames a head must be seen MOVING to be counted
MOTION_RATIO = 0.05                        # fraction of the person's region that must be moving
MOTION_GRACE = 15                          # frames a person stays "moving" after last motion
WARMUP_FRAMES = 30                         # frames for the background model to learn the scene
LOG_INTERVAL = 5                           # seconds between database records
USE_CSV = True                             # False = run without PostgreSQL
CSV_PATH = "people_log.csv"                # where the log is saved
HOURLY_PATH = "hourly_summary.csv"         # hourly summary built from the log


# (c) Haar cascade classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if cascade.empty():
    raise SystemExit("Could not load the Haar cascade file.")

# (b) Segmentation: MOG2 background subtractor separates moving foreground from background
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25,
                                                   detectShadows=True)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def get_motion_mask(frame):
    """Return a clean binary mask: white = moving pixels, black = background."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)           # reduce sensor noise
    mask = bg_subtractor.apply(blurred)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)   # drop shadows (value 127)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)        # remove small specks
    mask = cv2.dilate(mask, KERNEL, iterations=2)                # fill gaps in the body
    return mask


def motion_ratio(mask, box):
    """Fraction of moving pixels in the person's region (face box widened to include the body)."""
    x, y, w, h = box
    mh, mw = mask.shape
    x0, x1 = max(x - w // 2, 0), min(x + w + w // 2, mw)
    y0, y1 = max(y - h // 4, 0), min(y + h * 4, mh)
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return cv2.countNonZero(region) / region.size


class CentroidTracker:
    """(e) Gives each head a persistent ID by matching box centers between frames.
    A person is only counted while moving (checked against the motion mask)."""

    def __init__(self):
        self.next_id = 0
        self.tracks = {}       # id -> track dict (see _new_track)
        self.total_unique = 0  # unique MOVING people counted since the program started

    @staticmethod
    def _new_track(box, center):
        return {"box": box, "center": center, "missed": 0,
                "moving_hits": 0,         # frames seen with motion
                "since_motion": 999,      # frames since motion was last seen
                "confirmed": False,       # True once counted as a unique person
                "announced": False,       # True once its "counted" event is logged
                "is_moving": False}       # last logged state, used to detect changes

    def _check_motion(self, t, mask, learning):
        """Update a track's motion state from the motion mask."""
        if not learning and motion_ratio(mask, t["box"]) >= MOTION_RATIO:
            t["since_motion"] = 0
            t["moving_hits"] += 1
            if not t["confirmed"] and t["moving_hits"] >= MIN_MOVING_HITS:
                t["confirmed"] = True              # moved long enough -> one unique person
                self.total_unique += 1
        else:
            t["since_motion"] += 1
            if not t["confirmed"] and t["since_motion"] > MOTION_GRACE:
                t["moving_hits"] = 0               # stood still too long: restart confirmation

    def update(self, boxes, mask, learning=False):
        centers = [(x + w // 2, y + h // 2) for x, y, w, h in boxes]
        pairs = sorted(
            (((t["center"][0] - c[0]) ** 2 + (t["center"][1] - c[1]) ** 2) ** 0.5, tid, i)
            for tid, t in self.tracks.items() for i, c in enumerate(centers))
        used_t, used_d = set(), set()
        for dist, tid, i in pairs:             # match nearest track to nearest detection
            if dist > MAX_DISTANCE or tid in used_t or i in used_d:
                continue
            t = self.tracks[tid]
            t["box"], t["center"], t["missed"] = boxes[i], centers[i], 0
            self._check_motion(t, mask, learning)
            used_t.add(tid)
            used_d.add(i)
        for tid in list(self.tracks):          # lost tracks
            if tid not in used_t:
                t = self.tracks[tid]
                t["missed"] += 1
                t["since_motion"] += 1
                if t["missed"] > MAX_MISSED:
                    del self.tracks[tid]
        for i, box in enumerate(boxes):        # new heads
            if i not in used_d:
                t = self._new_track(box, centers[i])
                self._check_motion(t, mask, learning)
                self.tracks[self.next_id] = t
                self.next_id += 1

    def moving(self):
        """Confirmed people seen in the current frame who are currently moving."""
        return {i: t for i, t in self.tracks.items()
                if t["confirmed"] and t["missed"] == 0 and t["since_motion"] <= MOTION_GRACE}

    def still(self):
        """Confirmed people seen in the current frame who are standing still (not counted)."""
        return {i: t for i, t in self.tracks.items()
                if t["confirmed"] and t["missed"] == 0 and t["since_motion"] > MOTION_GRACE}

    def collect_events(self):
        """Return (person_id, event) for every change since the last frame."""
        events = []
        for tid, t in self.tracks.items():
            if not t["confirmed"]:
                continue
            moving_now = t["since_motion"] <= MOTION_GRACE
            if not t["announced"]:                       # first time confirmed
                t["announced"], t["is_moving"] = True, True
                events.append((tid, "counted"))
            elif moving_now and not t["is_moving"]:     # was still, now moving
                t["is_moving"] = True
                events.append((tid, "started_moving"))
            elif not moving_now and t["is_moving"]:     # was moving, now still
                t["is_moving"] = False
                events.append((tid, "stopped_moving"))
        return events

def open_csv():
    """Open the CSV log in append mode and write the header if the file is new."""
    if not USE_CSV:
        return None, None
    is_new = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    log_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(log_file)
    if is_new:
        writer.writerow(["timestamp", "person_id", "event", "moving_now", "unique_total"])
    return log_file, writer

def write_hourly_summary():
    """Read the event log and write one summary row per hour."""
    if not os.path.exists(CSV_PATH):
        return
    hours = {}                                   # "2026-11-14 14:00" -> totals
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            hour = row["timestamp"][:13] + ":00"
            h = hours.setdefault(hour, {"counted": 0, "started": 0, "stopped": 0, "peak": 0})
            if row["event"] == "counted":
                h["counted"] += 1
            elif row["event"] == "started_moving":
                h["started"] += 1
            elif row["event"] == "stopped_moving":
                h["stopped"] += 1
            h["peak"] = max(h["peak"], int(row["moving_now"]))
    with open(HOURLY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "new_people_counted", "started_moving_events",
                         "stopped_moving_events", "peak_moving_count"])
        for hour in sorted(hours):
            h = hours[hour]
            writer.writerow([hour, h["counted"], h["started"], h["stopped"], h["peak"]])

def main():
    log_file, writer = open_csv()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open video. Check VIDEO_PATH.")

    tracker = CentroidTracker()
    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # (a) Image operations: resize, grayscale, histogram equalization
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # (b) Segmentation: motion mask (also fed every frame so the background keeps learning)
            mask = get_motion_mask(frame)
            learning = frame_no <= WARMUP_FRAMES      # ignore motion while the model warms up

            # (c) Haar cascade: detect heads (frontal faces)
            boxes = [tuple(map(int, b)) for b in
                     cascade.detectMultiScale(gray, 1.1, MIN_NEIGHBORS, minSize=MIN_FACE_SIZE)]

            # (e) Tracking: assign IDs, count only people who are moving
            tracker.update(boxes, mask, learning)
            people = tracker.moving()

            # Draw results: green = moving (counted), gray = standing still (not counted)
            for t in tracker.still().values():
                x, y, bw, bh = t["box"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (160, 160, 160), 1)
                cv2.putText(frame, "still", (x, max(y - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
            for tid, t in people.items():
                x, y, bw, bh = t["box"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid} moving", (x, max(y - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            status = "Calibrating background..." if learning else f"Moving people: {len(people)}"
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Unique people: {tracker.total_unique}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Log every motion event with a timestamp
            if writer is not None:
                for tid, event in tracker.collect_events():
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     tid, event, len(people), tracker.total_unique])
                    log_file.flush()           # save to disk right away

            cv2.imshow("People Counter", frame)
            # cv2.imshow("Motion Mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                name = datetime.now().strftime("screenshots/result_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, frame)
                cv2.imwrite(name.replace("result_", "mask_"), mask)
                print("Saved", name)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
        if USE_CSV:
            write_hourly_summary()
        print(f"Stopped. Unique moving people counted: {tracker.total_unique}")


if __name__ == "__main__":
    main()