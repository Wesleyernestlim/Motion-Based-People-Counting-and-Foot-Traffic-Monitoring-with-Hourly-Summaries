# Motion-Based People Counting and Foot-Traffic Monitoring with Hourly Summaries

**Course:** CSC 126 Computer Vision, Midterm Project
**School:** Caraga State University, College of Computing and Information Sciences

**Members:** 
   `Wesley Ernest Lim`
   `Lyndy Fe B. Salise`

## Project Description

This project is a computer vision application that counts people who are moving in front of a camera. It works with a live webcam or a recorded video file. The system shows each moving person with a box and an ID number, reports how many people are moving at the moment, and keeps a running total of the different people counted. Every motion event is saved with its date and time to a CSV file, and an hourly summary file is produced when the program stops.

Only people who are **moving** are counted. A person who stands or sits still is drawn in gray and is not included in the count, so the project measures foot traffic and not the total number of people present in a room.

### Real-World Use

Stores, schools, offices, and other public places need to know how many people pass through and at what times. Counting by hand is slow and error-prone, and simple motion sensors cannot tell people from other moving objects. This project provides a low-cost, camera-based alternative.

## Techniques Used

`(a) Image Operations` = Resize, grayscale, histogram equalization | Prepares each frame so detection is faster and works better in uneven lighting 

`(b) Image Segmentation` = Background subtraction (MOG2), Gaussian blur, thresholding, morphological cleanup | Separates moving parts of the scene from the still background (motion mask) |

`(c) Haar Cascade Classifier` = `haarcascade_frontalface_default.xml` | Finds front-facing faces in each frame |

`(e) Motion and Object Tracking` = Centroid tracker with motion events | Gives each person an ID that stays the same between frames and reports when a person is counted, stops moving, or starts moving again |

## How It Works

1. Each frame is read from the webcam or video and resized.
2. The frame is converted to grayscale and its contrast is improved.
3. A motion mask is created using background subtraction. White areas are moving.
4. The Haar Cascade finds faces in the frame.
5. The tracker links each face to the closest face from the previous frame and gives it an ID.
6. A person is considered to be moving when enough of the area around their face is white in the motion mask.
7. A person is added to the unique total after being seen moving for several frames.
8. The annotated frame is displayed. Each time a person is counted, stops moving, or starts moving again, a row with the date and time is added to `people_log.csv`.
9. When the program stops, `hourly_summary.csv` is created from the log.

## Project Structure

```
pipcount/
├── main.py                # the application
├── requirements.txt       # dependencies
├── README.md              # this file
├── samples/
│   └── sample.mp4         # sample input video
├── screenshots/           # output screenshots (saved with the S key)
├── people_log.csv         # created when the program runs
└── hourly_summary.csv     # created when the program stops
```

## Requirements

- Python 3.8 or newer
- A webcam (for live mode) or a video file
- The packages listed in `requirements.txt` (OpenCV)

## Setup

1. Download or clone this repository, then open a terminal in the project folder.
2. Create a virtual environment: *(Optional but recommended)* 

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   On macOS or Linux, use `source venv/bin/activate` for the second command.

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

## How to Run

**Using a webcam (default):**

```
python main.py
```

**Using a video file:**

1. Put the video in the `samples/` folder.
2. Open `main.py` and change these settings near the top:

   ```python
   USE_WEBCAM = False
   VIDEO_PATH = "samples/sample.mp4"
   ```

3. Run `python main.py`.

When the program starts, the screen shows "Calibrating background..." for a few moments while it learns the scene. Keep the camera fixed during this time.

### Controls

| Key | Action |
|---|---|
| `Q` | Stop the program and create the hourly summary |
| `S` | Save a screenshot to the `screenshots/` folder |

## Output

### On the screen

- **Green box with "ID n moving":** a person who is moving and is counted.
- **Gray box with "still":** a person who has stopped moving and is not in the moving count.
- **Moving people:** how many people are moving right now.
- **Unique people:** how many different people have been counted since the program started.

### Saved files

**`people_log.csv`:** one row each time a motion event happens. Nothing is written during quiet periods.

| Column | Meaning |
|---|---|
| `timestamp` | Date and time of the event |
| `person_id` | ID of the person involved |
| `event` | `counted` (person added to the unique total), `stopped_moving`, or `started_moving` |
| `moving_now` | Number of people moving right after the event |
| `unique_total` | Running total of different people counted |

Example:

```
timestamp,person_id,event,moving_now,unique_total
2026-11-14 14:02:11,0,counted,1,1
2026-11-14 14:02:19,0,stopped_moving,0,1
2026-11-14 14:02:31,0,started_moving,1,1
```

**`hourly_summary.csv`:** one row per hour, rebuilt from the whole log each time the program is stopped normally. Hours with no events do not appear.

| Column | Meaning |
|---|---|
| `hour` | The hour being summarized |
| `new_people_counted` | Number of people added to the unique total during the hour |
| `started_moving_events` | Number of times a still person started moving |
| `stopped_moving_events` | Number of times a moving person stopped |
| `peak_moving_count` | Highest number of people moving at the time of an event in that hour |

> Close the CSV files before running the program. If a file is open in Excel, it may be locked and the program will not be able to save.

## Settings

These can be changed at the top of `main.py`.

| Setting | Default | Description |
|---|---|---|
| `USE_WEBCAM` | `True` | `True` for the webcam, `False` for a video file |
| `VIDEO_PATH` | `samples/sample.mp4` | Video file used when `USE_WEBCAM` is `False` |
| `FRAME_WIDTH` | `900` | Width the frame is resized to |
| `MIN_NEIGHBORS` | `5` | Higher values give fewer wrong face detections |
| `MIN_FACE_SIZE` | `(30, 30)` | Smallest face the detector will accept |
| `MAX_DISTANCE` | `60` | Farthest a person can move between frames and keep the same ID (pixels) |
| `MAX_MISSED` | `20` | Frames a person can be lost before their ID is dropped |
| `MIN_MOVING_HITS` | `5` | Frames a person must be seen moving before being counted |
| `MOTION_RATIO` | `0.05` | Share of the area around a face that must be moving. Raise it if still people are counted; lower it if walking people are missed |
| `MOTION_GRACE` | `15` | Frames a person stays "moving" after their last motion |
| `WARMUP_FRAMES` | `30` | Frames used to learn the background before counting starts |
| `USE_CSV` | `True` | `False` to run without saving files |
| `CSV_PATH` | `people_log.csv` | Name of the event log file |
| `HOURLY_PATH` | `hourly_summary.csv` | Name of the hourly summary file |

## Limitations

- The Haar Cascade finds only front-facing faces. People seen from above, from behind, or from the side are often missed, so the system works best at entrances, doorways, and corridors where people walk toward the camera.
- The detector can mark objects that look like faces. If these are moving, they may be counted as people.
- Fast movement can blur a person, which may cause the detector to lose them for a few frames. When this happens, a "stopped_moving" event may be logged even though the person is still moving.
- The count of unique people is approximate. Someone who leaves and returns, or who is hidden for a long time, may receive a new ID and be counted again.
- A still person standing next to a moving person may be counted as moving.
- The camera must stay fixed. Shaking or sudden changes in lighting can be mistaken for motion.
- The peak moving count in the hourly summary is approximate, because the number of people moving is saved only when an event happens.
- When a video file is used, the log follows the computer's clock and not the video's own timeline.

## Sample Input and Screenshots

- Sample input video: `samples/sample.mp4`
- Output screenshots: see the `screenshots/` folder.

screenshots/4c283e04-5ae0-4c2f-b030-ab7b2b084789.jpg
`![Moving person detected]`

## Possible Improvements

- Use a full-body or pedestrian detector, or count moving shapes in the motion mask, so the system works with cameras placed above or to the side of a scene.
- Add a column with the video's own time so the log matches recorded footage.
- Count people crossing a line, such as a doorway, to get separate numbers for entering and leaving.
- Show the hourly summary as a chart, and store the records in a database for long-term use.

## Acknowledgments

Built with [Python](https://www.python.org/) and [OpenCV](https://opencv.org/) for the CSC 126 Computer Vision course.
