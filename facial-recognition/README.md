# facial-recognition

[![EN](https://img.shields.io/badge/lang-EN-blue)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red)](README.zh.md)

This folder contains small example scripts that use the `face_recognition` library (which wraps dlib) together with OpenCV for drawing and displaying results.

Files

- `face_finder.py` — find a target person in a group photo by comparing face embeddings and draw a box around the match.
- `find_faces_in_picture.py` — (utility) detect and list faces in a single image. See the script for details.
- `images/` — sample images used by the scripts (put `target.jpg`, `group.jpg`, etc. here).
- `requirements.txt` — pinned Python dependencies (you can install via `pip install -r requirements.txt`).

Quick overview

- The scripts use `face_recognition` for face detection and embedding extraction:
  1. Detection: by default `face_recognition.face_locations(...)` uses a HOG-based detector (fast on CPU).
  2. Embeddings: `face_recognition.face_encodings(...)` uses dlib's face recognition model (a ResNet-style CNN) to produce a numeric embedding for each face. These embeddings are compared with `face_recognition.compare_faces(...)`.

Therefore the pipeline mixes both traditional (HOG) detection and deep models (CNN embeddings). You can explicitly force the CNN detector by passing `model="cnn"` to `face_locations`.

Installation (macOS / zsh)

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

If pip fails installing `dlib` (a dependency of `face_recognition`) on macOS, install the build prerequisites first:

```bash
# macOS (Homebrew)
brew install cmake pkg-config
pip install dlib
pip install face_recognition
```

Notes on prerequisites

- On some platforms installing `dlib` from source requires CMake and a C++ toolchain. Prebuilt wheels exist for many common Python versions — prefer pip when possible.
- If you plan to use GPU-accelerated detection/processing, you will need a different build of dlib or a different stack; the basic `face_recognition` usage works on CPU.

Usage: `face_finder.py`

1. Put your images in the `images/` folder (default names used by the script are `target.jpg` and `group.jpg`).
2. Run:

```bash
python face_finder.py
```

What the script does

- Loads the "target" face image and computes its face embedding (a numeric vector produced by dlib's face recognition CNN).
- Loads the group image, detects face locations, computes embeddings for each detected face, and compares each embedding to the target embedding using `compare_faces` (default tolerance = 0.6).
- If a match is found it draws a rectangle and the label "Target" and shows the result using OpenCV's `imshow`.

Tips and common tweaks

- Use the CNN detector (more accurate but slower) for `face_locations`:

```py
face_locations = face_recognition.face_locations(image, model="cnn")
```

- Headless / server usage: replace `cv2.imshow(...)` + `waitKey` with `cv2.imwrite('out.jpg', image_bgr)` to save the result instead of opening a window.
- Make the script robust to missing faces: check `face_recognition.face_encodings(...)` returns non-empty lists before indexing `[0]`.
- To mark all matches (not only the first), remove the `break` after a match so every matching face is labeled.
- Adjust the tolerance in `compare_faces(known_encodings, face_encoding, tolerance=0.6)` to make matching stricter/looser.

Security & privacy

- These scripts are for local experimentation and assume you have permission to process the images. Face recognition can be sensitive — comply with local laws and privacy best practices.

Acknowledgements

- The project uses the `face_recognition` library by Adam Geitgey (see the project page: [face_recognition on GitHub](https://github.com/ageitgey/face_recognition)) and `dlib` for the underlying models.

License

- This folder contains example/demo code. Reuse under your project's license; the README and examples are provided as-is.

If you want, I can:

- Make `face_finder.py` more robust (add CLI args to choose detector or output path, and add checks for missing encodings). 
- Provide a more detailed macOS installation script for `dlib` if you run into build errors.

