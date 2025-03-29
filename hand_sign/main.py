import cv2
import math
import mediapipe as mp
import threading
from pedalboard import Gain, Pedalboard, PitchShift, Reverb
from pedalboard.io import AudioFile, AudioStream


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
ImageFormat = mp.ImageFormat  # For creating an MP image.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

replacement_image = cv2.imread("lebron.png", cv2.IMREAD_UNCHANGED)
left_gesture = "None"
right_gesture = "None"
audio_params = {"gain": 0, "reverb": 0, "pitchshift": 0}


# Create a gesture recognizer instance with the live stream mode:
def recognise_gesture(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    global left_gesture, right_gesture
    has_left_changed, has_right_changed = False, False
    for handed, gesture, landmark in zip(
        result.handedness, result.gestures, result.hand_landmarks
    ):
        hand, gest = handed[0].category_name, gesture[0].category_name
        score = gesture[0].score
        if hand == "Left":
            left_gesture = gest
            has_left_changed = True
        if hand == "Right":
            right_gesture = gest
            has_right_changed = True
    if not has_left_changed:
        left_gesture = "None"

    if not has_right_changed:
        right_gesture = "None"


def draw_lines(results):
    left_mid = None
    right_mid = None
    # Hand landmark detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the thumb tip and index tip.
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Get the wrist landmark (center of the palm)
            wrist = hand_landmarks.landmark[0]

            # Convert normalized coordinates to pixel values.
            x1, y1 = int(thumb_tip.x * frame.shape[1]), int(
                thumb_tip.y * frame.shape[0]
            )
            x2, y2 = int(index_tip.x * frame.shape[1]), int(
                index_tip.y * frame.shape[0]
            )
            wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(
                wrist.y * frame.shape[0]
            )

            # Compute the midpoint between thumb tip and index tip.
            mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Save midpoint based on hand side.
            if wrist_x < frame.shape[1] // 2:
                left_mid = mid_point
                hand_color = (255, 0, 0)  # Blue for left hand
            else:
                right_mid = mid_point
                hand_color = (0, 255, 0)  # Green for right hand

            # Draw a line from the thumb tip to the index tip.
            cv2.line(frame, (x1, y1), (x2, y2), hand_color, 2)

            # Calculate the Euclidean distance between thumb tip and index tip.
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Update audio parameter based on hand side.
            if wrist_x < frame.shape[1] // 2:
                volume = (
                    (2 * (max(30, min(500, distance)) - 30) / (500 - 30)) - 1
                ) * 10
                audio_params["gain"] = volume
                cv2.putText(
                    frame,
                    f"Gain: {volume:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hand_color,
                    2,
                )
            else:
                pitchshift = (
                    (2 * (max(30, min(500, distance)) - 30) / (500 - 30)) - 1
                ) * 5
                audio_params["pitchshift"] = pitchshift
                cv2.putText(
                    frame,
                    f"Pitch: {pitchshift:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    hand_color,
                    2,
                )

    # Draw a line connecting the midpoints of the left and right hands if available.
    if left_mid is not None and right_mid is not None:
        cv2.line(frame, left_mid, right_mid, (0, 0, 255), 2)
        hand_distance = math.sqrt(
            (right_mid[0] - left_mid[0]) ** 2 + (right_mid[1] - left_mid[1]) ** 2
        )
        mid_between = (
            (left_mid[0] + right_mid[0]) // 2,
            (left_mid[1] + right_mid[1]) // 2,
        )
        reverb = min(1, max(0, (max(20, min(1500, hand_distance)) - 20) / (1500 - 20)))
        audio_params["reverb"] = reverb
        cv2.putText(
            frame,
            f"Reverb: {reverb:.2f}",
            mid_between,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )


def audio_stream():
    output_device = AudioStream.default_output_device_name  # Specify the output device

    with AudioStream(output_device_name=output_device) as stream:
        with AudioFile("audio.mp3") as f:
            while f.tell() < f.frames:
                board = Pedalboard(
                    [
                        PitchShift(semitones=audio_params["pitchshift"]),
                        Gain(gain_db=audio_params["gain"]),
                        Reverb(room_size=audio_params["reverb"]),
                    ]
                )
                chunk = f.read(f.samplerate / 10)
                effected = board(chunk, f.samplerate / 10, reset=True)
                while left_gesture == "Open_Palm" or right_gesture == "Open_Palm":
                    continue
                stream.write(effected, f.samplerate)


# Start the audio stream in a background thread.
audio_thread = threading.Thread(target=audio_stream, daemon=True)
audio_thread.start()


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=recognise_gesture,
    num_hands=2,
)
with GestureRecognizer.create_from_options(options) as recognizer:

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5
    ) as face_detection:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for natural interaction and convert color.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    # Increase size of the replacement image by a scale factor.
                    scale = 4  # adjust factor as needed
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    # Center the enlarged region on the detected face.
                    center_x = x + w // 2
                    center_y = y + h // 2
                    new_x = max(0, center_x - new_w // 2)
                    new_y = max(0, center_y - new_h // 2)
                    # Ensure the new dimensions fit within the frame boundaries.
                    if new_x + new_w > iw:
                        new_w = iw - new_x
                    if new_y + new_h > ih:
                        new_h = ih - new_y

                    # Resize the replacement image to the new dimensions.
                    replacement_resized = cv2.resize(replacement_image, (new_w, new_h))

                    # If the replacement image has an alpha channel, blend using it.
                    if replacement_resized.shape[2] == 4:
                        alpha_s = replacement_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            frame[new_y : new_y + new_h, new_x : new_x + new_w, c] = (
                                alpha_s * replacement_resized[:, :, c]
                                + alpha_l
                                * frame[new_y : new_y + new_h, new_x : new_x + new_w, c]
                            )
                    else:
                        frame[new_y : new_y + new_h, new_x : new_x + new_w] = (
                            replacement_resized
                        )

            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb)
            # Use current time in ms as timestamp.
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Process the frame asynchronously.
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Hand landmark detection
            results = hands.process(rgb)
            draw_lines(results)

            # # Display the webcam frame.
            cv2.putText(
                frame,
                f"Left Hand: {left_gesture}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Right Hand: {right_gesture}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


def update_audio_params(audio_params, left_dist, right_dist, left_to_right_dst):
    if left_dist > -1:
        volume = max(-1, min(1, 2 * (left_dist - 30) / (500 - 30)))
        audio_params["gain"] = volume * 15

    if right_dist > -1:
        pitchshift = max(-1, min(1, 2 * (right_dist - 30) / (500 - 30)))
        audio_params["pitchshift"] = pitchshift * 10

    if left_to_right_dst > -1:
        reverb = -10 + (right_dist - 30) / (500 - 30) * 20
        audio_params["reverb"] = reverb
