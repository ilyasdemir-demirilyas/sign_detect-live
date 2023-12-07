pip install opencv-python==4.8.1.78

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model

# Initialize MediaPipe Holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained model
model = load_model('ısa_eller.h5')

# Streamlit UI setup
st.title('Live Sign Language Detection Using MediaPipe with Streamlit GUI')
st.sidebar.title('Sign Language Detection')
st.sidebar.subheader('-Parameter')

# Actions and colors
actions = ['i', 'l', 'y', 'a', 's']
colors = [(245, 117, 16), (117, 245, 16), (245, 17, 126), (17, 25, 16), (25, 117, 216)]

# Function to resize the image
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Function for MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# Function to extract keypoints
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

# Function to run hand detection on live video
def run_hand_detection():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sequence, sentence, predictions = [], [], []
    threshold = 0.7
    previous_word = None  # Keep track of the previous detected word
    fps_text = "FPS: 0"

    # Video capture from webcam
    vid = cv2.VideoCapture(0)

    stframe = st.empty()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, img = vid.read()
            img = cv2.flip(img, 1)

            # Mediapipe detection
            img, results = mediapipe_detection(img, holistic)

            # Draw landmarks
            draw_styled_landmarks(img, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-120:]

            if len(sequence) == 120:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    current_word = actions[np.argmax(res)]

                    # Check if the current word is different from the previous one
                    if current_word != previous_word:
                        sentence.append(current_word)
                        previous_word = current_word

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Visualize probabilities
                    img = prob_viz(res, actions, img, colors)

                    # Viz probability values
                    prob_str = " ".join([f"{actions[i]}: {res[i]:.2f}" for i in range(len(actions))])
                    cv2.putText(img, prob_str, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Calculate and display FPS
                fps = int(vid.get(cv2.CAP_PROP_FPS))
                fps_text = f"FPS: {fps}"

            cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(img, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display FPS in the top-right corner
            cv2.putText(img, fps_text, (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Resize the frame
            frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)

            # Display the frame in Streamlit
            stframe.image(frame, channels='BGR', use_column_width=True)

    # Release the webcam
    vid.release()


# Main application
if st.button('Start Detection'):
    run_hand_detection()
