import cv2  # imports opencv for image and video processing
import os  # imports os module for system commands
from inference_sdk import InferenceConfiguration, InferenceHTTPClient  # imports inference configuration and http client for model predictions

# function to control a device via a system command
def control_device(command):
    os.system(f"kasa --host 192.168.1.145 {command}")  # sends a system command to control the device through the kasa smart plug

# function to capture frames from the camera and analyze for object detection
def capture_and_analyze_frame():
    # initialize video capture on the specified device
    cap = cv2.VideoCapture('/dev/video0')  # opens the video capture on the specified device (/dev/video0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # sets the width of the captured frames to 1280 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # sets the height of the captured frames to 720 pixels

    frame_count = 0  # initializes the frame counter
    cat_detected_frames = 0  # initializes a counter for frames with cat detection

    while True:
        # clear the buffer by grabbing a few frames and discarding them
        for _ in range(5):
            cap.grab()  # grabs frames from the camera but discards them to clear the buffer

        success, frame = cap.read()  # captures the latest frame
        frame_count += 1  # increments the frame counter

        if success:  # checks if the frame was captured successfully
            # save the captured frame
            image_path = 'captured_frame.jpg'  # defines the path to save the captured image
            cv2.imwrite(image_path, frame)  # saves the captured frame as an image

            print(f"Status: Analysing Frame ({frame_count})")  # prints the current status with frame count

            # analyze the captured frame
            predictions = analyze_frame(image_path)  # calls the function to analyze the saved frame
            cat_detected = False  # flag to indicate whether a cat is detected

            if predictions:  # checks if any predictions were made
                for prediction in predictions['predictions']:  # iterates over each prediction
                    if prediction['class'] == 'cat':  # checks if the detected class is 'cat'
                        cat_detected = True  # sets the flag to true if a cat is detected
                        print(f"Detected a cat with {prediction['confidence']*100:.2f}% confidence at (x={prediction['x']}, y={prediction['y']})")  # prints details about the detection
                        break  # exits the loop after detecting a cat
            
            print(f"Detected Cat: {cat_detected}")  # prints whether a cat was detected

            if cat_detected:  # if a cat is detected
                if cat_detected_frames == 0:  # checks if the detection is new (i.e., not previously detected)
                    control_device("on")  # turns on the device if a cat is detected
                cat_detected_frames = 4  # resets the frame timer to 4 after detecting a cat
            else:  # if no cat is detected
                if cat_detected_frames > 0:  # checks if there were recent frames with a cat detected
                    cat_detected_frames -= 1  # decrements the frame countdown
                    print(f"Frame countdown: {cat_detected_frames}")  # prints the countdown
                if cat_detected_frames == 0:  # if the countdown reaches zero
                    control_device("off")  # turns off the device

        else:  # if the frame capture failed
            print(f"Failed to capture frame {frame_count}")  # prints an error message

    cap.release()  # releases the video capture object when done

# function to send the captured frame to the inference model for analysis
def analyze_frame(image_path):
    MODEL_ID = "coco/24"  # specifies the model id to use for inference (e.g., a coco dataset model)
    config = InferenceConfiguration(confidence_threshold=0.5, iou_threshold=0.5)  # creates the configuration for inference with confidence and iou thresholds

    client = InferenceHTTPClient(
        api_url="http://localhost:9001",  # defines the url for the inference api
        api_key="VfTdKbx8wnZGeThn9vnT",  # provides the api key for authentication
    )
    client.configure(config)  # configures the client with the specified configuration
    client.select_model(MODEL_ID)  # selects the model to use for inference

    predictions = client.infer(image_path)  # sends the image to the inference api and gets predictions
    return predictions  # returns the predictions

if __name__ == '__main__':
    capture_and_analyze_frame()  # calls the function to capture and analyze frames
