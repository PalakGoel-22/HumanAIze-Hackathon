from flask import Flask, request, jsonify, render_template
import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import moviepy.editor as mpe
from faster_whisper import WhisperModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the Whisper model
model_size = "large-v2"
model1 = WhisperModel(model_size, device="cpu", compute_type="default")

# Load the GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Function to generate descriptive text using GPT-2 with a specific prompt
def generate_descriptive_text(text, max_length=100):
    prompt = f"Write four lines about the given text: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    descriptive_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return descriptive_text

# Function to split text into bullet points
def split_text_into_segments(text):
    bullet_points = text.split('. ')
    segments = [f"• {point.strip()}." for point in bullet_points if point]
    return segments

# Function to generate images from text using LimeWire API
def generate_images_from_text(text, num_images=1):
    url = "https://api.limewire.com/api/image/generation"
    images = []

    for i in range(num_images):
        payload = {
            "prompt": f"{text} variation {i}",
            "aspect_ratio": "1:1"
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Version": "v1",
            "Accept": "application/json",
            "Authorization": "Bearer lmwr_sk_RU3WxU9n1C_NH0TEtcBJnc555jyQVsEck8okTnGcRR7gHR0K"
        }

        response = requests.post(url, json=payload, headers=headers)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.content}")

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            print("Response Text:", response.text)
            continue

        if 'data' in data and len(data['data']) > 0:
            image_url = data['data'][0]['asset_url']
            image_response = requests.get(image_url)
            print(f"Image URL: {image_url}")
            if image_response.status_code == 200:
                image = Image.open(BytesIO(image_response.content))
                images.append(image)
            else:
                print("Failed to download image")
        else:
            print("No image data found in the response")

    return images

# Function to add text to a video frame
def add_text_to_frame(frame, text, position=(50, 50), font_scale=0.5, font_color=(255, 255, 255), font_thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    frame_h, frame_w, _ = frame.shape
    x, y = position
    # Ensure text is within frame bounds
    if x + text_w > frame_w:
        x = frame_w - text_w - 10
    if y - text_h < 0:
        y = text_h + 10
    # Add text background
    cv2.rectangle(frame, (x, y - text_h), (x + text_w, y), (0, 0, 0), -1)
    # Add text
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
    return frame

# Function to create video from images with captions
def create_video_from_images(images, audio_path, output_path, captions, fps=1):
    if len(images) == 0:
        # Create a simple black image with the caption text if no images were retrieved
        width, height = 720, 720
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        draw.text((50, 50), captions, fill=(255, 255, 255), font=font)
        images = [image]

    height, width = images[0].size
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i, image in enumerate(images):
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame = add_text_to_frame(frame, captions[i])
        video.write(frame)

    video.release()

    # Add audio to video if audio_path is provided
    if audio_path:
        video_clip = mpe.VideoFileClip(output_path)
        audio_clip = mpe.AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec="libx264")

# Main function to process audio and create video
def process_audio_to_video(audio_path, output_video_path):
    # Step 1: Transcribe audio to text
    segments, _ = model1.transcribe(audio_path)
    transcription = " ".join([segment.text for segment in segments])
    print("Transcription:", transcription)

    # Step 2: Generate descriptive text from the transcription
    descriptive_text = generate_descriptive_text(transcription)
    print("Descriptive Text:", descriptive_text)

    # Step 3: Split descriptive text into segments
    text_segments = split_text_into_segments(descriptive_text)
    print("Text Segments:", text_segments)

    # Step 4: Generate images from each text segment
    images = []
    for segment in text_segments:
        images.extend(generate_images_from_text(segment, num_images=1))

    # Step 5: Create video from images with captions
    create_video_from_images(images, audio_path, output_video_path, text_segments)

def generate_images_and_video_from_text(text, output_video_path):
    # Step 1: Generate descriptive text from the given text
    descriptive_text = generate_descriptive_text(text)
    print("Descriptive Text:", descriptive_text)

    # Step 2: Split descriptive text into segments
    text_segments = split_text_into_segments(descriptive_text)
    print("Text Segments:", text_segments)

    # Step 3: Generate images from each text segment
    images = []
    for segment in text_segments:
        images.extend(generate_images_from_text(segment, num_images=1))

    # Step 4: Create video from images with captions
    create_video_from_images(images, None, output_video_path, text_segments)

@app.route('/')
def index():
    return render_template('Template.html')

@app.route('/upload', methods=['POST'])
def upload():
    output_video_path = 'static/output_video.mp4'
    
    if 'audio' in request.files:
        # Handle audio file upload
        audio_file = request.files['audio']
        audio_path = 'uploaded_audio.wav'
        audio_file.save(audio_path)
        process_audio_to_video(audio_path, output_video_path)
    elif 'text' in request.form:
        # Handle text input
        text = request.form['text']
        generate_images_and_video_from_text(text, output_video_path)
    else:
        return jsonify({'error': 'No audio file or text provided'}), 400

    return jsonify({'status': 'Processing completed', 'video_url': output_video_path})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5020, debug=True)
