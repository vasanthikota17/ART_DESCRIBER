from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from googletrans import Translator
from fer import FER
from PIL import Image
import cv2
import asyncio

app = Flask(__name__)

# Path for uploaded files
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to detect human emotions
def detect_emotions(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unsupported format.")
    detector = FER(mtcnn=True)
    emotion_data = detector.top_emotion(image)
    if emotion_data is not None:
        emotion, confidence = emotion_data
        return f"With a deep sense of {emotion}, the expression radiates warmth and connection."
    return "The expression is serene, yet no clear emotion stands out."

# Function to provide color meaning based on emotional context
def get_color_meaning(color):
    color_meanings = {
        "red": "The color red evokes passion and warmth, signaling love, strength, and energy.",
        "blue": "Blue brings a sense of tranquility, calm, and deep trust.",
        "yellow": "Yellow bursts with happiness, optimism, and creativity.",
        "green": "Green, the color of life and renewal, symbolizes balance and harmony.",
        "black": "Black represents sophistication and depth.",
        "white": "White symbolizes purity, innocence, and clarity.",
        "purple": "Purple conveys luxury, creativity, and spirituality.",
        "orange": "Orange radiates warmth and enthusiasm."
    }
    return color_meanings.get(color.lower(), "This color brings its own unique energy.")

# Function to describe artwork and detect emotions
def describe_and_detect(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract dominant color and provide meaning
    dominant_color = extract_dominant_color(image)
    color_meaning = get_color_meaning(dominant_color)
    description += f" The dominant color here creates a distinct mood. {color_meaning}."
    
    # Detect emotions
    emotion_result = detect_emotions(image_path)
    description += f" {emotion_result}"
    
    return description

# Function to extract the dominant color
def extract_dominant_color(image):
    image = image.resize((50, 50))
    pixels = list(image.getdata())
    avg_color = tuple(sum(c) // len(c) for c in zip(*pixels))
    if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
        return "red"
    elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
        return "green"
    elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
        return "blue"
    else:
        return "unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        description = describe_and_detect(file_path)
        
        language = request.form.get('language')
        translator = Translator()
        
        # Await the coroutine to get the translated description
        translated_description = await asyncio.to_thread(translator.translate, description, src='en', dest=language)
        
        # Ensure that the 'static' directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        
        # Now translated_description.text is available after the await
        tts = gTTS(text=translated_description.text, lang=language, slow=False)
        audio_file = f"static/{language}_description.mp3"
        tts.save(audio_file)
        
        return jsonify({
            'description': translated_description.text,
            'audio': audio_file
        })
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
