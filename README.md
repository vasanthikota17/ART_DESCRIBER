**Project Description: **
Image to Text and Speech Application
This project focuses on developing a web-based application that seamlessly converts images containing text into both textual and speech outputs. It leverages Optical Character Recognition (OCR) and Text-to-Speech (TTS) technologies to provide a comprehensive solution for extracting and vocalizing textual information from images.

Key Features:**
**Image Upload:** Users can upload image files containing text.
**Language Support: **Users can select the desired language for text recognition and speech synthesis.
**Text Extraction:** Extracts text from the uploaded image using OCR technology.
**Speech Output:** Converts the extracted text into natural-sounding speech using a TTS engine.
**User-Friendly Interface:** The interface is designed to be intuitive, with a clean layout and aesthetic background.

**Objectives:**
Enable visually impaired users to access text content from images through audio output.
Provide a tool for quick and easy text extraction for educational and professional purposes.
Support multiple languages for a broader user base.

**Technologies Used:**
Frontend: HTML, CSS, JavaScript, Bootstrap for a responsive and interactive UI.
Backend: Node.js or Flask (depending on the implementation) to process user inputs.
OCR Engine: Tesseract.js or similar OCR libraries for text extraction.
TTS Engine: Google Text-to-Speech (gTTS) or any preferred TTS library for generating speech output.

**Use Cases:**
Assisting visually impaired users in understanding text from images.
Converting image-based documents into text for editing or storage.
Providing language-learning aids by reading out text in the chosen language.
This application bridges the gap between visual and auditory mediums, making it accessible and useful for a wide range of users.

**Features and Workflow:**

**Image Upload:**
Users upload an image via the /upload route.
The application ensures only valid file types (.png, .jpg, .jpeg) are accepted.
**Image Captioning:**
The Salesforce/blip-image-captioning-base model is used to generate a descriptive caption for the uploaded image.
The BlipProcessor and BlipForConditionalGeneration classes process the image and generate text.
**Emotion Detection:**
The FER library detects emotions from the image. If an emotion is identified, a human-like sentiment description is included in the caption.
**Dominant Color Detection:**
The application's extract_dominant_color function resizes the image and analyzes pixel data to determine the dominant color (e.g., red, green, or blue).
The application assigns symbolic meanings to colors and includes this information in the caption.
**Text Translation:**
Users can specify a target language. The Google Translator API translates the image caption into the selected language.
**Text-to-Speech (TTS):**
The translated caption is converted into speech using the gTTS library, and an audio file is saved in the static/ directory.
The audio file link is sent in the JSON response.


# Static Folder
Create a file named README.md inside the static folder with this content
This folder is used to store static files like audio files, images, CSS, or JavaScript that are served directly to users.

- Audio files generated by the app (e.g., `description.mp3`) are stored here.
- Ensure this folder exists for the application to run correctly.
  
# Uploads Folder
Create a file named README.md inside the uploads folder with this content
This folder is used to temporarily store files uploaded by users (e.g., images for processing).

- User-uploaded files are saved here before processing.
- Files may be cleared automatically after processing

Here's the structure of the folder and files for project
project-root/
│
├── app.py                     # Main Python file containing your Flask application logic
├── requirements.txt           # List of all Python dependencies for the project
├── templates/                 # Folder for HTML templates
│   └── index.html             # Main HTML template for the app
├── static/                    # Folder for static files (CSS, JavaScript, audio files, etc.)
│   ├── README.md              # Placeholder file to ensure the folder is tracked in GitHub
│   └── (Generated audio files will be stored here)
├── uploads/                   # Folder for user-uploaded files
│   ├── README.md              # Placeholder file to ensure the folder is tracked in GitHub
│   └── (Uploaded image files will be temporarily stored here)
├── README.md                  # Documentation file for your project
└── venv/                      # Virtual environment folder (not uploaded to GitHub)
