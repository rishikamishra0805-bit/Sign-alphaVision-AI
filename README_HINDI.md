# SignVision AI — Windows 10 Project

## इस ZIP में क्या है?
- `app.py` — Tkinter desktop application
- `requirements.txt` — required Python libraries
- `setup_windows.bat` — first-time installation
- `run_windows.bat` — application start
- `data/` — captured hand-landmark samples
- `models/` — trained classifier file

## जरूरी सच्चाई
यह एक **train-your-own-sign prototype** है। ZIP में पहले से trained ISL recognition weights शामिल नहीं हैं। इसलिए पहली बार A–Z या words की सही prediction अपने-आप नहीं होगी। पहले webcam से labels के samples capture करके model train करना होगा।

यह version एक समय में एक हाथ के static hand-pose landmarks (21 points × x,y,z = 63 features) classify करता है। Dynamic/moving signs, two-hand signs, facial expressions और continuous natural ISL grammar अभी supported नहीं हैं। Word labels (HELLO, THANK YOU आदि) को भी तभी classify किया जा सकता है जब उनके सही static samples collect करके train किया जाए। केवल label list होना trained word recognition नहीं है।

## Windows 10 setup
1. Python **3.11 64-bit** install करें: https://www.python.org/downloads/
2. Installation में `Add Python to PATH` tick करें।
3. ZIP extract करें, उदाहरण: `D:\SignVisionAI`
4. `setup_windows.bat` पर double-click करें। Internet connection जरूरी है क्योंकि pip packages install होंगे।
5. Installation पूरी होने पर `run_windows.bat` चलाएँ।
6. Webcam permission allow करें।

## Model training कैसे करें?
1. Webcam के सामने एक sign दिखाएँ।
2. Dropdown में उसका label चुनें (जैसे A)।
3. `Capture 25 frames` दबाएँ और हाथ को स्थिर रखें।
4. अलग-अलग दूरी/रोशनी में उसी label के लिए कई बार capture करें।
5. कम-से-कम दो labels के samples लेने के बाद `Train / Update Model` दबाएँ।
6. बाकी alphabets और word labels के लिए यही प्रक्रिया दोहराएँ।
7. अलग-अलग लोगों और lighting conditions से data लेने पर generalisation बेहतर हो सकती है।

**महत्वपूर्ण:** सिर्फ 25 लगातार frames लगभग एक ही pose के samples हैं; वे 25 independent examples नहीं हैं। बेहतर results के लिए हर label के लिए कई अलग-अलग sessions/poses collect करें। Training के बाद unfamiliar signs पर test करें।

## Presentation में क्या कहें?
“SignVision AI is a computer-vision prototype for Indian Sign Language hand-pose classification. It uses OpenCV for webcam input, MediaPipe for extracting 21 hand landmarks, and a Random Forest classifier for label prediction. The application supports alphabet and user-defined word-label training, confidence scores, a sign-label dictionary, and sentence formation. The current prototype focuses on static one-hand gestures and requires a labelled dataset for reliable predictions.”

## Limitations
- No pretrained ISL model is bundled.
- The dictionary lists labels and meanings; it is not an authoritative visual ISL sign chart.
- Static one-hand pose classification only.
- It is not a complete ISL translator.
- Confidence scores are model probabilities, not a guarantee of real-world accuracy.

## Troubleshooting
- Camera not found: close Zoom/Teams/Camera app and try again.
- `py -3.11` not found: install Python 3.11 64-bit.
- MediaPipe installation error: confirm Python 3.11 and 64-bit Windows.
- Low confidence: capture more varied samples and improve lighting.
