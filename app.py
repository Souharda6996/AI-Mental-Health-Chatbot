from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import json
import os
from transformers import pipeline
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Initialize emotion detection model
try:
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
except:
    emotion_classifier = None
    print("Warning: Emotion classifier not loaded. Install transformers and torch.")

# Initialize sentiment analysis
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except:
    sentiment_analyzer = None
    print("Warning: Sentiment analyzer not loaded.")

# Data storage files
MOOD_DATA_FILE = 'mood_data.json'
CHAT_HISTORY_FILE = 'chat_history.json'

# Helper functions
def load_json_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def save_json_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def detect_emotion(text):
    """Detect emotion from text using transformer model"""
    if emotion_classifier is None:
        return "neutral"
    
    try:
        results = emotion_classifier(text)
        if results and len(results) > 0:
            # Get the emotion with highest score
            top_emotion = max(results[0], key=lambda x: x['score'])
            return top_emotion['label']
    except:
        pass
    return "neutral"

def analyze_sentiment(text):
    """Analyze sentiment of the text"""
    if sentiment_analyzer is None:
        return {"label": "NEUTRAL", "score": 0.5}
    
    try:
        result = sentiment_analyzer(text)[0]
        return result
    except:
        return {"label": "NEUTRAL", "score": 0.5}

def generate_response(user_message, emotion, sentiment):
    """Generate contextual response based on emotion and sentiment"""
    
    # Emotion-based responses
    emotion_responses = {
        "sadness": [
            "I'm sorry you're feeling this way. Remember, it's okay to feel sad sometimes. Would you like to talk about what's troubling you?",
            "I hear you. Sadness is a natural emotion. What's been weighing on your mind?",
            "I'm here to listen. Sometimes sharing what's bothering us can help lighten the load."
        ],
        "joy": [
            "That's wonderful! I'm so glad to hear you're feeling happy. What's bringing you joy today?",
            "Your positive energy is contagious! Tell me more about what's making you feel good.",
            "I love seeing you in good spirits! What's been going well for you?"
        ],
        "anger": [
            "I can sense you're upset. It's completely valid to feel angry. Would you like to talk about what's frustrating you?",
            "Anger is a natural response. Let's work through this together. What's triggering these feelings?",
            "I'm here to support you. Sometimes expressing anger in a safe space helps. What happened?"
        ],
        "fear": [
            "I understand you're feeling anxious or scared. You're not alone. What's concerning you right now?",
            "Fear can be overwhelming, but facing it together makes it more manageable. Want to share what's worrying you?",
            "It's brave of you to acknowledge your fear. Let's talk about what's making you feel this way."
        ],
        "surprise": [
            "That sounds unexpected! How are you processing this?",
            "Surprises can be intense. How are you feeling about this?",
            "I'm here to help you work through this unexpected situation."
        ],
        "love": [
            "That's beautiful! Love and connection are so important. Tell me more!",
            "It's wonderful to feel love and appreciation. What's inspiring these feelings?",
            "Love is a powerful emotion. I'm glad you're experiencing it!"
        ],
        "neutral": [
            "I'm here to listen. What's on your mind today?",
            "How can I support you today?",
            "Tell me what you'd like to talk about."
        ]
    }
    
    # Get responses for detected emotion
    responses = emotion_responses.get(emotion.lower(), emotion_responses["neutral"])
    
    # Select response based on sentiment intensity
    if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.8:
        # High negative sentiment - be more supportive
        supportive_additions = [
            " Remember, you're stronger than you think.",
            " I'm here for you, and things can get better.",
            " You don't have to go through this alone."
        ]
        base_response = np.random.choice(responses)
        return base_response + np.random.choice(supportive_additions)
    
    return np.random.choice(responses)

def get_daily_suggestion(emotion):
    """Provide daily wellness suggestions based on emotion"""
    suggestions = {
        "sadness": [
            "Try a 10-minute walk outside to boost your mood",
            "Listen to your favorite uplifting music",
            "Reach out to a friend or loved one",
            "Practice gratitude by writing down 3 things you're thankful for"
        ],
        "joy": [
            "Share your happiness with someone you care about",
            "Engage in an activity you love",
            "Take a moment to appreciate this positive feeling",
            "Use this positive energy to tackle a goal"
        ],
        "anger": [
            "Try deep breathing: breathe in for 4, hold for 4, out for 4",
            "Physical exercise can help release tension",
            "Write down your feelings in a journal",
            "Take a break and do something calming"
        ],
        "fear": [
            "Practice grounding: name 5 things you can see, 4 you can touch, 3 you can hear",
            "Try progressive muscle relaxation",
            "Talk to someone you trust about your concerns",
            "Focus on what you can control right now"
        ],
        "neutral": [
            "Stay hydrated and eat nutritious meals",
            "Get some gentle exercise today",
            "Practice mindfulness for 5-10 minutes",
            "Connect with nature, even if just looking outside"
        ]
    }
    
    emotion_key = emotion.lower() if emotion.lower() in suggestions else "neutral"
    return np.random.choice(suggestions[emotion_key])

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Detect emotion and sentiment
    emotion = detect_emotion(user_message)
    sentiment = analyze_sentiment(user_message)
    
    # Generate response
    bot_response = generate_response(user_message, emotion, sentiment)
    
    # Save chat history
    chat_history = load_json_file(CHAT_HISTORY_FILE)
    chat_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_message': user_message,
        'bot_response': bot_response,
        'emotion': emotion,
        'sentiment': sentiment,
        'user_id': session.get('user_id')
    }
    chat_history.append(chat_entry)
    save_json_file(CHAT_HISTORY_FILE, chat_history)
    
    return jsonify({
        'response': bot_response,
        'emotion': emotion,
        'sentiment': sentiment
    })

@app.route('/log-mood', methods=['POST'])
def log_mood():
    data = request.json
    mood = data.get('mood')
    notes = data.get('notes', '')
    
    if not mood:
        return jsonify({'error': 'No mood provided'}), 400
    
    # Load existing mood data
    mood_data = load_json_file(MOOD_DATA_FILE)
    
    # Add new mood entry
    mood_entry = {
        'timestamp': datetime.now().isoformat(),
        'mood': mood,
        'notes': notes,
        'user_id': session.get('user_id')
    }
    mood_data.append(mood_entry)
    save_json_file(MOOD_DATA_FILE, mood_data)
    
    # Get suggestion based on mood
    suggestion = get_daily_suggestion(mood)
    
    return jsonify({
        'success': True,
        'suggestion': suggestion
    })

@app.route('/mood-history', methods=['GET'])
def mood_history():
    mood_data = load_json_file(MOOD_DATA_FILE)
    user_id = session.get('user_id')
    
    # Filter mood data for current user
    user_moods = [entry for entry in mood_data if entry.get('user_id') == user_id]
    
    # Get last 30 entries
    recent_moods = user_moods[-30:] if len(user_moods) > 30 else user_moods
    
    return jsonify(recent_moods)

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
