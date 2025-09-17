from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the static directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Create directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = Flask(__name__, 
            static_folder=STATIC_DIR,
            template_folder=TEMPLATE_DIR)
CORS(app)

# Serve static files directly
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# Initialize model and tokenizer as None
tokenizer = None
model = None
model_loaded = False

# Don't load the model at startup to avoid memory issues
# We'll load it on-demand or use fallback responses

MAX_INPUT_LEN = 128
MAX_NEW_TOKENS = 40

def clean_artifacts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[\s*'|'\s*\]|\"\s*\"|\[\s*\"|\"\s*\]", " ", text)
    text = text.replace("']", "").replace("['", "").replace('"]', "").replace('["', "")
    text = re.sub(r"<bot>[:]*", "<bot>:", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r'^[^A-Za-z0-9]+', '', text)
    text = re.sub(r'[^A-Za-z0-9\.\!\?]+$', '', text)
    return text.strip()

def first_n_sentences(text: str, n=2) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    if not parts:
        return text.strip()
    keep = parts[:n]
    out = " ".join(s.rstrip() for s in keep).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out

def load_model_if_needed():
    """Load the model only when needed and only once"""
    global tokenizer, model, model_loaded
    
    if model_loaded:
        return True
        
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Check possible model paths
        possible_paths = [
            "mental_health_chatbot",
            "backend/mental_health_chatbot", 
            "./mental_health_chatbot",
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if not model_path:
            logger.warning("Model path not found")
            return False
            
        logger.info(f"Loading model from: {model_path}")
        
        # Load with optimizations for limited memory
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True,
            torch_dtype="auto",  # Use appropriate dtype
            low_cpu_mem_usage=True  # Reduce memory usage during loading
        )
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
            
        model_loaded = True
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_bot_reply(user_message):
    """
    Use your trained model to generate responses with better error handling
    """
    # Try to load model if not loaded (but don't force it)
    model_available = load_model_if_needed()
    
    # If model isn't available, use fallback
    if not model_available:
        return generate_fallback_response(user_message)
    
    # Skip empty messages
    if not user_message or not user_message.strip():
        return "I'm here to listen. How are you feeling today?"
    
    try:
        # Build prompt
        prompt = f"<user>: {user_message} <bot>:"
        logger.info(f"Generating response for prompt: {prompt}")
        
        # Tokenize with attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
            padding=True
        )
        
        # Generate response with timeout protection
        import signal
        from contextlib import contextmanager
        
        class TimeoutException(Exception):
            pass
        
        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutException("Timed out!")
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
        
        try:
            with time_limit(10):  # 10 second timeout
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        top_p=0.85,
                        temperature=0.6,
                        top_k=50,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
        except TimeoutException:
            logger.warning("Model generation timed out")
            return generate_fallback_response(user_message)
        
        # Decode the response
        out_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        logger.info(f"Raw model output: {out_text}")
        
        # Extract the bot's response
        if "<bot>:" in out_text:
            out_text = out_text.split("<bot>:")[-1]
        if "<user>:" in out_text:
            out_text = out_text.split("<user>:")[0]
        
        # Clean up the response
        out_text = clean_artifacts(out_text)
        out_text = first_n_sentences(out_text, n=2)
        
        # If we got an empty response, use fallback
        if not out_text or out_text.isspace():
            logger.warning("Model returned empty response, using fallback")
            return generate_fallback_response(user_message)
            
        logger.info(f"Final response: {out_text}")
        return out_text
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return generate_fallback_response(user_message)

def generate_fallback_response(user_message):
    """
    Comprehensive fallback responses - optimized for production
    """
    if not user_message:
        return "Hello! How are you feeling today?"
    
    user_message_lower = user_message.lower()
    
    # Crisis detection
    crisis_keywords = ["suicide", "kill myself", "end it all", "want to die", "self harm", "hurting myself"]
    if any(keyword in user_message_lower for keyword in crisis_keywords):
        return "I'm really concerned about what you're sharing. Please reach out to a crisis helpline immediately. You can call or text 988 in the US for the Suicide & Crisis Lifeline."
    
    # Emotional states
    if any(word in user_message_lower for word in ["stress", "stressed", "overwhelmed", "pressure"]):
        return "I understand you're feeling stressed. Try taking deep breaths - inhale for 4 seconds, hold for 4, exhale for 6. This can help calm your nervous system."
    
    elif any(word in user_message_lower for word in ["sad", "depress", "unhappy", "miserable", "down", "hopeless"]):
        return "I'm sorry you're feeling this way. It's okay to not be okay. Would you like to talk about what's going on?"
    
    elif any(word in user_message_lower for word in ["anxiety", "anxious", "panic", "nervous", "scared", "worry", "worried"]):
        return "Anxiety can feel overwhelming. Have you tried grounding techniques? Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste."
    
    elif any(word in user_message_lower for word in ["angry", "mad", "frustrated", "annoyed", "irritated"]):
        return "It sounds like you're feeling frustrated. Sometimes taking a short break or doing deep breathing can help when emotions feel intense."
    
    # Greetings
    elif any(word in user_message_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello there! I'm here to listen. How are you feeling today?"
    
    elif any(word in user_message_lower for word in ["thank", "thanks", "appreciate"]):
        return "You're welcome! I'm glad I could help."
    
    elif "how are you" in user_message_lower:
        return "I'm here and ready to listen. How are you feeling today?"
    
    # Default response
    else:
        return "Thank you for sharing that with me. I'm here to listen."

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    """
    Handles chat messages sent from the frontend.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "I didn't receive your message. Could you try again?"})
        
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "I'm here to listen. How are you feeling today?"})
        
        bot_reply = generate_bot_reply(user_message)
        return jsonify({"reply": bot_reply})
        
    except Exception as e:
        logger.error(f"Error in get_response: {e}")
        return jsonify({"reply": "I'm having trouble processing your message right now. Please try again in a moment."})

@app.route("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "model_available": model_loaded
    })

# Test endpoint to verify images are loading
@app.route("/test-images")
def test_images():
    """Test endpoint to verify images are loading"""
    images = [
        'logo.png', 'hero.png', 'feature1.png', 'feature2.png', 
        'feature3.png', 'about-hero.png', 'mission.png', 'suyashgupta.png',
        'instagram.png', 'call.png', 'linkedin.png', 'github.png', 'mail.png'
    ]
    
    result = "<h1>Image Test</h1>"
    for image in images:
        image_path = os.path.join(STATIC_DIR, image)
        if os.path.exists(image_path):
            result += f'<p>✓ {image} - EXISTS - <img src="/static/{image}" height="50"></p>'
        else:
            result += f'<p>✗ {image} - MISSING</p>'
    
    return result

# ---------- Main ----------
if __name__ == "__main__":
    # For production, use the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
