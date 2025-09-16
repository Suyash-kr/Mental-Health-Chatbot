from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import re
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize model and tokenizer as None
tokenizer = None
model = None
model_loaded = False
model_load_attempted = False

def load_model():
    global tokenizer, model, model_loaded, model_load_attempted
    
    if model_load_attempted:
        return
    
    model_load_attempted = True
    model_path = r"C:/Users/explo/Desktop/Mental Health Chatbot/backend/mental_health_chatbot"
    
    try:
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return
        
        # Check if the path is a directory with model files
        required_files = ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt']
        existing_files = os.listdir(model_path)
        logger.info(f"Files in model directory: {existing_files}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
            
        model_loaded = True
        logger.info("✅ Model and tokenizer loaded successfully!")
        
        # Test the model with a simple prompt
        test_prompt = "<user>: Hello <bot>:"
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id
            )
        
        test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model test output: {test_output}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Load model when the app starts
load_model()

MAX_INPUT_LEN = 128
MAX_NEW_TOKENS = 40

def clean_artifacts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[\s*'|'\s*\]|\"\s*\"|\[\s*\"|\"\s*\]", " ", text)
    text = text.replace("']","").replace("['","").replace('"]','').replace('["',"")
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

def generate_bot_reply(user_message):
    """
    Use your trained model to generate responses with better error handling
    """
    # If model isn't loaded, use fallback
    if not model_loaded:
        logger.warning("Model not loaded, using fallback responses")
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
        
        # Generate response
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
        import traceback
        logger.error(traceback.format_exc())
        return generate_fallback_response(user_message)

def generate_fallback_response(user_message):
    """
    Comprehensive fallback responses
    """
    if not user_message:
        return "Hello! How are you feeling today?"
    
    user_message_lower = user_message.lower()
    
    # Crisis detection - always prioritize safety
    crisis_keywords = ["suicide", "kill myself", "end it all", "want to die", "self harm", "hurting myself"]
    if any(keyword in user_message_lower for keyword in crisis_keywords):
        return "I'm really concerned about what you're sharing. Please reach out to a crisis helpline immediately. You can call or text 988 in the US for the Suicide & Crisis Lifeline. Your life is precious and there are people who want to help."
    
    # Emotional states
    if any(word in user_message_lower for word in ["stress", "stressed", "overwhelmed", "pressure"]):
        return "I understand you're feeling stressed. Try taking deep breaths - inhale for 4 seconds, hold for 4, exhale for 6. This can help calm your nervous system. Would you like to talk about what's causing the stress?"
    
    elif any(word in user_message_lower for word in ["sad", "depress", "unhappy", "miserable", "down", "hopeless"]):
        return "I'm sorry you're feeling this way. It's okay to not be okay. Sometimes just talking about what's bothering us can help. Would you like to share more about what's going on?"
    
    elif any(word in user_message_lower for word in ["anxiety", "anxious", "panic", "nervous", "scared", "worry", "worried"]):
        return "Anxiety can feel overwhelming. Have you tried grounding techniques? Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. This can help bring you back to the present moment."
    
    elif any(word in user_message_lower for word in ["angry", "mad", "frustrated", "annoyed", "irritated"]):
        return "It sounds like you're feeling really frustrated. Sometimes taking a short break, going for a walk, or doing some deep breathing can help when emotions feel intense. Would any of those help right now?"
    
    # Greetings
    elif any(word in user_message_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "Hello there! I'm here to listen. How are you feeling today?"
    
    elif any(word in user_message_lower for word in ["thank", "thanks", "appreciate"]):
        return "You're welcome! I'm glad I could help. Remember I'm here whenever you need to talk."
    
    elif "how are you" in user_message_lower:
        return "I'm here and ready to listen. How are you feeling today?"
    
    # Common concerns
    elif any(word in user_message_lower for word in ["sleep", "tired", "insomnia", "exhausted", "can't sleep"]):
        return "Sleep issues are common when we're stressed. Try maintaining a regular sleep schedule, avoiding screens before bed, and creating a calming bedtime routine. Would you like to talk about what might be affecting your sleep?"
    
    elif any(word in user_message_lower for word in ["lonely", "alone", "isolated", "no friends"]):
        return "Feeling lonely can be really difficult. Remember that many people experience loneliness, and it doesn't mean there's anything wrong with you. Would you like to talk about what's been making you feel this way?"
    
    # Default response
    else:
        return "Thank you for sharing that with me. I'm here to listen. Would you like to talk more about how you're feeling?"

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
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"reply": "I'm having trouble processing your message right now. Please try again in a moment."})

# Add a health check endpoint
@app.route("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "model_load_attempted": model_load_attempted
    })

# ---------- Main ----------
if __name__ == "__main__":
    # Print status message
    if model_loaded:
        print("✅ Model loaded successfully! Starting Flask server...")
    else:
        print("⚠️  Model not loaded. Using fallback responses. Starting Flask server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)