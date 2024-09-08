from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained GPT-2 model and tokenizer from Hugging Face hub
print("Loading the pre-trained GPT-2 model and tokenizer...")

model_name = "gpt2"  # You can specify different versions or variants of GPT-2
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

print("Model and tokenizer loaded successfully!")

# Function to generate text
def generate_text(prompt, max_length=500, temperature=0.7, top_k=50, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# API route for text generation
@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate text from a provided prompt.
    
    Expects a JSON payload with 'prompt' key.
    """
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # Generate text
    generated_text = generate_text(prompt)
    return jsonify({'generated_text': generated_text})

# Main entry point
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
