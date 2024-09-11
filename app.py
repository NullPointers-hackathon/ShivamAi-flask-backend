from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained GPT-2 model and tokenizer from Hugging Face hub
print("Loading the pre-trained GPT-2 model and tokenizer...")

model_name = "distilgpt2"  # Lighter model for less memory consumption
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

print("Model and tokenizer loaded successfully!")

# Function to generate text
def generate_text(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.9):
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
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return jsonify({'error': 'Please use POST method with a JSON payload.'}), 405
    # Existing POST logic
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    generated_text = generate_text(prompt)
    return jsonify({'generated_text': generated_text})


# Main entry point
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
