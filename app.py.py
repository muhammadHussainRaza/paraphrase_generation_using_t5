from flask import Flask, render_template, request
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load the model and tokenizer from the saved directory
model_dir = "saved_t5_model"  # Replace with your model directory path
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Preprocessing function for inference
def preprocess_input(sentence):
    return "paraphrase: " + sentence

# Generate paraphrases function
def generate_paraphrase(input_text, model, tokenizer, max_length=128, num_beams=5, num_return_sequences=4, top_k=100, top_p=0.9, temperature=1.0):
    input_text = preprocess_input(input_text)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length + 20,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        do_sample=True,
        early_stopping=True
    )

    paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrased_texts

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_sentence = request.form["sentence"]
        paraphrased_sentences = generate_paraphrase(input_sentence, model, tokenizer)
        return render_template("index.html", original=input_sentence, paraphrases=paraphrased_sentences)
    return render_template("index.html", original=None, paraphrases=None)

if __name__ == "__main__":
    app.run(debug=True)
