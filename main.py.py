import streamlit as st
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch


# load model and tokenizer
model = XLNetForSequenceClassification.from_pretrained("paraphrase_xlnet_model")
tokenizer = XLNetTokenizer.from_pretrained('paraphrase_xlnet_model')

# Inference function
def make_inference(sentence1, sentence2):
    # Set device to match model's device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the correct device
    model.to(device)

    # Tokenize the input sentences
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class


# inference function

# app
st.title("Sequence Pair Classification Using XLNet LLM Model")
sent1 = st.text_input("Enter first sentence....")
sent2 = st.text_input("Enter Second Sentence....")

if st.button("classify"):
    if sent1 and sent2:
        resulted_class = make_inference(sent1,sent2)

        if resulted_class==1:
            st.write("Paraphrase Detected")
        else:
            st.write("No Paraphrase Detected")
    else:
        st.write("enter bother both sentences")