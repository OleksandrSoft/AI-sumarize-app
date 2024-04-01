from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_text(text, model_name="t5-small"):
    # Load pre-trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example text to summarize
text = """
Insert your comment here.
"""

# Get the summary
summary = summarize_text(text)
print("Summary:", summary)