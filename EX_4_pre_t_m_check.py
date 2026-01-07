# New headlines to classify
new_headlines = [
    "Microsoft launches cloud AI platform",
    "Apple unveils new iPhone with advanced camera",
    "Messi scores in Champions League",
    "NASA plans mission to Mars",
    "India defeats Australia in cricket final"
]

# Tokenize the new headlines
new_encodings = tokenizer(new_headlines, truncation=True, padding=True, return_tensors="pt")

# Predict using the fine-tuned model
with torch.no_grad(): # disables gradient calculation for speed
    outputs = model(**new_encodings)
    predictions = torch.argmax(outputs.logits, dim=1) # get class with highest probability

# Convert numeric predictions back to labels
pred_labels = [list(label_map.keys())[i] for i in predictions]

# Print the results
for i, h in enumerate(new_headlines):
    print(f"Headline: {h} | Predicted: {pred_labels[i]}")
