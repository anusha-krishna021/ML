import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# -------------------------------
# Dataset
# -------------------------------
headlines = [
    "Apple releases new AI-powered processor",
    "Ronaldo scores a hat-trick",
    "Google announces breakthrough in quantum computing",
    "India wins the cricket world cup",
    "Tesla reveals autonomous car",
    "Lionel Messi signs contract",
    "Cybersecurity experts warn of malware attacks",
    "Olympic committee reveals new schedule",
    "NASA tests reusable rockets",
    "The football league season begins today"
]

labels = [
    "tech", "sports", "tech", "sports", "tech",
    "sports", "tech", "sports", "tech", "sports"
]

# Map labels to numbers
label_map = {"tech": 0, "sports": 1}
y = [label_map[l] for l in labels]

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(headlines, truncation=True, padding=True, return_tensors="pt")

# -------------------------------
# Dataset class
# -------------------------------
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = NewsDataset(encodings, y)

# -------------------------------
# Load pretrained model
# -------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=5,
    learning_rate=5e-5,
    no_cuda=False
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -------------------------------
# Train the model
# -------------------------------
trainer.train()

# -------------------------------
# Predict
# -------------------------------
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

pred_labels = [list(label_map.keys())[i] for i in predictions]

# -------------------------------
# Print results
# -------------------------------
for i, h in enumerate(headlines):
    print(f"ID {i+1}: {h} | Actual: {labels[i]} | Predicted: {pred_labels[i]}")
