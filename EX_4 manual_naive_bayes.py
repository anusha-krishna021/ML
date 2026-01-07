# -------------------------------
# Manual Naive Bayes for Headlines
# -------------------------------

import math

# -------------------------------
# Step 1: Dataset
# -------------------------------
data = [
    {"headline": "Apple releases new AI-powered processor", "label": "tech"},
    {"headline": "Ronaldo scores a hat-trick", "label": "sports"},
    {"headline": "Google announces breakthrough in quantum computing", "label": "tech"},
    {"headline": "India wins the cricket world cup", "label": "sports"},
    {"headline": "Tesla reveals autonomous car", "label": "tech"},
    {"headline": "Lionel Messi signs contract", "label": "sports"},
    {"headline": "Cybersecurity experts warn of malware attacks", "label": "tech"},
    {"headline": "Olympic committee reveals new schedule", "label": "sports"},
    {"headline": "NASA tests reusable rockets", "label": "tech"},
    {"headline": "The football league season begins today", "label": "sports"}
]

# -------------------------------
# Step 2: Preprocessing
# Lowercase and split words
# -------------------------------
for d in data:
    d["words"] = d["headline"].lower().replace("-", " ").split()

# -------------------------------
# Step 3: Separate by class
# -------------------------------
classes = {}
for d in data:
    label = d["label"]
    if label not in classes:
        classes[label] = []
    classes[label].append(d["words"])

# -------------------------------
# Step 4: Build word frequency for each class
# -------------------------------
word_freq = {}
total_words_in_class = {}
vocab = set()

for label, headlines in classes.items():
    word_freq[label] = {}
    total_words = 0
    for words in headlines:
        for w in words:
            vocab.add(w)
            word_freq[label][w] = word_freq[label].get(w, 0) + 1
            total_words += 1
    total_words_in_class[label] = total_words

vocab_size = len(vocab)

# -------------------------------
# Step 5: Calculate priors
# -------------------------------
priors = {}
total_docs = len(data)
for label in classes:
    priors[label] = len(classes[label]) / total_docs

# -------------------------------
# Step 6: Prediction function
# -------------------------------
def predict(headline):
    words = headline.lower().replace("-", " ").split()
    scores = {}
    for label in classes:
        log_prob = math.log(priors[label]) # use log to avoid underflow
        for w in words:
            # Laplace smoothing
            count = word_freq[label].get(w, 0)
            prob = (count + 1) / (total_words_in_class[label] + vocab_size)
            log_prob += math.log(prob)
        scores[label] = log_prob
    # Return label with highest probability
    return max(scores, key=scores.get), scores

# -------------------------------
# Step 7: Test all headlines
# -------------------------------
correct = 0
print(f"{'ID':<3} {'Headline':<50} {'Actual':<7} {'Predicted':<7}")
for i, d in enumerate(data, 1):
    pred, probs = predict(d["headline"])
    if pred == d["label"]:
        correct += 1
    print(f"{i:<3} {d['headline']:<50} {d['label']:<7} {pred:<7}")

accuracy = correct / total_docs
print(f"\nAccuracy: {accuracy*100:.2f}%")

