import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 1. Create the 'model' folder if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')
    print("Created 'model' folder.")

# 2. Create dummy data to generate the structure
# This ensures the files are valid and compatible with the main app
dummy_texts = [
    "free money win lottery",   # Spam sample
    "hello friend meeting",     # Real sample
    "click this link offer",    # Spam sample
    "good morning have a nice day" # Real sample
]
dummy_labels = [1, 0, 1, 0] # 1=Spam, 0=Real

# 3. Generate 'feature.pkl' (The Vocabulary)
print("Generating feature.pkl...")
cv = CountVectorizer(stop_words='english')
cv.fit(dummy_texts)
vocabulary = cv.vocabulary_

with open('model/feature.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)

# 4. Generate 'naiveBayes.pkl' (The Classifier)
print("Generating naiveBayes.pkl...")
classifier = MultinomialNB()
X = cv.transform(dummy_texts)
classifier.fit(X, dummy_labels)

with open('model/naiveBayes.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("\nSUCCESS! Files created:")
print("- model/feature.pkl")
print("- model/naiveBayes.pkl")
print("You can now run Spam.py without errors.")