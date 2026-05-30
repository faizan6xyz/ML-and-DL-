"""
=============================================================================
 COMPLETE NLP ROADMAP — Code + Explanations
 For: Data Science & ML Engineering
 Covers: Preprocessing → Embeddings → Tasks → Deep Learning → Evaluation
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: Clean and normalize raw text before feeding it into any model.
WHY:  Raw text is messy — punctuation, HTML tags, inconsistent casing,
      contractions, extra spaces. Models perform much better on clean input.
"""

import re
import string
import nltk

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

sample_text = "  Hello! I'm learning NLP in 2024. It's AMAZING! Visit https://nlp.org for more.  "

# 1.1 Lowercasing
lower = sample_text.lower()
print("1.1 Lowercase:", lower)

# 1.2 Remove URLs
no_urls = re.sub(r'http\S+|www\S+', '', lower)
print("1.2 No URLs:", no_urls)

# 1.3 Remove punctuation & special characters
no_punct = re.sub(r"[^a-z0-9\s]", "", no_urls)
print("1.3 No Punctuation:", no_punct)

# 1.4 Tokenization — splitting text into words or sentences
"""
WHY: Models can't work with raw strings; they need discrete tokens.
     Word tokenize splits into words; sent_tokenize splits into sentences.
"""
word_tokens = word_tokenize(no_punct)
sent_tokens = sent_tokenize(sample_text)
print("1.4a Word Tokens:", word_tokens)
print("1.4b Sentence Tokens:", sent_tokens)

# 1.5 Stopword Removal
"""
WHY: Words like 'is', 'the', 'in' appear everywhere but carry little meaning.
     Removing them reduces noise (useful for classical ML, less so for Transformers).
"""
stop_words = set(stopwords.words('english'))
filtered = [w for w in word_tokens if w not in stop_words and w.strip()]
print("1.5 After Stopword Removal:", filtered)

# 1.6 Stemming
"""
WHY: Reduces words to their root form by chopping off suffixes.
     Fast but crude — "running" → "run", "studies" → "studi" (not a real word).
"""
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
print("1.6 Stemmed:", stemmed)

# 1.7 Lemmatization
"""
WHY: Like stemming but smarter — uses vocabulary to get the real root word.
     "studies" → "study", "running" → "run". Slower but more accurate.
"""
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
print("1.7 Lemmatized:", lemmatized)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TEXT REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: Convert text into numbers (vectors) that models can process.
WHY:  ML models only understand numbers. The quality of representation
      directly determines model performance.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
]

# 2.1 Bag of Words (BoW)
"""
CONCEPT: Create a vocabulary of all unique words, then represent each
          document as a vector of word counts.
PROBLEM:  Ignores word order. "dog bites man" = "man bites dog".
          Also ignores word importance — common words dominate.
"""
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus)
print("\n2.1 BoW Vocabulary:", bow_vectorizer.get_feature_names_out())
print("2.1 BoW Matrix:\n", bow_matrix.toarray())

# 2.2 TF-IDF (Term Frequency - Inverse Document Frequency)
"""
CONCEPT: Weights words by how important they are to a document vs. the corpus.
         TF  = how often word appears in this document
         IDF = penalizes words common across ALL documents (like "the")
RESULT:  "cat" gets a high score in doc1 if it's rare in other docs.
USE:     Text classification, search ranking, keyword extraction.
"""
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print("\n2.2 TF-IDF Matrix:\n", tfidf_matrix.toarray().round(3))

# 2.3 Word2Vec (via Gensim)
"""
CONCEPT: Neural network trained to predict a word from its neighbors (CBOW)
          or predict neighbors from a word (Skip-gram).
RESULT:  Each word gets a dense vector where similar words cluster together.
         Famous property: king - man + woman ≈ queen
USE:     Semantic similarity, document embeddings, feature input for models.
"""
from gensim.models import Word2Vec

sentences = [s.split() for s in corpus]
w2v_model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, epochs=100, seed=42)

print("\n2.3 Word2Vec vector for 'cat':", w2v_model.wv['cat'].round(3))
print("2.3 Most similar to 'cat':", w2v_model.wv.most_similar('cat', topn=2))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CORE NLP TASKS
# ─────────────────────────────────────────────────────────────────────────────

# 3.1 POS Tagging (Part-of-Speech)
"""
WHAT: Label each word with its grammatical role — noun, verb, adjective, etc.
WHY:  Helps understand sentence structure. Used in parsing, NER, and
      rule-based systems. "Book the flight" vs "Read the book" — same word,
      different POS tags (verb vs noun).
"""
from nltk import pos_tag

pos_sentence = "The quick brown fox jumps over the lazy dog"
pos_tokens = word_tokenize(pos_sentence)
pos_tags = pos_tag(pos_tokens)
print("\n3.1 POS Tags:", pos_tags)
# Tag meanings: NN=Noun, VBZ=Verb, JJ=Adjective, DT=Determiner, IN=Preposition

# 3.2 Named Entity Recognition (NER)
"""
WHAT: Identify and classify named entities — people, organizations, locations,
      dates, monetary values, etc.
WHY:  Core task for information extraction. Used in finance (extracting company
      names), healthcare (extracting drug names), news analysis, etc.
"""
from nltk import ne_chunk
from nltk.tree import Tree

ner_sentence = "Apple was founded by Steve Jobs in Cupertino California in 1976"
ner_tokens = word_tokenize(ner_sentence)
ner_tags = pos_tag(ner_tokens)
ner_tree = ne_chunk(ner_tags)

entities = []
for subtree in ner_tree:
    if isinstance(subtree, Tree):
        entity = " ".join([word for word, tag in subtree.leaves()])
        label = subtree.label()
        entities.append((entity, label))

print("\n3.2 Named Entities:", entities)

# 3.3 Text Classification (Sentiment Analysis)
"""
WHAT: Assign a category label to a piece of text.
WHY:  Sentiment analysis, spam detection, topic classification, intent detection.
      One of the most common real-world NLP tasks in production.
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

reviews = [
    "This product is amazing and works perfectly",
    "Absolutely love it, highly recommend",
    "Great quality and fast shipping",
    "Best purchase I have ever made",
    "Really happy with this product",
    "Terrible product, broke after one day",
    "Complete waste of money, very disappointed",
    "Worst experience ever, do not buy",
    "Poor quality and bad customer service",
    "Returned immediately, totally useless",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.3, random_state=42
)

# Pipeline: TF-IDF → Logistic Regression
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # unigrams + bigrams
    ('clf', LogisticRegression(max_iter=1000))
])

sentiment_pipeline.fit(X_train, y_train)
y_pred = sentiment_pipeline.predict(X_test)
print("\n3.3 Sentiment Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], zero_division=0))

# Predict new examples
new_reviews = ["Outstanding quality!", "Terrible, never again"]
predictions = sentiment_pipeline.predict(new_reviews)
for review, pred in zip(new_reviews, predictions):
    print(f"  '{review}' → {'Positive' if pred == 1 else 'Negative'}")

# 3.4 Cosine Similarity (Semantic Similarity)
"""
WHAT: Measure how similar two texts are using their vector representations.
WHY:  Used in search engines, recommendation systems, duplicate detection,
      and retrieval-augmented generation (RAG).
CONCEPT: Two vectors pointing in the same direction = cosine similarity of 1.
          Perpendicular vectors = 0. Opposite directions = -1.
"""
from sklearn.metrics.pairwise import cosine_similarity

sim_corpus = [
    "I love machine learning",
    "Deep learning is a subset of machine learning",
    "I enjoy cooking Italian food",
]

sim_vectorizer = TfidfVectorizer()
sim_matrix = sim_vectorizer.fit_transform(sim_corpus)
similarity = cosine_similarity(sim_matrix)

print("\n3.4 Cosine Similarity Matrix:")
for i, doc1 in enumerate(sim_corpus):
    for j, doc2 in enumerate(sim_corpus):
        if i < j:
            print(f"  '{doc1[:30]}...' ↔ '{doc2[:30]}...': {similarity[i][j]:.3f}")

# 3.5 Topic Modeling (LDA)
"""
WHAT: Discover hidden topics in a collection of documents — unsupervised.
WHY:  Useful for exploring large text corpora, document organization,
      content recommendation, and understanding what themes exist in data.
CONCEPT: LDA assumes each document is a mix of topics, and each topic is
          a distribution over words.
"""
from sklearn.decomposition import LatentDirichletAllocation

topic_corpus = [
    "neural networks deep learning training gradient",
    "python programming code software development",
    "machine learning algorithms data classification",
    "javascript web frontend react components",
    "backpropagation activation functions neural",
    "database sql queries tables indexes",
]

lda_vectorizer = CountVectorizer(stop_words='english')
lda_dtm = lda_vectorizer.fit_transform(topic_corpus)

lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_model.fit(lda_dtm)

feature_names = lda_vectorizer.get_feature_names_out()
print("\n3.5 LDA Topic Modeling:")
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
    print(f"  Topic {topic_idx + 1}: {', '.join(top_words)}")

# 3.6 Text Summarization (Extractive)
"""
WHAT: Produce a shorter version of a document that retains key information.
      Extractive = pick the most important existing sentences.
      Abstractive = generate new sentences (requires seq2seq models like T5).
WHY:  News summarization, document review, meeting notes, report generation.
APPROACH: Rank sentences by TF-IDF score and pick top-N.
"""
def extractive_summarize(text, num_sentences=2):
    sentences = sent_tokenize(text)
    
    # Score sentences using TF-IDF
    sent_vectorizer = TfidfVectorizer()
    sent_matrix = sent_vectorizer.fit_transform(sentences)
    
    # Score = sum of TF-IDF scores for all words in sentence
    scores = sent_matrix.sum(axis=1).A1
    ranked_indices = np.argsort(scores)[::-1][:num_sentences]
    
    # Return sentences in original order
    summary_sentences = [sentences[i] for i in sorted(ranked_indices)]
    return " ".join(summary_sentences)

long_text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence 
that focuses on the interaction between computers and human language. 
It involves tasks such as text classification, machine translation, and 
sentiment analysis. Deep learning has revolutionized NLP in recent years.
The Transformer architecture, introduced in 2017, became the foundation 
for models like BERT and GPT. These models achieve state-of-the-art results 
on almost every NLP benchmark. Transfer learning allows practitioners to 
fine-tune these large models on specific tasks with relatively little data.
"""

summary = extractive_summarize(long_text, num_sentences=2)
print("\n3.6 Extractive Summary:")
print(" ", summary)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DEEP LEARNING FOR NLP — TRANSFORMERS
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: The Transformer (Vaswani et al., 2017 — "Attention is All You Need")
      is the architecture powering BERT, GPT, T5, and all modern LLMs.

KEY CONCEPTS:

1. SELF-ATTENTION
   Each word looks at all other words in the sequence to compute its
   contextual representation. "Bank" in "river bank" vs "bank account"
   gets different representations based on surrounding words.

   Attention(Q, K, V) = softmax(QK^T / √d_k) V
   Q = Query, K = Key, V = Value (all learned projections of the input)

2. MULTI-HEAD ATTENTION
   Run self-attention multiple times in parallel with different weight matrices.
   Each "head" learns to attend to different types of relationships
   (e.g., one head for syntax, another for semantics).

3. POSITIONAL ENCODING
   Since Transformers process all tokens in parallel (no recurrence),
   we add positional encodings to inject order information.
   sin/cos functions of different frequencies encode position.

4. FEED-FORWARD LAYERS
   After attention, each position goes through the same 2-layer MLP.
   This adds non-linearity and increases model capacity.

5. ENCODER vs DECODER
   - Encoder (BERT): reads the whole input, good for understanding tasks
     (classification, NER, QA)
   - Decoder (GPT): generates text left-to-right, good for generation
   - Encoder-Decoder (T5, BART): good for seq2seq (translation, summarization)
"""

import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Core attention mechanism.
    Q, K, V: Query, Key, Value matrices
    Returns weighted combination of Values based on Query-Key similarity.
    """
    d_k = Q.shape[-1]  # dimension of key vectors
    
    # Step 1: Compute attention scores (how much each position attends to others)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # Scale to prevent vanishing gradients
    
    # Step 2: Softmax to get attention weights (probabilities that sum to 1)
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))  # numerical stability
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Step 3: Weighted sum of Values
    output = np.dot(weights, V)
    return output, weights

# Demo with tiny example
np.random.seed(42)
seq_len, d_model = 4, 8  # 4 tokens, 8-dimensional embeddings
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)
print("\n4.1 Self-Attention:")
print(f"  Input shape: {Q.shape} (seq_len=4, d_model=8)")
print(f"  Output shape: {attn_output.shape}")
print(f"  Attention weights (row = token, col = how much it attends to each token):")
print(attn_weights.round(3))

def positional_encoding(seq_len, d_model):
    """
    Inject position information using sin/cos waves of different frequencies.
    Even dimensions use sin, odd dimensions use cos.
    The model learns to use these patterns to understand word order.
    """
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]   = np.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

PE = positional_encoding(seq_len=6, d_model=8)
print("\n4.2 Positional Encoding (6 positions, 8 dims):")
print(PE.round(3))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: USING PRE-TRAINED TRANSFORMERS (Hugging Face)
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: Instead of training from scratch (requires millions of data points
      and GPUs), use models pre-trained on massive corpora, then fine-tune.

KEY MODELS:
  BERT  (Google, 2018) — Bidirectional Encoder. Pre-trained with:
        - Masked Language Model (MLM): predict masked words
        - Next Sentence Prediction (NSP): are these sentences consecutive?
        Best for: classification, NER, QA (understanding tasks)

  GPT   (OpenAI, 2018+) — Decoder. Trained to predict next token.
        Best for: text generation, completion, chat

  T5    (Google, 2020) — Encoder-Decoder. Frames ALL tasks as text→text.
        "Translate English to French: Hello" → "Bonjour"
        Best for: translation, summarization, QA

  RoBERTa — Improved BERT (more data, no NSP, longer training)

FINE-TUNING WORKFLOW:
  1. Load pre-trained model + tokenizer from Hugging Face Hub
  2. Add a task-specific head (classification layer, etc.)
  3. Train on your labeled data with a small learning rate (2e-5 to 5e-5)
  4. The pre-trained weights already encode language knowledge —
     fine-tuning adapts them to your specific task

NOTE: The code below shows the API structure. To actually run it,
      install: pip install transformers torch
      and uncomment the code.
"""

print("\n5. Hugging Face Transformers — API Overview (install transformers to run)")

HUGGINGFACE_EXAMPLE = '''
# ── Sentiment Analysis with Pipeline ──────────────────────────────────────
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely loved this movie!")
# → [{'label': 'POSITIVE', 'score': 0.9998}]

# ── Tokenization ───────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

text = "The stock market crashed today"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
# inputs = {"input_ids": tensor([[101, 1996, ...]]), "attention_mask": tensor([[1, 1, ...]])}
# input_ids: token IDs from vocabulary (101 = [CLS], 102 = [SEP])
# attention_mask: 1 for real tokens, 0 for padding tokens

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits            # raw scores before softmax
    probs = torch.softmax(logits, -1)  # convert to probabilities

# ── Fine-tuning Example ────────────────────────────────────────────────────
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # small LR preserves pre-trained knowledge
    warmup_steps=500,            # gradually increase LR at the start
    weight_decay=0.01,           # L2 regularization
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# ── Zero-shot Classification (no training data needed!) ────────────────────
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "The new iPhone has a better camera",
    candidate_labels=["technology", "sports", "politics"]
)
# → {'labels': ['technology', 'sports', 'politics'], 'scores': [0.98, 0.01, 0.01]}

# ── Named Entity Recognition ───────────────────────────────────────────────
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")
result = ner("Elon Musk founded SpaceX in Hawthorne, California")
# → [{'entity_group': 'PER', 'word': 'Elon Musk', 'score': 0.999, ...},
#    {'entity_group': 'ORG', 'word': 'SpaceX', ...},
#    {'entity_group': 'LOC', 'word': 'Hawthorne', ...}]
'''

print(HUGGINGFACE_EXAMPLE)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LLMs & MODERN NLP (RAG, Fine-tuning, Embeddings)
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: Large Language Models (LLMs) like GPT-4, Claude, Llama are trained on
      trillions of tokens and can perform almost any NLP task via prompting.

RAG (Retrieval-Augmented Generation):
  PROBLEM: LLMs have a knowledge cutoff and can hallucinate facts.
  SOLUTION: Retrieve relevant documents from a vector database at inference
            time, and inject them into the prompt as context.

  PIPELINE:
    1. Offline: Chunk documents → embed with model → store in vector DB
    2. Online:  User query → embed query → retrieve top-K similar chunks
                → LLM generates answer grounded in retrieved context

FINE-TUNING TECHNIQUES (for LLMs):
  Full Fine-tuning: Update all parameters. Expensive (needs many GPUs).
  
  LoRA (Low-Rank Adaptation):
    Instead of updating W (huge matrix), learn two small matrices A and B
    such that ΔW = A × B. Reduces trainable params by 10,000x!
    Only A and B are trained; original W is frozen.
  
  QLoRA: LoRA + 4-bit quantization. Fine-tune 70B models on a single GPU!

VECTOR DATABASES:
  Store embeddings as vectors. Support fast approximate nearest-neighbor
  (ANN) search using algorithms like HNSW, IVF.
  Popular choices: FAISS (local), Pinecone, Weaviate, ChromaDB, Qdrant.
"""

print("6. LLMs & Modern NLP — Architecture Concepts")

MODERN_NLP_CODE = '''
# ── RAG Pipeline (conceptual) ──────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# OFFLINE: Build the knowledge base
documents = [
    "Python was created by Guido van Rossum in 1991",
    "BERT was published by Google in 2018",
    "Transformers use self-attention mechanisms",
    "PyTorch is developed by Meta AI",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings
doc_embeddings = embedder.encode(documents)  # shape: (4, 384)

# Build FAISS index for fast similarity search
index = faiss.IndexFlatL2(384)  # L2 distance in 384-dim space
index.add(doc_embeddings.astype(np.float32))

# ONLINE: Answer a user query
query = "Who made BERT?"
query_embedding = embedder.encode([query])

# Retrieve top-2 most relevant documents
distances, indices = index.search(query_embedding.astype(np.float32), k=2)
retrieved = [documents[i] for i in indices[0]]

# Feed retrieved context to LLM
prompt = f"""Answer based on context only.
Context: {" ".join(retrieved)}
Question: {query}
Answer:"""

# ── LoRA Fine-tuning (with PEFT library) ──────────────────────────────────
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=16,               # rank of LoRA matrices (higher = more params)
    lora_alpha=32,      # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to apply LoRA to
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# → trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%
# Only 0.06% of parameters are trained! The rest are frozen.
'''

print(MODERN_NLP_CODE)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: Quantify how well your NLP model performs.
WHY:  Different tasks need different metrics — accuracy is often misleading
      for imbalanced datasets or generation tasks.
"""

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

# 7.1 Classification Metrics
"""
PRECISION = TP / (TP + FP) → Of all predicted positives, how many are correct?
RECALL    = TP / (TP + FN) → Of all actual positives, how many did we catch?
F1-SCORE  = 2 × (P × R) / (P + R) → Harmonic mean of Precision and Recall.
            Use when you care equally about both.

WHEN TO USE WHICH:
  High Recall needed  → Medical diagnosis (don't miss cancer cases)
  High Precision needed → Spam filter (don't mark real emails as spam)
  F1 balanced         → Most production NLP tasks
"""

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
cm = confusion_matrix(y_true, y_pred)

print("\n7.1 Classification Metrics:")
print(f"  Accuracy:  {acc:.3f}  (correct predictions / total)")
print(f"  Precision: {prec:.3f}  (of predicted positive, how many correct)")
print(f"  Recall:    {rec:.3f}  (of actual positive, how many found)")
print(f"  F1 Score:  {f1:.3f}  (harmonic mean of precision & recall)")
print(f"  Confusion Matrix:\n{cm}")

# 7.2 BLEU Score (for Machine Translation / Text Generation)
"""
CONCEPT: Measures how many n-grams in the generated text appear in the
          reference (human) text.
BLEU-1: unigram overlap, BLEU-4: up to 4-gram overlap (standard).
RANGE:  0 (terrible) to 1 (perfect match).
LIMIT:  Penalizes valid paraphrases. A good translation can score low if
        it uses different but correct words.
"""
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["the", "cat", "is", "on", "the", "mat"]]
hypothesis_good = ["the", "cat", "sits", "on", "the", "mat"]
hypothesis_bad  = ["dog", "runs", "in", "park"]

smoother = SmoothingFunction().method1
bleu_good = sentence_bleu(reference, hypothesis_good, smoothing_function=smoother)
bleu_bad  = sentence_bleu(reference, hypothesis_bad,  smoothing_function=smoother)
print(f"\n7.2 BLEU Score:")
print(f"  Good hypothesis: {bleu_good:.4f}")
print(f"  Bad  hypothesis: {bleu_bad:.4f}")

# 7.3 ROUGE Score (for Summarization)
"""
CONCEPT: Measures overlap between generated summary and reference summary.
ROUGE-1: unigram overlap (individual words)
ROUGE-2: bigram overlap (word pairs)
ROUGE-L: longest common subsequence (order matters)
"""
def rouge_n(hypothesis, reference, n=1):
    """Simple ROUGE-N implementation."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    hyp_ngrams = get_ngrams(hypothesis.lower().split(), n)
    ref_ngrams = get_ngrams(reference.lower().split(), n)
    
    if not ref_ngrams:
        return 0.0
    
    overlap = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
    recall    = overlap / len(ref_ngrams)  if ref_ngrams else 0
    precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return f1

ref_summary  = "The cat sat on the mat near the window"
hyp_summary1 = "A cat was sitting on a mat"
hyp_summary2 = "Dogs played in the park all day"

print(f"\n7.3 ROUGE-1 F1:")
print(f"  Good summary: {rouge_n(hyp_summary1, ref_summary, n=1):.3f}")
print(f"  Bad  summary: {rouge_n(hyp_summary2, ref_summary, n=1):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ML ENGINEERING FOR NLP
# ─────────────────────────────────────────────────────────────────────────────
"""
WHAT: Taking NLP models from notebook to production.
WHY:  A great model that can't serve 1000 req/sec or drifts silently is
      useless in the real world. ML engineering is what bridges research
      and production.
"""

print("\n8. ML Engineering for NLP — Key Concepts")

MLENG_CONCEPTS = '''
# ── 8.1 Text Data Pipeline ────────────────────────────────────────────────
# For large corpora (GBs of text), use streaming to avoid memory overflow.

from datasets import load_dataset

# Stream huge datasets without loading all into RAM
dataset = load_dataset("wikipedia", "20220301.en", streaming=True)

# Batch processing with HF datasets (uses Apache Arrow under the hood)
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=512),
    batched=True,  # process 1000 examples at a time
    num_proc=4     # parallel processing
)

# ── 8.2 Model Serving with FastAPI ────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()
model = pipeline("sentiment-analysis")  # loaded once at startup

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    result = model(request.text)[0]
    return {"label": result["label"], "confidence": round(result["score"], 4)}

# Run: uvicorn main:app --host 0.0.0.0 --port 8000

# ── 8.3 Model Optimization ────────────────────────────────────────────────
# Quantization: reduce model size and increase inference speed

import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Dynamic Quantization: convert weights to INT8 (from FP32)
# Result: ~4x smaller, ~2-3x faster on CPU, minimal accuracy loss
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Export to ONNX for cross-platform deployment (works on TensorFlow, C++, etc.)
torch.onnx.export(
    model,
    dummy_input,          # example input tensor
    "model.onnx",
    opset_version=14,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}}
)

# ── 8.4 Monitoring NLP Models in Production ───────────────────────────────
"""
KEY THINGS TO MONITOR:
  1. Data Drift    — Input text distribution changes (new slang, domain shift)
  2. Concept Drift — Relationship between text and labels changes
                     (word meanings shift over time)
  3. Latency       — P50, P95, P99 response times
  4. Confidence    — If avg confidence drops, model may be confused by new data
  5. Label Distribution — If predictions skew toward one class unexpectedly

TOOLS: Evidently AI, WhyLabs, Arize AI, MLflow, Prometheus + Grafana
"""

import mlflow

mlflow.set_experiment("nlp-sentiment-model")
with mlflow.start_run():
    mlflow.log_param("model", "bert-base-uncased")
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_metric("val_f1", 0.923)
    mlflow.log_metric("val_accuracy", 0.941)
    mlflow.sklearn.log_model(sentiment_pipeline, "model")

# ── 8.5 Experiment Tracking with Weights & Biases ─────────────────────────
import wandb

wandb.init(project="nlp-experiments", config={
    "model": "bert-base-uncased",
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16
})

# Log metrics during training
for epoch in range(3):
    train_loss = 0.3 - epoch * 0.05
    val_f1     = 0.85 + epoch * 0.03
    wandb.log({"train_loss": train_loss, "val_f1": val_f1, "epoch": epoch})

wandb.finish()
'''

print(MLENG_CONCEPTS)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & LEARNING PATH
# ─────────────────────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    NLP LEARNING ROADMAP SUMMARY                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  SECTION 1: Preprocessing   → Clean text (tokenize, stem, lemmatize)  ║
║  SECTION 2: Representation  → BoW, TF-IDF, Word2Vec embeddings        ║
║  SECTION 3: Core Tasks      → Classification, NER, POS, Summarization ║
║  SECTION 4: Transformers    → Self-attention, positional encoding      ║
║  SECTION 5: Hugging Face    → BERT, GPT, T5 pipelines & fine-tuning   ║
║  SECTION 6: Modern NLP      → RAG, LoRA, Vector DBs, LLM pipelines    ║
║  SECTION 7: Evaluation      → Accuracy, F1, BLEU, ROUGE               ║
║  SECTION 8: ML Engineering  → Serving, quantization, monitoring       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  RECOMMENDED ORDER (since you know some NLP already):                  ║
║  1. Master Transformers architecture (Section 4)                       ║
║  2. Hugging Face fine-tuning workflows (Section 5)                     ║
║  3. Build a RAG pipeline end-to-end (Section 6)                        ║
║  4. Deploy a model with FastAPI + monitoring (Section 8)               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")