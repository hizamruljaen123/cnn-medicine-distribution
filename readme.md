# SINTA Journal Analysis with Lexical Chain and Google SERP Scraper

## Introduction
This workflow combines:
1. **Lexical Chain** for semantic analysis of SINTA journal texts.  
2. **Google SERP Scraper** to automatically collect SINTA journal links and abstracts.  

It can be used for scientometric analysis, topic extraction, or automated literature review.

---

## 1. Lexical Chain for SINTA Journals
Lexical chain is an NLP technique to detect semantic relationships between words in a text.  
It is useful to find **main topics** or trends in journal articles.

### Steps
1. Collect abstracts or full texts from SINTA journals.  
2. Tokenize text and perform **POS tagging**.  
3. Build **lexical chains** by linking semantically related words (synonyms, hypernyms).  
4. Assign weights (TF-IDF or normalized frequency) to terms.  
5. Identify dominant topics.

### Python Example
```python
import nltk
from nltk.corpus import wordnet
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Sample abstracts
abstracts = [
    "IoT in agriculture improves crop monitoring. IoT sensors collect data efficiently.",
    "Machine learning enhances predictive analysis for crop yield and pest detection."
]

lexical_chains = []

for text in abstracts:
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    chains = {}
    for word, pos in pos_tags:
        syns = wordnet.synsets(word)
        if syns:
            key = syns[0].lemmas()[0].name()
            if key not in chains:
                chains[key] = []
            chains[key].append(word)
    lexical_chains.append(chains)

print("Lexical Chains per Abstract:", lexical_chains)
