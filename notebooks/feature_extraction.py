# It is a simple script to define the set of functions that will be used in Task 2 to extract features.

import re
import json
import numpy as np
import textstat
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

FEATURE_COLUMNS = [
    'label', 'text_id',
    'ttr', 'mattr',
    'adj_noun_ratio', 'adv_per_100', 'past_tense_ratio', 'noun_density',
    'tree_depth',
    'comma_rate', 'semicolon_rate', 'colon_rate', 'emdash_rate',
    'exclamation_rate', 'question_rate',
    'fk_grade', 'reading_ease',
]


def load_jsonl(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_ttr(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_mattr(text, window=50):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < window:
        return len(set(words)) / len(words) if words else 0.0
    ttrs = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        ttrs.append(len(set(chunk)) / window)
    return np.mean(ttrs)


def compute_pos_features(text):
    doc = nlp(text)
    total_tokens = len([t for t in doc if not t.is_space])
    pos_counts = Counter(t.pos_ for t in doc if not t.is_space and not t.is_punct)

    noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    adj_count  = pos_counts.get('ADJ', 0)
    adv_count  = pos_counts.get('ADV', 0)
    verb_count = pos_counts.get('VERB', 0)
    past_verbs = sum(1 for t in doc if t.tag_ == 'VBD')

    return {
        'adj_noun_ratio':  adj_count  / noun_count   if noun_count   > 0 else 0.0,
        'adv_per_100':    (adv_count  / total_tokens * 100) if total_tokens > 0 else 0.0,
        'past_tense_ratio': past_verbs / verb_count  if verb_count   > 0 else 0.0,
        'noun_density':    noun_count  / total_tokens if total_tokens > 0 else 0.0,
    }


def compute_subtree_depth(token):
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(compute_subtree_depth(child) for child in children)


def compute_tree_depth(text):
    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        roots = [t for t in sent if t.head == t]
        if roots:
            depths.append(compute_subtree_depth(roots[0]))
    return np.mean(depths) if depths else 0.0


def compute_punctuation_features(text):
    word_count = len(text.split())
    scale = 1000 / word_count if word_count > 0 else 0
    return {
        'comma_rate':       text.count(',')                               * scale,
        'semicolon_rate':   text.count(';')                               * scale,
        'colon_rate':       text.count(':')                               * scale,
        'emdash_rate':      (text.count('\u2014') + text.count('--'))     * scale,
        'exclamation_rate': text.count('!')                               * scale,
        'question_rate':    text.count('?')                               * scale,
    }


def compute_readability_features(text):
    return {
        'fk_grade':     textstat.flesch_kincaid_grade(text),
        'reading_ease': textstat.flesch_reading_ease(text),
    }


def extract_all_features(text, label, text_id):
    """Run all feature extractors and return a single flat dict."""
    row = {'label': label, 'text_id': text_id}
    row['ttr']   = compute_ttr(text)
    row['mattr'] = compute_mattr(text)
    row.update(compute_pos_features(text))
    row['tree_depth'] = compute_tree_depth(text)
    row.update(compute_punctuation_features(text))
    row.update(compute_readability_features(text))
    return row
