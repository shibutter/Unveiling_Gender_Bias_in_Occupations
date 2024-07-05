import spacy
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer as CV
import fighting_words_py3 as fw

nlp = spacy.load('nl_core_news_lg')

def is_relevant_occupation_or_pronoun(token, occupation_lower):
    return (occupation_lower in token.text.lower() or
            occupation_lower in token.lemma_.lower() or
            token.text.lower() in ['hij', 'zij'])

def get_verbs_adjectives(text, occupation, nlp):
    doc = nlp(text)
    occupation_lower = occupation.lower()
    verbs = []
    adjectives = []

    for token in doc:
        if is_relevant_occupation_or_pronoun(token, occupation_lower):
            for sent_token in token.sent:
                if sent_token.head == token or any(child == token for child in sent_token.head.children):
                    if sent_token.pos_ == 'VERB':
                        verbs.append(sent_token.lemma_)
                    elif sent_token.pos_ == 'ADJ':
                        adjectives.append(sent_token.lemma_)
    return verbs, adjectives

def get_fighting_words(df, nlp):
    male_data = df[df['Gender_ENG'] == 'Male']
    female_data = df[df['Gender_ENG'] == 'Female']

    male_verbs_adjectives = [get_verbs_adjectives(fw.basic_sanitize(row['Story']), row['Occupation'], nlp) for _, row in male_data.iterrows()]
    female_verbs_adjectives = [get_verbs_adjectives(fw.basic_sanitize(row['Story']), row['Occupation'], nlp) for _, row in female_data.iterrows()]

    male_verbs = [verb for verbs, _ in male_verbs_adjectives for verb in verbs]
    female_verbs = [verb for verbs, _ in female_verbs_adjectives for verb in verbs]
    male_adjectives = [adj for _, adjs in male_verbs_adjectives for adj in adjs]
    female_adjectives = [adj for _, adjs in female_verbs_adjectives for adj in adjs]

    verbs_comparison = fw.bayes_compare_language(male_verbs, female_verbs, ngram=1, prior=.01, cv=CV())
    adjectives_comparison = fw.bayes_compare_language(male_adjectives, female_adjectives, ngram=1, prior=.01, cv=CV())

    return verbs_comparison, adjectives_comparison

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item, zscore in data:
            writer.writerow([item, zscore])

def process_and_save(data_path, prefix, nlp):
    df = pd.read_csv(data_path, sep=';', encoding='ISO-8859-1')
    verbs, adjectives = get_fighting_words(df, nlp)
    save_to_csv(verbs, f'{prefix}_verbs.csv')
    save_to_csv(adjectives, f'{prefix}_adj.csv')

# datasets = {
