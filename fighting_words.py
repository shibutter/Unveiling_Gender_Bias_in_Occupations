import spacy
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
import numpy as np

exclude = set(string.punctuation)

def basic_sanitize(in_string):
    '''Returns a very roughly sanitized version of the input string.'''  
    in_string = ''.join([ch for ch in in_string if ch not in exclude])
    in_string = in_string.lower()
    in_string = ' '.join(in_string.split())
    return in_string

def bayes_compare_language(l1, l2, ngram = 1, prior=.01, cv = None):
    '''
    Arguments:
    - l1, l2; a list of strings from each language sample
    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
    2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    if cv is None and type(prior) is not float:
        print("If using a non-uniform prior:")
        print("Please also pass a count vectorizer with the vocabulary parameter set.")
        quit()
    l1 = [basic_sanitize(l) for l in l1]
    l2 = [basic_sanitize(l) for l in l2]
    if cv is None:
        cv = CV(decode_error = 'ignore', min_df = 1, max_df = 1.0, ngram_range=(1,ngram),
                binary = False,
                max_features = 15000)
    counts_mat = cv.fit_transform(l1+l2).toarray()
    # Now sum over languages...
    vocab_size = len(cv.vocabulary_)
    print("Vocab size is {}".format(vocab_size))
    if type(prior) is float:
        priors = np.array([prior for i in range(vocab_size)])
    else:
        priors = prior
    z_scores = np.empty(priors.shape[0])
    count_matrix = np.empty([2, vocab_size], dtype=np.float32)
    count_matrix[0, :] = np.sum(counts_mat[:len(l1), :], axis = 0)
    count_matrix[1, :] = np.sum(counts_mat[len(l1):, :], axis = 0)
    a0 = np.sum(priors)
    n1 = 1.*np.sum(count_matrix[0,:])
    n2 = 1.*np.sum(count_matrix[1,:])
    print("Comparing language...")
    for i in range(vocab_size):
        #compute delta
        term1 = np.log((count_matrix[0,i] + priors[i])/(n1 + a0 - count_matrix[0,i] - priors[i]))
        term2 = np.log((count_matrix[1,i] + priors[i])/(n2 + a0 - count_matrix[1,i] - priors[i]))        
        delta = term1 - term2
        #compute variance on delta
        var = 1./(count_matrix[0,i] + priors[i]) + 1./(count_matrix[1,i] + priors[i])
        #store final score
        z_scores[i] = delta/np.sqrt(var)
    index_to_term = {v:k for k,v in cv.vocabulary_.items()}
    sorted_indices = np.argsort(z_scores)
    return_list = []
    for i in sorted_indices:
        return_list.append((index_to_term[i], z_scores[i]))
    return return_list

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

    male_verbs_adjectives = [get_verbs_adjectives(basic_sanitize(row['Story']), row['Occupation'], nlp) for _, row in male_data.iterrows()]
    female_verbs_adjectives = [get_verbs_adjectives(basic_sanitize(row['Story']), row['Occupation'], nlp) for _, row in female_data.iterrows()]

    male_verbs = [verb for verbs, _ in male_verbs_adjectives for verb in verbs]
    female_verbs = [verb for verbs, _ in female_verbs_adjectives for verb in verbs]
    male_adjectives = [adj for _, adjs in male_verbs_adjectives for adj in adjs]
    female_adjectives = [adj for _, adjs in female_verbs_adjectives for adj in adjs]

    verbs_comparison = bayes_compare_language(male_verbs, female_verbs, ngram=1, prior=.01, cv=CV())
    adjectives_comparison = bayes_compare_language(male_adjectives, female_adjectives, ngram=1, prior=.01, cv=CV())

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
