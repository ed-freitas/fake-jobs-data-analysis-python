import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data['job_description'] = data['job_description'].fillna("")
    data['requirements'] = data['requirements'].fillna("")
    return data

def add_features(data):
    data['description_length'] = data['job_description'].apply(len)
    data['word_count'] = data['job_description'].apply(lambda x: len(x.split()))
    data['avg_word_length'] = data['job_description'].apply(
        lambda x: sum(len(word) for word in x.split()) / (len(x.split()) or 1)
    )
    data['exclamation_count'] = data['job_description'].apply(lambda x: x.count('!'))
    data['question_count'] = data['job_description'].apply(lambda x: x.count('?'))
    data['uppercase_word_count'] = data['job_description'].apply(
        lambda x: sum(1 for word in x.split() if word.isupper())
    )
    
    keywords = ['work from home', 'earn', 'immediate start', 'no experience required', 'urgent', 'easy money']
    data['contains_generic_terms'] = data['job_description'].apply(
        lambda x: any(term in x.lower() for term in keywords)
    )
    data['keyword_density'] = data['job_description'].apply(
        lambda x: sum(word in x.lower() for word in keywords) / (len(x.split()) or 1)
    )
    data['contains_contact_info'] = data['job_description'].apply(
        lambda x: any(term in x.lower() for term in ['email', 'phone', 'contact'])
    )
    data['contains_url'] = data['job_description'].apply(lambda x: 'http' in x or 'www' in x)
    data['special_char_count'] = data['job_description'].apply(
        lambda x: sum(1 for char in x if char in ['$', '%', '&', '*'])
    )
    data['title_description_similarity'] = data.apply(
        lambda row: SequenceMatcher(None, row['job_title'], row['job_description']).ratio(), axis=1
    )
    return data

def flag_anomalies(data):
    data['is_short_description'] = data['description_length'] < 50
    data['is_low_word_count'] = data['word_count'] < 10
    data['has_excessive_exclamations'] = data['exclamation_count'] > 3
    data['has_excessive_special_chars'] = data['special_char_count'] > 5
    data['has_unusual_format'] = data['uppercase_word_count'] > 5
    data['has_high_keyword_density'] = data['keyword_density'] > 0.1
    data['has_contact_info'] = data['contains_contact_info']
    data['has_url'] = data['contains_url']
    data['is_high_similarity'] = data['title_description_similarity'] > 0.8
    data['has_generic_terms'] = data['contains_generic_terms']

    data['potentially_fake'] = (
        data['is_short_description'] |
        data['is_low_word_count'] |
        data['has_excessive_exclamations'] |
        data['has_excessive_special_chars'] |
        data['has_unusual_format'] |
        data['has_high_keyword_density'] |
        data['has_contact_info'] |
        data['has_url'] |
        data['is_high_similarity'] |
        data['has_generic_terms']
    ).astype(int)
    return data

def visualize_data(data):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='potentially_fake', data=data, palette="Set2")
    plt.title("Distribution of Potentially Fake vs. Real Job Posts")
    plt.xlabel("Job Post Classification (0: Real, 1: Potentially Fake)")
    plt.ylabel("Count")
    plt.show()

    anomaly_features = [
        'description_length', 'word_count', 'avg_word_length', 
        'exclamation_count', 'special_char_count', 'uppercase_word_count'
    ]
    for feature in anomaly_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='potentially_fake', y=feature, data=data, palette="Set3")
        plt.title(f"{feature.capitalize()} by Job Post Classification")
        plt.xlabel("Job Post Classification (0: Real, 1: Potentially Fake)")
        plt.ylabel(feature.capitalize())
        plt.show()

def save_dataset(data, output_file):
    data.to_csv(output_file, index=False)
    print(f"Dataset with rule-based classification saved to {output_file}.")

def main():
    file_path = "linkedin_job_posts.csv"
    output_file = "linkedin_job_posts_with_potentially_fake.csv"

    data = load_dataset(file_path)
    data = add_features(data)
    data = flag_anomalies(data)
    visualize_data(data)
    save_dataset(data, output_file)

if __name__ == "__main__":
    main()
