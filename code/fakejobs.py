import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data['job_description']  = data['job_description'].fillna("")
    data['requirements'] = data['requirements'].fillna("")
    return data

def add_features(data):
    data['description_length'] = data['job_description'].apply(len)
    data['exclamation_count']  = data['job_description'].apply(lambda x: x.count('!'))
    data['question_count'] = data['job_description'].apply(lambda x: x.count('?'))
    data['uppercase_word_count'] = data['job_description'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    data['contains_generic_terms'] = data['job_description'].apply(lambda x: any(term in x.lower() for term in ['work from home', 'earn', 'immediate start', 'no experience required']))
    return data

def flag_anomalies(data):
    data['is_short_description'] = data['description_length'] < 50
    data['has_excessive_exclamations'] = data['exclamation_count'] > 3
    data['has_unusual_format'] = data['uppercase_word_count'] > 5
    data['has_generic_terms'] = data['contains_generic_terms']
    data['potentially_fake'] = (
        data['is_short_description'] |
        data['has_excessive_exclamations'] |
        data['has_unusual_format'] |
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

    anomaly_features = ['description_length', 'exclamation_count', 'uppercase_word_count']
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