import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords

def plot_distribution(data, difficulty_mapping):
    fig, ax = plt.subplots()
    sns.countplot(x='difficulty', data=data, order=difficulty_mapping.values(), ax=ax)
    return fig

def generate_wordcloud(text):
    french_stopwords = set(stopwords.words('french'))
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=french_stopwords).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig
