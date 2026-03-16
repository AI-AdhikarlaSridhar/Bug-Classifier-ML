from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class BugTopicModeling:
    """
    Topic Modeling for Bug Reports using LDA.
    Enables unsupervised discovery of recurring issues in industrial software.
    """
    def __init__(self, n_topics=5):
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=42)

    def extract_topics(self, bug_descriptions):
        """
        Extracts dominant topics from a corpus of bug descriptions.
        """
        tf = self.tf_vectorizer.fit_transform(bug_descriptions)
        self.lda.fit(tf)
        
        feature_names = self.tf_vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append({"topic_id": topic_idx, "words": top_words})
        
        return topics

if __name__ == "__main__":
    model = BugTopicModeling(n_topics=2)
    # Simulated corpus
    corpus = [
        "Packet drop in the network gateway core network network",
        "Gateway core packet drop network issue gateway",
        "Memory leak in the orchestration service service",
        "Orchestration service memory leak service issue"
    ]
    extracted = model.extract_topics(corpus)
    for t in extracted:
        print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
