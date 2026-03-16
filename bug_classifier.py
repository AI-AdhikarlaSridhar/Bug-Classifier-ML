import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class BugClassifierEngine:
    """
    Automated Bug Classification Engine for Industrial Software Engineering.
    Utilizes NLP techniques to route bug reports to the correct engineering departments.
    Inspired by research conducted at Ericsson.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('clf', LogisticRegression(solver='liblinear'))
        ])
        self.is_trained = False
        self.departments = ["Core-Network", "Cloud-Infra", "RAN-Systems", "User-Experience"]

    def train_model(self, bug_reports, assigned_depts):
        """
        Trains the classifier on historical bug report data.
        """
        self.pipeline.fit(bug_reports, assigned_depts)
        self.is_trained = True
        print("Bug Classification Model trained successfully.")

    def predict_department(self, bug_description):
        """
        Predicts the best department for a new incoming bug report.
        """
        if not self.is_trained:
            return "Error: Model must be trained before prediction."
        
        prediction = self.pipeline.predict([bug_description])[0]
        confidence = np.max(self.pipeline.predict_proba([bug_description]))
        
        return {
            "department": prediction,
            "confidence": f"{confidence:.2%}"
        }

if __name__ == "__main__":
    engine = BugClassifierEngine()
    # Mock industrial data
    reports = [
        "Network latency issues in core packet gateway",
        "Cloud orchestration failure in Kubernetes cluster",
        "Radio signal drop at high frequency bands",
        "UI lag in the management dashboard"
    ]
    depts = ["Core-Network", "Cloud-Infra", "RAN-Systems", "User-Experience"]
    
    engine.train_model(reports, depts)
    
    new_bug = "Unexpected disconnect in the packet core system"
    result = engine.predict_department(new_bug)
    print(f"Routing Result: {result}")
