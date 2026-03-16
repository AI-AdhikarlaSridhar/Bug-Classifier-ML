# Bug-Classifier-ML ðŸ› ï¸ðŸž
**Automated Bug Classification & Routing Framework**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine-Learning-green.svg)](https://github.com/AI-AdhikarlaSridhar)
[![AI](https://img.shields.io/badge/AI-Engineer-orange.svg)](https://github.com/AI-AdhikarlaSridhar)

## **Project Overview**
**Bug-Classifier-ML** is an industrial-scale framework designed to automate the classification and routing of bug reports using advanced **Natural Language Processing (NLP)** and **Machine Learning**. 

This project is inspired by my research and work at **Ericsson**, where I implemented predictive models to streamline bug triage processes, reducing manual intervention and increasing response efficiency.

---

## **Key Architecture**
The framework consists of two core analytical modules:
1.  **`BugClassifierEngine`**: A Supervised Learning pipeline (TF-IDF + Logistic Regression) that predicts the target department for new incoming bug reports based on their textual descriptions.
2.  **`BugTopicModeling`**: An Unsupervised Learning module using **Latent Dirichlet Allocation (LDA)** to discover recurring issue themes and hidden topics in large-scale bug report corpora.

---

## **Features**
-   ðŸ“ˆ **Automated Routing**: Predict the engineering department for any bug report with high confidence.
-   ðŸ”¡ **NLP Optimized**: Leverages TF-IDF vectorization and industrial-grade preprocessing.
-   ðŸ§  **Unsupervised Theme Discovery**: Uncover structural trends in bugs through LDA topic modeling.
-   ðŸ­ **Industrial Scalability**: Designed for integration into large-scale Jira or similar triage workflows.

---

## **Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/AI-AdhikarlaSridhar/Bug-Classifier-ML.git
cd Bug-Classifier-ML
pip install -r requirements.txt
```

---

## **Example Usage**

```python
from bug_classifier import BugClassifierEngine
from topic_modeling import BugTopicModeling

# 1. Predictive Bug Routing
engine = BugClassifierEngine()
engine.train_model(["network latency in gateway"], ["Core-Network"])
result = engine.predict_department("Unexpected packet drop in packet gateway")
print(f"Routing Department: {result['department']} (Confidence: {result['confidence']})")

# 2. Topic Discovery
model = BugTopicModeling(n_topics=2)
topics = model.extract_topics(["Memory leak in service", "Orchestration service crash"])
for t in topics:
    print(f"Topic {t['topic_id']}: {', '.join(t['words'])}")
```

---

## **Why This Project?**
In large-scale software environments like Ericsson, manual bug triage is a significant bottleneck. This project showcases how **Machine Learning** can be applied to "Meta-Engineering"â€”the art of using AI to improve the software engineering process itself. By combining **NLP** with **Statistics**, we can create self-optimizing development lifecycles.

---

## **Connect & Contribute**
I am always open to discussing AI for software engineering, predictive analytics, and ML industrialization.

-   **LinkedIn:** [Adhikarla Sridhar](https://www.linkedin.com/in/adhikarla-sridhar-7212689b/)
-   **Portfolio:** [AI-AdhikarlaSridhar](https://github.com/AI-AdhikarlaSridhar)

---

> "Automating the software lifecycle through intelligent data science."

---
*Disclaimer: This framework is a conceptual demonstration based on published research and industrial practices.*
