import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PlantDiseaseRAG:
    """
    Retrieval-Augmented Generation (RAG) Engine for Plant Pathology.
    Uses TF-IDF Vector Space Embeddings + Cosine Similarity over the 38-class
    PlantVillage Agricultural Knowledge Base.
    """
    def __init__(self, kb_path="knowledge_base.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, kb_path)
        with open(full_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

        self.class_keys = list(self.kb.keys())
        self.documents = []
        for k in self.class_keys:
            item = self.kb[k]
            doc = (
                f"Class: {k}. Crop: {item['crop']}. Disease: {item['disease']}. "
                f"Pathogen: {item['pathogen']}. Symptoms: {item['symptoms']}. "
                f"Causes: {item['causes']}. Organic Treatment: {item['organic_treatment']}. "
                f"Chemical Treatment: {item['chemical_treatment']}. Prevention: {item['prevention']}."
            )
            self.documents.append(doc)

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve_by_class(self, class_name):
        return self.kb.get(class_name, self.kb[self.class_keys[0]])

    def query(self, user_question, active_class=None, top_k=3):
        query_text = user_question
        if active_class and active_class in self.kb:
            query_text = f"{active_class.replace('___', ' ')} {user_question}"

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]

        retrieved_docs = []
        for idx in top_indices:
            key = self.class_keys[idx]
            retrieved_docs.append({
                "class_key": key,
                "score": round(float(sims[idx]), 4),
                "data": self.kb[key]
            })

        primary = self.kb[active_class] if (active_class and active_class in self.kb) else retrieved_docs[0]["data"]
        q_lower = user_question.lower()

        # Synthesize grounded response from retrieved context
        parts = [f"**RAG Grounded Diagnosis ({primary['disease']}):**"]
        if any(w in q_lower for w in ["symptom", "look", "sign", "spot", "identify"]):
            parts.append(f"• **Clinical Symptoms:** {primary['symptoms']}")
        if any(w in q_lower for w in ["treat", "cure", "medicine", "spray", "fix", "fungicide", "organic", "chemical"]):
            parts.append(f"• **Organic Control:** {primary['organic_treatment']}")
            parts.append(f"• **Chemical Control:** {primary['chemical_treatment']}")
        if any(w in q_lower for w in ["prevent", "avoid", "stop", "future", "cause", "why"]):
            parts.append(f"• **Etiology / Causes:** {primary['causes']}")
            parts.append(f"• **Prevention Protocol:** {primary['prevention']}")
        if len(parts) == 1:
            parts.append(f"• **Pathogen:** {primary['pathogen']}")
            parts.append(f"• **Symptoms:** {primary['symptoms']}")
            parts.append(f"• **Organic Remedy:** {primary['organic_treatment']}")
            parts.append(f"• **Chemical Remedy:** {primary['chemical_treatment']}")
            parts.append(f"• **Prevention:** {primary['prevention']}")

        parts.append(f"• **UN-SDG Alignment:** {primary['sdg_impact']}")
        return {
            "answer": "\n".join(parts),
            "retrieved_sources": retrieved_docs
        }
