# Install necessary libraries (uncomment if running in Colab)
# !pip install openai sklearn pandas matplotlib

import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# ==== DATA PREP ====
fashion_data = [
    {"name": "Boho Dress", "desc": "Flowy, earthy tones for festival vibes", "vibes": ["boho", "cozy"]},
    {"name": "Urban Jacket", "desc": "Sharply tailored, bold colors for city nights", "vibes": ["urban", "chic"]},
    {"name": "Sporty Hoodie", "desc": "Energetic, active wear in neon shades", "vibes": ["energetic", "sporty"]},
    {"name": "Vintage Jeans", "desc": "Retro comfort with faded blue feel", "vibes": ["retro", "casual"]},
    {"name": "Minimal Tee", "desc": "Clean lines, monochrome style", "vibes": ["minimal", "chic"]},
    {"name": "Cozy Sweater", "desc": "Soft, warm knit for lazy evenings", "vibes": ["cozy", "classic"]},
    {"name": "Party Skirt", "desc": "Sparkly, upbeat style for nightlife", "vibes": ["party", "chic"]},
    {"name": "Outdoor Boots", "desc": "Sturdy, practical for adventure", "vibes": ["adventurous", "rugged"]}
]
df = pd.DataFrame(fashion_data)

# ==== EMBEDDING ====
openai.api_key = "sk-..."  # <-- Insert your actual OpenAI API key (keep private!)

def get_embedding(text, model="text-embedding-ada-002"):
    # Defensive coding in real-world: retry logic, API fail handling (see below)
    try:
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.zeros(1536)  # Or correct dim for model used

# Embed product descriptions
df["embeddings"] = df["desc"].apply(get_embedding)

# ==== MATCHING FUNCTION ====
def vibe_matcher(query, product_df, threshold=0.7):
    query_emb = get_embedding(query)
    embedding_matrix = np.stack(product_df["embeddings"].values)
    sims = cosine_similarity([query_emb], embedding_matrix)[0]
    product_df = product_df.copy()  # Avoid modifying original DataFrame
    product_df["sim_score"] = sims
    top = product_df.sort_values("sim_score", ascending=False).head(3)
    # Edge fallback message
    if top.iloc[0]["sim_score"] < threshold:
        return "No strong vibe match found. Try similar keywords?", top[["name", "desc", "sim_score"]]
    return top[["name", "desc", "sim_score"]]

# ==== EVALUATION AND TIMING ====
queries = ["energetic urban chic", "cozy autumn", "retro party night"]
matches = []
latencies = []
for q in queries:
    start = timer()
    result = vibe_matcher(q, df)
    end = timer()
    matches.append(result)
    latencies.append(end - start)

# Logging matches above threshold
good_match_flags = []
for match in matches:
    # If fallback triggered, match[0] is string
    if isinstance(match, tuple):
        top = match[1]
        good_match_flags.append(top["sim_score"].max() > 0.7)
    elif isinstance(match, pd.DataFrame):
        good_match_flags.append(match["sim_score"].max() > 0.7)
    else:
        good_match_flags.append(False)

# ==== LATENCY VISUALIZATION ====
plt.figure(figsize=(6,4))
plt.plot(queries, latencies, marker="o")
plt.ylabel("Latency (seconds)")
plt.xlabel("Query")
plt.title("Vibe Matcher Latency per Query")
plt.grid(True)
plt.show()

# ==== REFLECTION ====
reflection = [
    "Integrate Pinecone or Weaviate for scalable recall, faster search, and persistence.",
    "Add multimodal support (product images, user profiles) for richer vibe context.",
    "Edge cases handled: fallback prompt if max(sim_score) < threshold; tag-based manual matches.",
    "Metrics logging (latency, good_match ratio) can be tracked in production.",
    "Custom vibe taxonomy: could let users select/define vibes for more granular matching."
]
print("\n--- Reflection & Improvements ---")
for bullet in reflection:
    print("-", bullet)
