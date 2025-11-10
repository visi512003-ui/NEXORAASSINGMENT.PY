# Install necessary libraries (uncomment if running in Colab)
# !pip install openai sklearn pandas matplotlib

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from openai import OpenAI
import os

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
print("Fashion Products Dataset:")
print(df[['name', 'desc']].to_string(index=False))
print("\n" + "="*70 + "\n")

# ==== EMBEDDING ====
# Initialize OpenAI client (NEW API syntax)
client = OpenAI(api_key="sk-proj-VvkdomC9JAwiWMAGH2IxEqw1GzQvXzshC-TUZQCCl6bMb9d_oqNiLqxQ")  # <-- Your actual OpenAI API key
def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding using new OpenAI API syntax with error handling"""
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        # Return zero vector as fallback (dimension 1536 for text-embedding-3-small)
        return np.zeros(1536)

print("Generating embeddings for product descriptions...")
# Embed product descriptions
df["embeddings"] = df["desc"].apply(get_embedding)
print(f"Embeddings generated successfully! (Dimension: {len(df['embeddings'].iloc[0])})")
print("\n" + "="*70 + "\n")

# ==== MATCHING FUNCTION ====
def vibe_matcher(query, product_df, threshold=0.7):
    """Match query to products using cosine similarity"""
    print(f"Query: '{query}'")
    query_emb = get_embedding(query)
    embedding_matrix = np.stack(product_df["embeddings"].values)
    sims = cosine_similarity([query_emb], embedding_matrix)[0]
    product_df = product_df.copy()
    product_df["sim_score"] = sims
    top = product_df.sort_values("sim_score", ascending=False).head(3)
    
    # Edge fallback message
    if top.iloc[0]["sim_score"] < threshold:
        print(f"  No strong vibe match found (max score: {top.iloc[0]['sim_score']:.3f} < {threshold})")
        print("Try similar keywords?\n")
        return "No strong match", top[["name", "desc", "sim_score"]]
    
    return top[["name", "desc", "sim_score"]]

# ==== EVALUATION AND TIMING ====
print("Testing Vibe Matcher with 3 queries...\n")
queries = ["energetic urban chic", "cozy autumn", "retro party night"]
matches = []
latencies = []

for i, q in enumerate(queries, 1):
    print(f"\n--- Query {i}/{len(queries)} ---")
    start = timer()
    result = vibe_matcher(q, df)
    end = timer()
    latency = end - start
    
    matches.append(result)
    latencies.append(latency)
    
    # Display results
    if isinstance(result, tuple):
        top = result[1]
    else:
        top = result
    
    print(f"\nTop 3 Matches:")
    for idx, row in top.iterrows():
        print(f"  {row['name']:20s} | Score: {row['sim_score']:.3f} | {row['desc']}")
    print(f"\nLatency: {latency:.4f} seconds")
    print("="*70)

# Logging metrics
print("\n\n EVALUATION METRICS\n" + "="*70)
good_match_flags = []
for i, match in enumerate(matches):
    if isinstance(match, tuple):
        top = match[1]
        good_match_flags.append(top["sim_score"].max() > 0.7)
    elif isinstance(match, pd.DataFrame):
        good_match_flags.append(match["sim_score"].max() > 0.7)
    else:
        good_match_flags.append(False)

print(f"Queries with good matches (score > 0.7): {sum(good_match_flags)}/{len(queries)}")
print(f"Average latency: {np.mean(latencies):.4f} seconds")
print(f"Max latency: {np.max(latencies):.4f} seconds")
print(f"Min latency: {np.min(latencies):.4f} seconds")

# ==== LATENCY VISUALIZATION ====
plt.figure(figsize=(8,5))
plt.plot(range(1, len(queries)+1), latencies, marker="o", linewidth=2, markersize=8, color='#2E86AB')
plt.ylabel("Latency (seconds)", fontsize=12)
plt.xlabel("Query Number", fontsize=12)
plt.title("Vibe Matcher Latency per Query", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(queries)+1))
plt.tight_layout()
plt.show()

print("\n" + "="*70)

# ==== REFLECTION ====
reflection = [
    "Integrate Pinecone or Weaviate for scalable recall, faster search, and persistence.",
    "Add multimodal support (product images, user profiles) for richer vibe context.",
    "Edge cases handled: fallback prompt if max(sim_score) < threshold; zero-vector for API failures.",
    "Metrics logging (latency, good_match ratio) can be tracked in production with monitoring tools.",
    "Custom vibe taxonomy: could let users select/define vibes for more granular matching."
]

print("\n\n💡 REFLECTION & IMPROVEMENTS\n" + "="*70)
for i, bullet in enumerate(reflection, 1):
    print(f"{i}. {bullet}")

print("\n" + "="*70)
print(" Vibe Matcher Prototype Complete!")
print("="*70)


# ==== BONUS: OPENAI CHAT API DEMO ====
# Demonstrating the CORRECT way to use OpenAI's Chat Completions API
# (The provided code snippet had incorrect syntax)

def chat_demo():
    """
    Demo: OpenAI Chat Completions API with streaming
    Shows the CORRECT API usage (not the invalid 'responses.create' endpoint)
    """
    print("\n\n" + "="*70)
    print(" BONUS: OpenAI Chat API Demo (Streaming)")
    print("="*70 + "\n")
    
    try:
        # CORRECT: Use chat.completions.create (not responses.create)
        # CORRECT: Use a valid model like gpt-3.5-turbo or gpt-4 (not gpt-5)
        # CORRECT: Use 'messages' parameter (not 'input')
        
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Valid model (gpt-5 doesn't exist)
            messages=[              # Correct parameter name
                {
                    "role": "user",
                    "content": "Say 'double bubble bath' ten times fast."
                }
            ],
            stream=True  # Enable streaming responses
        )
        
        print("🗨️ GPT Response (streaming): ")
        full_response = ""
        
        # Stream chunks as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n\n" + "="*70)
        print("✅ Chat API Demo Complete!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error in chat demo: {e}")
        print("\nNote: This demo requires a valid OpenAI API key with chat access.")
        print("If you're getting rate limit errors, the key might be restricted.")

# Uncomment the line below to run the chat demo
# chat_demo()  # WARNING: This will consume API credits!

print("\n💡 TIP: To test the Chat API, uncomment line above and run the script again.")
print("⚠️  Note: Your API key might be restricted to embeddings only.\n")
