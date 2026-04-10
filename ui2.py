#Allllllllll done - presentable to Sandra and Team
import jinja2
from datetime import datetime

# Fix for 'strftime_now' is undefined error
def strftime_now(format="%Y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(format)

# Inject the function into the default Jinja2 environment
jinja2.defaults.DEFAULT_NAMESPACE['strftime_now'] = strftime_now

import json
import os
import re
import faiss
import pickle
import numpy as np
import gradio as gr
from llama_cpp import Llama
latest_cr_json = {}

# PATH Check
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))

def find_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing required file. Checked:\n{paths}")

INDEX_PATH = find_existing_path([
    os.path.join(PROJECT_ROOT, "gradioForEmbeddings", "outputVectors", "granite_cr_index.bin"),
    os.path.join(PROJECT_ROOT, "embeddingsWorks", "outputVectors", "granite_cr_index.bin"),
])

META_PATH = find_existing_path([
    os.path.join(PROJECT_ROOT, "gradioForEmbeddings", "outputVectors", "granite_cr_metadata.pkl"),
    os.path.join(PROJECT_ROOT, "embeddingsWorks", "outputVectors", "granite_cr_metadata.pkl"),
])

EMBED_MODEL = find_existing_path([
    os.path.join(PROJECT_ROOT, "models", "granite-embedding-278m-multilingual-Q4_K_M.gguf")
])

GEN_MODEL = find_existing_path([
    os.path.join(PROJECT_ROOT, "models", "llms", "granite-3.3-2b-instruct-Q4_K_M.gguf")
])

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading embedding model...")
embedder = Llama(model_path=EMBED_MODEL, embedding=True, verbose=False)

print("Loading generation model...")
generator = Llama(model_path=GEN_MODEL, n_ctx=4096, verbose=False)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# =====================================================
# CORE FUNCTIONS
# =====================================================
def embed_query(text):
    vec = embedder.create_embedding(text)
    return np.array(vec["data"][0]["embedding"], dtype=np.float32)


def retrieve_similar_crs(query, k=3):
    q_vec = embed_query(query).reshape(1, -1)
    distances, idx = index.search(q_vec, k)
    
    results = []
    for i in range(len(idx[0])):
        meta_index = idx[0][i]
        if meta_index < 0: continue
            
        raw_distance = float(distances[0][i])
        cr_data = metadata[meta_index].copy()
        
        # IMPROVED INTUITIVE SCORING:
        # We assume distance 100 = 100% match and 300 = 0% match.
        # This makes your 149 distance show up around 75-80%
        base_line = 300  # Distances above this are 0%
        perfect_line = 100 # Distances below this are 100%
        
        if raw_distance <= perfect_line:
            match_score = 100.0
        else:
            match_score = max(0, 100 - ((raw_distance - perfect_line) / (base_line - perfect_line) * 100))
        
        cr_data['similarity_distance'] = raw_distance
        cr_data['match_score'] = round(match_score, 2)
        results.append(cr_data)
        
    return results


def summarize_patterns(crs):
    txt = ""
    for i, cr in enumerate(crs):
        txt += f"""
CR {i+1} Pattern:
Category: {cr.get('category')}
Impact: {cr.get('impact')}
Urgency: {cr.get('urgency')}
Implementation style: {str(cr.get('implementation_plan'))[:200]}
Test strategy: {str(cr.get('test_plan'))[:150]}
Backout: {str(cr.get('backout_plan'))[:150]}
"""
    return txt

def build_prompt(user_query, crs):
    return f"""
You are CRIA - IBM Change Request Intelligent Assistant.

STRICT:
Professional enterprise tone.
5 to 8 implementation steps.
No markdown.

Similar CR patterns:
{summarize_patterns(crs)}

User change idea:
{user_query}

Return EXACT format:

Short Description:
Implementation Plan:
Risk and Impact Analysis:
Test Plan:
Backout Plan:
Justification:
"""

def parse_sections(text):
    def get(name, nxt):
        m = re.search(name + r":\s*(.*?)\s*(?=" + nxt + r":)", text, re.S | re.I)
        return m.group(1).strip() if m else "Not Available"

    return (
        get("Short Description","Implementation Plan"),
        get("Implementation Plan","Risk and Impact Analysis"),
        get("Risk and Impact Analysis","Test Plan"),
        get("Test Plan","Backout Plan"),
        get("Backout Plan","Justification"),
        text.split("Justification:")[-1].strip()
    )

# =====================================================
# REFERENCE CR HTML CARDS
# =====================================================
def build_reference_cards(crs):
    cards = ""
    for cr in crs:
        score = cr.get('match_score', 0)
        badge_color = "#24a148" if score > 70 else "#f1c21b" if score > 40 else "#da1e28"
        
        cards += f"""
<details class='ref-card'>
<summary>
    <div class='summary-header'>
        <span class='cr-number'>{cr.get('number','N/A')}</span>
        <span style='background:{badge_color}; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:bold;'>
            {score}% Match
        </span>
        <span class='cr-title'>{cr.get('short_description','No Title')}</span>
    </div>
    <div class='summary-meta'>Impact: {cr.get('impact','N/A')} | Urgency: {cr.get('urgency','N/A')}</div>
</summary>

<div class='ref-content'>
    <div class='ref-grid'>
        <div><label>Confidence</label><span>{score}% Match Profile</span></div>
        <div><label>Category</label><span>{cr.get('category','-')}</span></div>
        <div><label>Environment</label><span>{cr.get('u_env','-')}</span></div>
        <div><label>Type</label><span>{cr.get('type','-')}</span></div>
    </div>

    <div class='ref-section'><label>Implementation Plan</label><p>{cr.get('implementation_plan','-')}</p></div>
    <div class='ref-section'><label>Risk and Impact</label><p>{cr.get('risk_impact_analysis','-')}</p></div>
</div>
</details>
"""
    return cards
# =====================================================
# STREAMING AGENT PIPELINE
# =====================================================
def run_cria(query):
    retrieved = retrieve_similar_crs(query, 3)
    ref_html = build_reference_cards(retrieved)

    yield ref_html, "Generating...", "", "", "", "", ""

    prompt = build_prompt(query, retrieved)
    resp = generator.create_chat_completion(
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=700
    )

    output = resp["choices"][0]["message"]["content"]
    print("\n===== LLM RAW OUTPUT =====\n")
    print(output)
    print("\n==========================\n")

    parsed = parse_sections(output)

    # 🔥 STORE JSON (NEW)
    global latest_cr_json
    latest_cr_json = {
        "short_description": parsed[0],
        "implementation_plan": parsed[1],
        "risk_analysis": parsed[2],
        "test_plan": parsed[3],
        "backout_plan": parsed[4],
        "justification": parsed[5]
    }

    print("\n===== JSON OUTPUT =====\n")
    print(json.dumps(latest_cr_json, indent=2))
    print("\n=======================\n")
    
    
    cr_json = {
        "short_description": parsed[0],
        "implementation_plan": parsed[1],
        "risk_analysis": parsed[2],
        "test_plan": parsed[3],
        "backout_plan": parsed[4],
        "justification": parsed[5]
    }
    print("\n===== JSON OUTPUT =====\n")
    print(json.dumps(cr_json, indent=2))
    print("\n=======================\n")

    yield ref_html, *parsed


# =====================================================
# Build CR as per ServiceNow
# =====================================================
import requests

def create_change_request(cr_json):
    url = "https://dev223898.service-now.com/api/now/table/change_request"
    
    auth = ("cr_writer", "o:M-&evNQxX.4fE}D:mk:x@Dz?Te&G0!}2K*&TlC.jX-DOUOrRml=4t=")

    payload = {
        "short_description": cr_json["short_description"],

        # CORE FIELDS
        "implementation_plan": cr_json["implementation_plan"],
        "test_plan": cr_json["test_plan"],
        "backout_plan": cr_json["backout_plan"],
        "justification": cr_json["justification"],
        "risk_impact_analysis": cr_json["risk_analysis"],

        "category": "Software",
        "type": "normal",

        # REQUIRED / IMPORTANT FLAGS
        "risk": "2",
        "impact": "2",
        "priority": "3",
        "scope": "2",                 
        "production_system": "true",
        "cab_required": "false",
        "unauthorized": "false",

        # OPTIONAL BUT GOOD PRACTICE
        "justification": cr_json["justification"],
        "description": cr_json["short_description"]  # keep minimal
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(url, auth=auth, headers=headers, json=payload)

    print("\n===== SERVICENOW RESPONSE =====\n")
    print(response.status_code, response.text)
    print("\n================================\n")

    return response.json()

# =====================================================
# Pushes CR to ServiceNow
# =====================================================
def push_to_snow():
    global latest_cr_json

    if not latest_cr_json:
        return "❌ No CR generated yet"

    response = create_change_request(latest_cr_json)

    try:
        cr_number = response["result"]["number"]
        return f"✅ CR Created: {cr_number}"
    except:
        return f"❌ Failed: {response}"


# =====================================================
# ENTERPRISE AI CSS
# =====================================================
css = """
/* Global & Header styles */
/* Global & Header styles */
.ai-header {
    background: linear-gradient(135deg, #0f62fe 0%, #4facfe 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(15, 98, 254, 0.25);
}
.ai-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff !important;
}
.ai-sub {
    font-size: 15px;
    color: #e5f0ff !important;
    margin-top: 6px;
}


/* Base Card Styling */
.gradio-container .form {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
}

/* Reference Cards Styling */
.ref-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 16px;
    transition: box-shadow 0.2s ease;
}
.ref-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.ref-card summary {
    cursor: pointer;
    padding: 16px;
    list-style: none;
    outline: none;
    border-bottom: 1px solid transparent;
}
.ref-card[open] summary {
    border-bottom: 1px solid #e0e0e0;
    background: #fafafa;
}
.summary-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.cr-number {
    background: #e5f0ff;
    color: #0043ce;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.cr-title {
    font-weight: 600;
    font-size: 15px;
    color: #161616;
}
.summary-meta {
    font-size: 12px;
    color: #525252;
    margin-left: 2px;
}
.ref-content {
    padding: 16px;
    background: #ffffff;
}
.ref-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.ref-grid div {
    background: #f4f4f4;
    padding: 10px 12px;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
}
.ref-grid label {
    font-size: 11px;
    text-transform: uppercase;
    color: #525252;
    display: block;
    margin-bottom: 4px;
    font-weight: 600;
}
.ref-grid span {
    font-size: 13px;
    color: #161616;
}
.ref-section {
    margin-top: 16px;
}
.ref-section label {
    font-size: 13px;
    font-weight: 600;
    color: #161616;
    display: block;
    margin-bottom: 6px;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 4px;
}
.ref-section p {
    font-size: 13px;
    color: #393939;
    background: #fdfdfd;
    padding: 10px;
    border-left: 3px solid #d1d1d1;
    max-height: 150px;
    overflow-y: auto;
    white-space: pre-wrap;
    margin: 0;
}

/* Customizing Textareas to look cleaner */
textarea {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
"""

# =====================================================
# UI (DASHBOARD LAYOUT)
# =====================================================
with gr.Blocks(title="CRIA | Demo") as demo:

    gr.HTML("""
        <div class='ai-header'>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div class='ai-title'>CRIA — Change Record Intelligent Assistant</div>
                    <div class='ai-sub'></div>
                </div>
                <div>
                    <svg width="40" height="40" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16 2L2 9L16 16L30 9L16 2Z" fill="#ffffff"/>
                        <path d="M2 23L16 30L30 23V16L16 23L2 16V23Z" fill="#ffffff"/>
                    </svg>
                </div>
            </div>
        </div>
        """)
    with gr.Row():
        # LEFT COLUMN: Controls & Context
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("User Input")
                query = gr.Textbox(
                    label="Describe the Change",
                    placeholder="Example: Restart EQT TOR2 VSIs after PSU issue...",
                    lines=4
                )
                run_btn = gr.Button("Generate Change Record", variant="primary", size="lg")
            
            with gr.Accordion("📚 Referenced CRs (Historical Data)", open=True):
                ref_html = gr.HTML("<p style='color:#666; font-size:13px; padding:10px;'>Similar historical records will appear here after generation.</p>")

        # RIGHT COLUMN: Generated Output
        with gr.Column(scale=8):
            with gr.Group():
                gr.Markdown("✨ CRIA Generated Change Record")
                
                short_desc = gr.Textbox(label="Short Description")
                impl = gr.Textbox(label="Implementation Plan", lines=8)

                with gr.Row():
                    risk = gr.Textbox(label="Risk Analysis", lines=5)
                    test = gr.Textbox(label="Test Plan", lines=5)

                with gr.Row():
                    back = gr.Textbox(label="Backout Plan", lines=5)
                    just = gr.Textbox(label="Justification", lines=5)

            create_btn = gr.Button("🚀 Create CR in ServiceNow", variant="secondary")
            cr_result = gr.Textbox(label="ServiceNow CR Status")

    run_btn.click(
        fn=run_cria,
        inputs=query,
        outputs=[ref_html, short_desc, impl, risk, test, back, just]
    )
    create_btn.click(
        fn=push_to_snow,
        inputs=[],
        outputs=cr_result
    )

if __name__ == "__main__":
    # In Gradio 6.0, theme and css are passed to launch()
    demo.launch(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
