from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import warnings
import os
from neo4j import GraphDatabase
import spacy
import time
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
import textwrap
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ---- NEO4J SETUP ----
neo4j_uri = "neo4j+s://d22644da.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "7aidZZ4FP6_BS6QCuDLvba5aKq03bQ3Eaj43MlRysPc"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# ---- ENVIRONMENT VARIABLES ----
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- PROMPT TEMPLATE ----
prompt_template = """
Use the following pieces of information to answer the user's question.
If the information contains a table, include the table in the response.
If a flowchart is requested, format the response as follows:

1. Start each decision point with a question mark (?)
2. Put each step on a new line
3. Keep the steps concise and clear
4. Order the steps logically from start to end
5. Include all possible paths and use → to indicate flow direction
6. For decision outcomes, format as: Yes → [action] or No → [action]

Context: {context}
Graph Insights: {graph_insights}
Question: {question}

Answer appropriately based on the user's request.
"""

def create_flowchart(text_content):
    import math

    # Create a directed graph
    G = nx.DiGraph()
    
    # Filter out introductory lines and empty lines
    lines = [line.strip() for line in text_content.split('\n') if line.strip() and not line.startswith("Here is the flowchart")]
    
    # Initialize variables
    node_counter = 0
    nodes = {}
    last_node = None
    decision_stack = []
    positions = {}
    current_x, current_y = 0, 0
    level_width = 4.0  # Reduced horizontal spacing
    level_height = 2.0  # Reduced vertical spacing
    decision_offset = level_width / 2  # Offset for decision branches

    # Add start node
    start_node = f'node_{node_counter}'
    G.add_node(start_node, label='Start', node_type='terminal')
    positions[start_node] = (current_x, current_y)
    last_node = start_node
    node_counter += 1
    current_y -= level_height

    for line in lines:
        if not line or line.startswith('Note:'):
            continue

        line = line.split('.', 1)[-1].strip()
        current_node = f'node_{node_counter}'
        wrapped_label = '\n'.join(textwrap.wrap(line, width=30))  # Reduced width

        if '?' in line:  # Decision node
            G.add_node(current_node, label=wrapped_label, node_type='decision')
            positions[current_node] = (current_x, current_y)
            if last_node:
                G.add_edge(last_node, current_node)
            decision_stack.append((current_node, current_x, current_y))
            last_node = current_node
            current_y -= level_height
        elif '→' in line:  # Decision outcome
            parts = line.split('→')
            answer, action = parts[0].strip(), parts[1].strip()
            wrapped_action = '\n'.join(textwrap.wrap(action, width=30))

            if decision_stack:
                decision_node, decision_x, decision_y = decision_stack[-1]

                # Adjust positioning based on Yes/No
                if answer.lower() == 'yes':
                    current_x = decision_x + decision_offset
                    current_y = decision_y - level_height
                else:  # No path
                    current_x = decision_x - decision_offset
                    current_y = decision_y - level_height

                current_node = f'node_{node_counter}'
                G.add_node(current_node, label=wrapped_action, node_type='process')
                positions[current_node] = (current_x, current_y)
                G.add_edge(decision_node, current_node, label=answer)

                if answer.lower() == 'no':
                    decision_stack.pop()

            last_node = current_node
            current_y -= level_height
        else:  # Regular process node
            G.add_node(current_node, label=wrapped_label, node_type='process')
            positions[current_node] = (current_x, current_y)
            if last_node:
                G.add_edge(last_node, current_node)
            last_node = current_node
            current_y -= level_height
        node_counter += 1

    # Add end node
    end_node = f'node_{node_counter}'
    G.add_node(end_node, label='End', node_type='terminal')
    positions[end_node] = (current_x, current_y - level_height)
    for node in [n for n in G.nodes() if G.out_degree(n) == 0 and n != end_node]:
        G.add_edge(node, end_node)

    # Calculate figure size dynamically
    pos_array = np.array(list(positions.values()))
    x_min, y_min = pos_array.min(axis=0)
    x_max, y_max = pos_array.max(axis=0)
    width = max(15, (x_max - x_min) * 1.5)
    height = max(15, -(y_max - y_min) * 1.2)

    # Plot flowchart
    plt.figure(figsize=(width, height), dpi=150, facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # Define node colors
    node_colors = {
        'decision': '#FFD966',  # Yellow
        'terminal': '#93C47D',  # Green
        'process': '#6FA8DC',  # Blue
    }

    # Draw nodes with varying styles
    for node_type in ['terminal', 'process', 'decision']:
        node_list = [node for node in G.nodes() if G.nodes[node].get('node_type') == node_type]
        nx.draw_networkx_nodes(G, positions, nodelist=node_list,
                               node_color=node_colors[node_type],
                               node_size=8000,
                               node_shape='o' if node_type == 'terminal' else 'd' if node_type == 'decision' else 's',
                               edgecolors='black')

    # Draw edges with reduced clutter
    nx.draw_networkx_edges(G, positions, arrowstyle='->', arrowsize=20,
                           edge_color='gray', width=2)

    # Draw node and edge labels
    nx.draw_networkx_labels(G, positions, labels=nx.get_node_attributes(G, 'label'),
                            font_size=10, font_weight='bold', font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, positions, edge_labels=nx.get_edge_attributes(G, 'label'),
                                 font_size=9, bbox=dict(facecolor='white', edgecolor='none', pad=2))

    plt.axis('off')
    plt.tight_layout()

    # Save and return flowchart path
    os.makedirs('static', exist_ok=True)
    unique_id = str(uuid.uuid4())[:8]
    flowchart_path = f'static/flowchart_{unique_id}.png'
    plt.savefig(flowchart_path, bbox_inches='tight', dpi=300)
    plt.close()
    return f"/static/flowchart_{unique_id}.png"



def populate_graph(documents, driver, nlp):
    with driver.session() as session:
        for doc in documents:
            doc_text = doc.text
            nlp_doc = nlp(doc_text)
            concepts = [ent.text for ent in nlp_doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

            for concept in concepts:
                session.run("MERGE (:Concept {name: $concept})", concept=concept)

            for i, concept in enumerate(concepts[:-1]):
                next_concept = concepts[i + 1]
                session.run(
                    """
                    MATCH (c1:Concept {name: $concept}), (c2:Concept {name: $next_concept})
                    MERGE (c1)-[:RELATED_TO]->(c2)
                    """,
                    concept=concept, next_concept=next_concept
                )

def get_graph_insights(question):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($question)
            OPTIONAL MATCH (c)-[r:RELATED_TO]->(other:Concept)
            RETURN c.name AS concept, collect(other.name) AS related_concepts
            """,
            question=question
        )
        insights = []
        for record in result:
            insights.append(f"Concept: {record['concept']}, Related Concepts: {', '.join(record['related_concepts'])}")
        return "\n".join(insights) if insights else "No relevant graph insights found."

# Define the context for your prompt
context = "This directory contains multiple documents providing examples and solutions for various Human resource tasks within an organization."

# Data ingestion: load all files from a directory
directory_path = "sample_folder"
reader = SimpleDirectoryReader(input_dir=directory_path)
documents = reader.load_data()

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Initialize the application
populate_graph(documents, driver, nlp)
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

# Set up embedding model and LLM
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

from llama_index.core import Settings
Settings.embed_model = embed_model
Settings.llm = llm

# Create and persist the vector store index
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, node_parser=nodes)
vector_index.storage_context.persist(persist_dir="./storage_mini")

# Load the index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
index = load_index_from_storage(storage_context)

# Flask app setup
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Query the index
        graph_insights = get_graph_insights(question)
        query_prompt = prompt_template.format(
            context=context,
            graph_insights=graph_insights,
            question=question
        )
        
        response = index.as_query_engine(service_context=storage_context).query(query_prompt)
        response_text = response.response if hasattr(response, 'response') else 'No response generated.'
        
        flowchart_path = None
        if "flowchart" in question.lower():
            flowchart_path = create_flowchart(response_text)
            if not os.path.exists(os.path.join('static', os.path.basename(flowchart_path))):
                return jsonify({'error': 'Failed to generate flowchart'}), 500
        
        return jsonify({
            'response_text': response_text,
            'flowchart_path': flowchart_path
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
