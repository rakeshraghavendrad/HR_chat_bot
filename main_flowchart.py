from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from IPython.display import display, HTML, Image
from graphviz import Digraph
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import warnings
import os
from neo4j import GraphDatabase
import spacy
import time

warnings.filterwarnings('ignore')

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

# Define the context for your prompt
context = "This directory contains multiple documents providing examples and solutions for various Human resource tasks within an organization."

# Data ingestion: load all files from a directory
directory_path = "sample_folder"
reader = SimpleDirectoryReader(input_dir=directory_path)
documents = reader.load_data()

# Load spacy model
nlp = spacy.load("en_core_web_sm")

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

def create_flowchart(text_content):
    dot = Digraph(comment='Process Flowchart')
    dot.attr(rankdir='TB')
    
    # Set global node and edge styles
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', 
            fontname='Arial', margin='0.3', height='0.6', width='2')
    dot.attr('edge', fontname='Arial', arrowsize='0.8')
    
    # Process text content to extract steps
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    
    # Initialize variables
    node_counter = 0
    nodes = {}  # Dictionary to store node references
    last_node = None
    decision_stack = []
    
    # Add start node
    start_node = f'node_{node_counter}'
    dot.node(start_node, 'Start', shape='oval', fillcolor='lightgreen')
    nodes['start'] = start_node
    last_node = start_node
    node_counter += 1

    for line in lines:
        if not line or line.startswith('Note:'): continue
        
        # Remove numbering if present
        line = line.split('.', 1)[-1].strip()
        current_node = f'node_{node_counter}'
        
        if '?' in line:  # Decision node
            # Create decision diamond
            dot.node(current_node, line, shape='diamond', fillcolor='lightyellow')
            if last_node:
                dot.edge(last_node, current_node)
            decision_stack.append((current_node, node_counter))  # Store both node id and counter
            last_node = current_node
            nodes[f'decision_{node_counter}'] = current_node
        
        elif '→' in line:  # Decision outcome
            parts = line.split('→')
            answer = parts[0].strip()
            action = parts[1].strip()
            
            if decision_stack:
                decision_node, dec_counter = decision_stack[-1]
                
                # Create action node for this branch
                action_node = f'node_{node_counter}'
                dot.node(action_node, action)
                dot.edge(decision_node, action_node, label=answer)
                
                nodes[f'action_{dec_counter}_{answer.lower()}'] = action_node
                
                # If this is a 'No' branch, pop the decision from stack
                if answer.lower() == 'no':
                    decision_stack.pop()
                
                last_node = action_node
            
        else:  # Regular process node
            dot.node(current_node, line)
            if last_node:
                # Connect to the previous node if it's not part of a decision branch
                if not decision_stack:
                    dot.edge(last_node, current_node)
                # If it's part of a decision branch, connect to the last action node
                else:
                    dot.edge(last_node, current_node)
            last_node = current_node
            nodes[f'process_{node_counter}'] = current_node
        
        node_counter += 1
    
    # Add end node
    end_node = f'node_{node_counter}'
    dot.node(end_node, 'End', shape='oval', fillcolor='lightgreen')
    
    # Connect any dangling nodes to the end node
    terminal_nodes = set()
    for node in nodes.values():
        # Check if this node has any outgoing edges
        has_outgoing = False
        for edge in dot.body:
            if edge.startswith(f'{node} ->'):
                has_outgoing = True
                break
        if not has_outgoing and node != end_node:
            terminal_nodes.add(node)
    
    # Connect terminal nodes to end node
    for node in terminal_nodes:
        dot.edge(node, end_node)
    
    # Graph styling
    dot.attr(dpi='300')
    dot.attr(size='8,8!')
    dot.attr(splines='ortho')
    
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = int(time.time())
    flowchart_file = os.path.join('static', f'flowchart_{timestamp}')
    
    # Render the flowchart
    dot.render(flowchart_file, format='png', cleanup=True)
    
    return f"/static/flowchart_{timestamp}.png"

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
    app.run(debug=True, port=5002)