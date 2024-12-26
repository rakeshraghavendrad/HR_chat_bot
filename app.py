from flask import Flask, request, jsonify, render_template
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
If a flowchart is requested, generate a flowchart based on the information.

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

# Function to extract entities and relationships from documents
def populate_graph(documents, driver, nlp):
    with driver.session() as session:
        for doc in documents:
            doc_text = doc.text
            nlp_doc = nlp(doc_text)
            concepts = [ent.text for ent in nlp_doc.ents if ent.label_ == "ORG" or ent.label_ == "PRODUCT"]

            for concept in concepts:
                session.run("MERGE (:Concept {name: $concept})", concept=concept)

            for i, concept in enumerate(concepts):
                if i + 1 < len(concepts):
                    next_concept = concepts[i + 1]
                    session.run(
                        """
                        MATCH (c1:Concept {name: $concept}), (c2:Concept {name: $next_concept})
                        MERGE (c1)-[:RELATED_TO]->(c2)
                        """,
                        concept=concept, next_concept=next_concept
                    )

# Populate the Neo4j graph
populate_graph(documents, driver, nlp)

# Split the documents into nodes
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

# Set up embedding model and LLM
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

from llama_index.core import Settings

# Configure global settings
Settings.embed_model = embed_model
Settings.llm = llm

# Create and persist the vector store index
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, node_parser=nodes)
vector_index.storage_context.persist(persist_dir="./storage_mini")

# Load the index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
index = load_index_from_storage(storage_context)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for the frontend

# Function to extract insights from Neo4j
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

# Function to generate a flowchart from graph insights
def create_flowchart(graph_insights):
    dot = Digraph()
    dot.attr(rankdir='LR')

    insights = graph_insights.split('\n')
    for insight in insights:
        if 'Concept:' in insight and 'Related Concepts:' in insight:
            parts = insight.split(', Related Concepts: ')
            concept = parts[0].replace('Concept: ', '').strip()
            related_concepts = parts[1].split(', ')

            dot.node(concept, concept)
            for related in related_concepts:
                dot.node(related, related)
                dot.edge(concept, related)

    flowchart_file = "static/flowchart"
    dot.render(flowchart_file, format='png', cleanup=True)
    return f"{flowchart_file}.png"

# Flask route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')

    # Ensure a question is provided
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Query the index
    graph_insights = get_graph_insights(question)
    query_prompt = prompt_template.format(context=context, graph_insights=graph_insights, question=question)
    response = index.as_query_engine(service_context=storage_context).query(query_prompt)

    # Prepare response
    response_text = response.response if hasattr(response, 'response') else 'No response generated.'
    tables_html = ""

    # Render tables
    if hasattr(response, 'tables') and response.tables:
        for table in response.tables:
            rows = table.strip().split("\n")
            tables_html += "<table border='1' style='border-collapse: collapse;'>"
            for i, row in enumerate(rows):
                cells = row.split("|")
                tag = 'th' if i == 0 else 'td'
                tables_html += "<tr>" + "".join([f"<{tag} style='padding: 5px;'>{cell.strip()}</{tag}>" for cell in cells]) + "</tr>"
            tables_html += "</table>"

    # Check for flowchart generation
    flowchart_path = None
    if "flowchart" in question.lower() and graph_insights:
        flowchart_path = create_flowchart(graph_insights)

    return jsonify({
        'response_text': response_text,
        'tables_html': tables_html,
        'flowchart_path': flowchart_path
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
