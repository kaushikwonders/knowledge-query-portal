import os
import uuid
import mysql.connector
import sqlite3
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_from_directory,
    redirect,
    url_for
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import traceback

# LangChain imports for PDFs
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# LangChain imports for Tabular data
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = "YOUR_FLASK_SECRET_KEY"

# ---------------------------
# Database Configuration (MySQL)
# ---------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "mydb")

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

# ---------------------------
# Folder to store uploaded files
# ---------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------
# Chain Memory Dictionaries
# ---------------------------
# For PDF document chains
CHAINS = {}
# For Tabular data (Excel/CSV) chains
TABLE_CHAINS = {}


# ---------------------------
# Helper Functions (for PDFs)
# ---------------------------


def sanitize_table_name(filename):
    name, _ = os.path.splitext(filename)
    name = name.lower()
    # Replace any non-alphanumeric character with an underscore
    name = re.sub(r'\W+', '_', name)
    return name

def get_pdf_text(pdf_paths):
    combined_text = ""
    for path in pdf_paths:
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text() or ""
                combined_text += text + "\n"
    return combined_text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# ---------------------------
# Access Control Helper
# ---------------------------
def can_user_access_document(viewer_user_id, doc_uploader_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql_viewer = """
    SELECT access_level, department_id, project_id 
    FROM users 
    WHERE id = %s
    """
    cursor.execute(sql_viewer, (viewer_user_id,))
    viewer = cursor.fetchone()
    if not viewer:
        conn.close()
        return False
    if viewer["access_level"] == "admin":
        conn.close()
        return True
    sql_uploader = """
    SELECT department_id, project_id FROM users WHERE id = %s
    """
    cursor.execute(sql_uploader, (doc_uploader_id,))
    uploader = cursor.fetchone()
    conn.close()
    if not uploader:
        return False
    if viewer["access_level"] == "department":
        return (viewer["department_id"] is not None and 
                viewer["department_id"] == uploader["department_id"])
    if viewer["access_level"] == "project":
        return (viewer["project_id"] is not None and
                viewer["project_id"] == uploader["project_id"])
    return False

# ---------------------------
# Authentication Utilities
# ---------------------------
def login_required(f):
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT access_level FROM users WHERE id=%s", (session["user_id"],))
        row = cursor.fetchone()
        conn.close()
        if not row or row["access_level"] != "admin":
            return "Access Denied: Admins only", 403
        return f(*args, **kwargs)
    return wrapper

# -----------------------------------------------------------------------------
# Home Page
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# -----------------------------------------------------------------------------
# Login & Logout
# -----------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            email_id = request.form.get("email_id", "").strip()
            password = request.form.get("password", "").strip()
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            sql = "SELECT * FROM users WHERE email_id=%s AND password=%s"
            cursor.execute(sql, (email_id, password))
            user = cursor.fetchone()
            conn.close()
            if user:
                session["user_id"] = user["id"]
                session["access_level"] = user["access_level"]
                session["user_email"] = user["email_id"]
                if "session_id" not in session:
                    session["session_id"] = str(uuid.uuid4())
                return redirect(url_for("home"))
            else:
                return render_template("login.html", error="Invalid credentials")
        return render_template("login.html")
    except Exception as e:
        print("Error during login:", str(e))
        print(traceback.format_exc())
        return "An error occurred during login.", 500

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -----------------------------------------------------------------------------
# Register Employee (Admin Only)
# -----------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
@admin_required
def register():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM departments")
    departments = cursor.fetchall()
    cursor.execute("SELECT * FROM projects")
    projects = cursor.fetchall()
    if request.method == "POST":
        email_id = request.form.get("email_id", "").strip()
        password = request.form.get("password", "").strip()
        access_level = request.form.get("access_level", "").strip()
        department_id = request.form.get("department_id", None)
        project_id = request.form.get("project_id", None)
        if access_level == "admin":
            department_id = None
            project_id = None
        elif access_level == "department":
            project_id = None
        elif access_level == "project":
            pass
        sql = """
        INSERT INTO users (email_id, password, access_level, department_id, project_id)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(sql, (email_id, password, access_level, department_id, project_id))
            conn.commit()
        except Exception as e:
            conn.close()
            return f"Error registering user: {str(e)}", 400
        conn.close()
        return redirect(url_for("home"))
    conn.close()
    return render_template("register.html", departments=departments, projects=projects)

# -----------------------------------------------------------------------------
# Delete Employee (Admin Only)
# -----------------------------------------------------------------------------
@app.route("/delete_employee", methods=["GET", "POST"])
@admin_required
def delete_employee():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE access_level != 'admin'")
    all_users = cursor.fetchall()
    if request.method == "POST":
        user_id = request.form.get("user_id")
        if not user_id:
            conn.close()
            return "No user selected", 400
        try:
            cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
            conn.commit()
        except Exception as e:
            conn.close()
            return f"Error deleting user: {str(e)}", 400
        conn.close()
        return redirect(url_for("home"))
    conn.close()
    return render_template("delete_employee.html", all_users=all_users)

# -----------------------------------------------------------------------------
# UPLOAD PDFs (All employees)
# -----------------------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_page():
    if request.method == "POST":
        if 'pdf_files' not in request.files:
            return jsonify({"error": "No files part in request"}), 400
        files = request.files.getlist("pdf_files")
        if not files or files[0].filename == '':
            return jsonify({"error": "No selected files"}), 400
        user_id = session["user_id"]
        conn = get_db_connection()
        cursor = conn.cursor()
        for file in files:
            filename = secure_filename(file.filename)
            if filename:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                cursor.execute("INSERT INTO documents (filename, uploaded_by) VALUES (%s, %s)",
                               (filename, user_id))
        conn.commit()
        conn.close()
        return redirect(url_for('select_page'))
    return render_template("index.html")

# -----------------------------------------------------------------------------
# SELECT PDFs to Process (All employees; role-based filtering)
# -----------------------------------------------------------------------------
@app.route("/select", methods=["GET", "POST"])
@login_required
def select_page():
    if request.method == "POST":
        selected_files = request.form.getlist("selected_files")
        if not selected_files:
            return "No files selected", 400
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        pdf_paths = [os.path.join(app.config["UPLOAD_FOLDER"], fname) for fname in selected_files]
        raw_text = get_pdf_text(pdf_paths)
        if not raw_text.strip():
            return "No text could be extracted from selected PDFs.", 400
        chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(chunks)
        conversation_chain = get_conversation_chain(vectorstore)
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        sid = session['session_id']
        CHAINS[sid] = conversation_chain
        session['selected_files'] = selected_files
        conn.close()
        return redirect(url_for("ask_page"))
    user_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT d.id, d.filename, d.uploaded_by
        FROM documents d
        WHERE d.filename LIKE '%.pdf'
        ORDER BY d.id DESC
    """)
    all_docs = cursor.fetchall()
    conn.close()
    accessible_filenames = []
    for doc in all_docs:
        if can_user_access_document(user_id, doc["uploaded_by"]):
            accessible_filenames.append(doc["filename"])
    return render_template("select.html", pdf_files=accessible_filenames)

# -----------------------------------------------------------------------------
# ASK Questions for PDFs (All employees)
# -----------------------------------------------------------------------------
@app.route("/askpage", methods=["GET"])
@login_required
def ask_page():
    selected_files = session.get('selected_files', [])
    return render_template("ask.html", selected_files=selected_files)

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Question is required."}), 400
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400
    sid = session.get("session_id")
    if not sid or sid not in CHAINS:
        return jsonify({"error": "No chain found. Please select and process files first."}), 400
    chain = CHAINS[sid]
    response = chain({"question": question})
    answer = response.get("answer", "")
    chat_history = response.get("chat_history", [])
    chat_history_list = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "unknown"
        chat_history_list.append({"role": role, "content": msg.content})
    return jsonify({"answer": answer, "chat_history": chat_history_list})

# -----------------------------------------------------------------------------
# VIEW FILES (All employees; role-based filtering)
# -----------------------------------------------------------------------------
@app.route("/files")
@login_required
def list_files():
    user_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT d.id, d.filename, d.uploaded_by 
        FROM documents d
        WHERE d.filename LIKE '%.pdf'
        ORDER BY d.id DESC
    """)
    all_docs = cursor.fetchall()
    conn.close()
    accessible_docs = []
    for doc in all_docs:
        if can_user_access_document(user_id, doc["uploaded_by"]):
            accessible_docs.append(doc)
    return render_template("files.html", pdf_docs=accessible_docs)

@app.route("/view/<filename>")
@login_required
def view_file(filename):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM documents WHERE filename=%s", (filename,))
    doc = cursor.fetchone()
    conn.close()
    if not doc:
        return "File not found in DB", 404
    if not can_user_access_document(session["user_id"], doc["uploaded_by"]):
        return "Access Denied", 403
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -----------------------------------------------------------------------------
# UPLOAD Tabular Data (CSV/XLSX) -> Redirect to select_tabular
# -----------------------------------------------------------------------------
@app.route("/upload_tabular", methods=["GET", "POST"])
@login_required
def upload_tabular():
    if request.method == "POST":
        file = request.files.get("tabular_file")
        if not file or file.filename == "":
            return "No file selected", 400
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            return "Only CSV/XLSX allowed", 400
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Record the file in the documents table
        user_id = session["user_id"]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO documents (filename, uploaded_by) VALUES (%s, %s)", (filename, user_id))
        conn.commit()
        conn.close()

        # Instead of processing immediately, redirect to select_tabular page.
        return redirect(url_for("select_tabular"))
    return render_template("upload_tabular.html")

# -----------------------------------------------------------------------------
# SELECT Tabular Files to Process
# -----------------------------------------------------------------------------
@app.route("/select_tabular", methods=["GET", "POST"])
@login_required
def select_tabular():
    if request.method == "POST":
        selected_files = request.form.getlist("selected_tabular_files")
        # Retrieve the optional prompt template field.
        prompt_template = request.form.get("prompt_template", "").strip()
        
        if not selected_files:
            return "No files selected", 400
        
        # Create a new temporary SQLite DB file.
        db_filename = f"temp_{uuid.uuid4().hex}.db"
        db_path = os.path.join(app.config["UPLOAD_FOLDER"], db_filename)
        conn = sqlite3.connect(db_path)
        
        table_names = []  # List to hold the names of the tables created.
        for fname in selected_files:
            fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            ext = os.path.splitext(fname)[1].lower()
            # Read file using pandas.
            if ext == ".csv":
                df = pd.read_csv(fpath)
            elif ext == ".xlsx":
                df = pd.read_excel(fpath)
            else:
                continue
            # Generate a valid table name from the file name.
            table_name = sanitize_table_name(fname)
            table_names.append(table_name)
            # Write the dataframe to a table with the derived name.
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        
        # Build the LangChain SQLDatabaseChain using the SQLite DB.
        from langchain_community.utilities import SQLDatabase
        from langchain_experimental.sql import SQLDatabaseChain
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        llm = ChatOpenAI()
        chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        
        # Ensure the session_id exists.
        session_id = session.get("session_id")
        if not session_id:
            session["session_id"] = str(uuid.uuid4())
            session_id = session["session_id"]
        
        # Store the chain along with the selected file names, prompt template, and table names.
        TABLE_CHAINS[session_id] = {
            "db_path": db_path,
            "chain": chain,
            "selected_tabular_files": selected_files,
            "prompt_template": prompt_template,
            "table_names": table_names,
        }
        return redirect(url_for("ask_tabular"))
    else:
        # GET: List only CSV and XLSX files from the documents table, filtering with LOWER() for case-insensitivity.
        user_id = session["user_id"]
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT d.id, d.filename, d.uploaded_by
            FROM documents d
            WHERE LOWER(d.filename) LIKE '%.csv' OR LOWER(d.filename) LIKE '%.xlsx'
            ORDER BY d.id DESC
        """)
        all_docs = cursor.fetchall()
        conn.close()
        
        accessible_files = []
        for doc in all_docs:
            if can_user_access_document(user_id, doc["uploaded_by"]):
                accessible_files.append(doc["filename"])
        return render_template("select_tabular.html", tabular_files=accessible_files)





# -----------------------------------------------------------------------------
# ASK Tabular Questions -> generate SQL and return results
# -----------------------------------------------------------------------------
@app.route("/ask_tabular", methods=["GET"])
@login_required
def ask_tabular():
    session_id = session.get("session_id")
    table_info = TABLE_CHAINS.get(session_id)
    # If chain exists, extract selected file names; otherwise set to None.
    if table_info:
        selected_files = table_info.get("selected_tabular_files")
    else:
        selected_files = None
    return render_template("ask_tabular.html", filename=selected_files)

import json

@app.route("/ask_tabular_api", methods=["POST"])
@login_required
def ask_tabular_api():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    session_id = session.get("session_id")
    table_info = TABLE_CHAINS.get(session_id)
    if not table_info:
        return jsonify({"error": "No chain found. Upload tabular data first."}), 400

    # Build a default prompt that instructs the LLM to produce only a single SQL statement
    # and not to use markdown formatting.
    table_names = table_info.get("table_names", [])
    default_prompt = (
        "Generate a single SQL query that can be executed directly without markdown formatting. "
        "The SQL code should not have ``` at the beginning or end, and should not include the word 'sql' in the output. "
        "Use the following tables: " + ", ".join(table_names) + "."
    )

    prompt_template = table_info.get("prompt_template", "").strip()
    if prompt_template:
        full_question = default_prompt + "\n" + prompt_template + "\n" + question
    else:
        full_question = default_prompt + "\n" + question

    print("Full prompt sent to chain.run():")
    print(full_question)

    chain = table_info["chain"]
    try:
        answer = chain.run(full_question)
    except Exception as e:
        print("Chain error:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": f"Error generating query: {str(e)}"}), 400

    # Remove any Markdown formatting if present.
    if answer.startswith("```"):
        answer = answer.lstrip("`").strip()
        if answer.lower().startswith("sql"):
            answer = answer[3:].strip()
        if answer.endswith("```"):
            answer = answer[:-3].strip()

    # Split by semicolon in case multiple statements are returned.
    statements = answer.split(";")
    final_query = ""
    for stmt in statements:
        stmt = stmt.strip()
        if stmt:
            final_query = stmt
            break

    # Encode final_query to ensure no problematic characters.
    try:
        final_query = final_query.encode("utf-8", errors="replace").decode("utf-8")
    except Exception as e:
        print("Encoding error:", str(e))
        final_query = "Error encoding query."

    print("Final SQL Query to execute:")
    print(final_query)

    # Return a JSON response ensuring proper encoding.
    response_data = {"answer": final_query}
    try:
        response_json = json.dumps(response_data)
    except Exception as e:
        print("JSON encoding error:", str(e))
        response_json = '{"answer": "Error encoding JSON response."}'
    return app.response_class(
        response=response_json,
        status=200,
        mimetype='application/json'
    )





# -----------------------------------------------------------------------------
# VIEW TABULAR FILES (Excel/CSV)
# -----------------------------------------------------------------------------
@app.route("/files_tabular")
@login_required
def files_tabular():
    user_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT d.id, d.filename, d.uploaded_by 
        FROM documents d
        WHERE LOWER(d.filename) LIKE '%.csv' OR LOWER(d.filename) LIKE '%.xlsx'
        ORDER BY d.id DESC
    """)
    all_docs = cursor.fetchall()
    conn.close()
    accessible_docs = []
    for doc in all_docs:
        if can_user_access_document(user_id, doc["uploaded_by"]):
            accessible_docs.append(doc)
    return render_template("files_tabular.html", tabular_docs=accessible_docs)

@app.route("/view_tabular/<filename>")
@login_required
def view_tabular_file(filename):
    """
    Serves the tabular file (CSV/XLSX) from the uploads folder.
    Setting as_attachment=True forces a download.
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
