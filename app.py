import streamlit as st
import pyodbc
import re
import sqlparse
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import io
import time

# Load environment variables from .env file
load_dotenv()

# Get API details from environment variables
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")            
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  
azure_api_version = os.getenv("API_VERSION")  

if not azure_openai_endpoint or not azure_openai_key or not azure_openai_deployment:
    raise ValueError("Missing Azure OpenAI credentials! Set them in the .env file.")

client = AzureOpenAI(
    api_key=azure_openai_key,  
    api_version=azure_api_version,
    azure_endpoint=azure_openai_endpoint
)

deployment_name = 'gpt-4o-mini'

def get_connection(server, database, username, password):
    return pyodbc.connect(f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}")

def extract_table_names(sql_code):
    tables = re.findall(r'\b(?:FROM|JOIN)\s+([\w\[\]]+\.)?[\w\[\]]+\b', sql_code, re.IGNORECASE)
    return list(set([t.strip() for t in tables if t]))

def extract_columns_from_query(query):
    """Improved column extraction with normalization to avoid duplicates."""
    patterns = [
        r'\bSELECT\s+((?:(?!\bFROM\b).)+)',  # Match everything in SELECT clause until FROM
        r'\bWHERE\s+([\w\.]+\s*=\s*[\w\.]+)',  # WHERE clause conditions
        r'\bGROUP BY\s+([\w\.]+)',  # GROUP BY columns
        r'\bORDER BY\s+([\w\.]+)',  # ORDER BY columns
        r'\bJOIN\s+[\w\.]+\s+ON\s+([\w\.]+\s*=\s*[\w\.]+)'  # JOIN conditions
    ]
    
    columns = set()
    column_alias_map = {}

    def normalize_column(column):
        # Remove alias or extra parts (AS, asc, desc, etc.)
        column = re.sub(r'\b(AS|asc|desc)\s+.+', '', column, flags=re.IGNORECASE).strip()
        if '.' in column:
            return column.split('.')[-1].strip('[]')
        return column.strip('[]')

    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        
        for match in matches:
            # If match is a tuple (for WHERE or JOIN, typically), iterate over the tuple
            if isinstance(match, tuple):
                for sub_match in match:
                    # Split by commas and clean up each part
                    parts = re.split(r'\s*,\s*', sub_match)
                    for part in parts:
                        column = normalize_column(part)
                        columns.add(column)
            else:
                # If match is a string (for SELECT, GROUP BY, etc.), split by commas directly
                parts = re.split(r'\s*,\s*', match)
                for part in parts:
                    column = normalize_column(part)
                    columns.add(column)

    # Remove duplicates and retain only one version of the column if both alias and non-alias are present
    final_columns = set()
    for col in columns:
        if col not in column_alias_map:
            column_alias_map[col] = True
        else:
            final_columns.add(col)

    return list(final_columns)

def generate_ai_documentation(proc_name, sql_code):
    prompt = f"""
    Analyze the following SQL stored procedure and provide detailed documentation:
    1. Tables used (list all tables explicitly)
    2. Columns used (list ALL columns with table prefixes)
    3. Transformations (joins, aggregations, filters)
    4. Business logic

    Use EXACTLY these section headers:
    ### Tables Used
    ### Columns Used
    ### Transformations
    ### Business Logic

    Stored Procedure: {proc_name}
    SQL Code: {sql_code}
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def parse_ai_response(ai_doc):
    """More robust parsing with strict section matching"""
    sections = {
        "Tables Used": [],
        "Columns Used": [],
        "Transformations": [],
        "Business Logic": []
    }
    
    current_section = None
    for line in ai_doc.split('\n'):
        line = line.strip()
        if line.startswith("### "):
            current_section = line[4:].strip().rstrip(':')
            if current_section not in sections:
                current_section = None
        elif current_section and line:
            if current_section == "Columns Used":
                sections[current_section].extend([col.strip() for col in line.split('- ')[1:] if col])
            else:
                sections[current_section].append(line)
    
    # Convert lists to strings
    return {k: '\n'.join(v) if isinstance(v, list) else v for k, v in sections.items()}

def format_documentation(proc_name, tables, columns, ai_doc, sp_query):
    """Enhanced formatting with merged AI and extracted data."""
    # Parse AI response
    ai_sections = parse_ai_response(ai_doc)
    
    # Ensure ai_sections['Columns Used'] is a list
    ai_columns = ai_sections['Columns Used']
    if isinstance(ai_columns, str):
        ai_columns = [col.strip() for col in ai_columns.split('\n') if col.strip()]
    
    # Merge extracted columns with AI-detected columns
    all_columns = list(set(columns + ai_columns))
    
    # Format sections
    formatted_doc = f"""
# Stored Procedure Documentation

## Procedure Name: **{proc_name}**

---

## 1. Overview
{ai_sections['Business Logic'] or 'No overview available'}

---

## 2. Tables Used
{format_list(ai_sections['Tables Used']) or format_list(tables)}

---

## 3. Columns Used
{format_list(all_columns, prefix='- ') or 'No columns detected'}

---

## 4. Transformations
{format_list(ai_sections['Transformations']) or 'No transformations detected'}

## 5. SP Query
{sp_query}

"""
    return formatted_doc

def format_list(items, prefix=''):
    """Helper to format lists consistently"""
    if isinstance(items, str):
        return items
    return '\n'.join([f"{prefix}{item}" for item in items if item]) if items else ''

def app():
    st.title("Stored Procedure Document Generator")
    st.sidebar.header("Database Connection")

    server = st.sidebar.text_input("SQL Server Address", "49.249.56.102")
    username = st.sidebar.text_input("Username", "sql")
    password = st.sidebar.text_input("Password", "Optisol@123", type="password")

    # Session state initialization
    if 'databases' not in st.session_state:
        st.session_state.databases = None
    if 'selected_database' not in st.session_state:
        st.session_state.selected_database = None
    if 'procedures' not in st.session_state:
        st.session_state.procedures = None

    # Fetch Databases
    if st.sidebar.button("Connect to Server"):
        try:
            with get_connection(server, "master", username, password) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sys.databases")
                st.session_state.databases = [db[0] for db in cursor.fetchall()]
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")

    # Database Selection
    if st.session_state.databases:
        st.session_state.selected_database = st.selectbox(
            "Select Database",
            st.session_state.databases
        )

    # Get Procedures
    if st.session_state.selected_database and st.button("Fetch Stored Procedures"):
        try:
            with get_connection(server, st.session_state.selected_database, username, password) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        name, 
                        OBJECT_DEFINITION(object_id) 
                    FROM sys.procedures
                    WHERE is_ms_shipped = 0
                    AND name NOT LIKE 'sp_%%diagram%'
                """)
                st.session_state.procedures = cursor.fetchall()

            if st.session_state.procedures:
                with st.spinner("Generating documentation..."):
                    docs = []
                    for proc_name, sql_code in st.session_state.procedures:
                        tables = extract_table_names(sql_code)
                        columns = extract_columns_from_query(sql_code)
                        ai_doc = generate_ai_documentation(proc_name, sql_code)
                        docs.append(format_documentation(proc_name, tables, columns, ai_doc, sql_code))

                    st.download_button(
                        "Download Documentation",
                        "\n\n".join(docs),
                        file_name="SP_Documentation.txt"
                    )
        except Exception as e:
            st.error(f"Procedure Error: {str(e)}")

if __name__ == "__main__":
    app()
