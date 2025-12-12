from PIL import Image, ImageOps
import cv2, numpy as np
import easyocr
import streamlit as st
import numpy as np
import json
import yaml
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
import os
import pickle
from groq import Groq


client = Groq(api_key="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")
print(client)

# Paths
# --------------------------------------------------
INDEX_PATH = "index.faiss"
DOCS_PATH = "chunks.pkl"

# --------------------------------------------------
# Lazy load containers
# --------------------------------------------------
embedder = None
index = None
documents = None


# --------------------------------------------------
# Load RAG resources only when needed
# --------------------------------------------------
def load_rag_resources():
    global embedder, index, documents

    if embedder is None:
        embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    if index is None:
        index = faiss.read_index(INDEX_PATH)

    if documents is None:
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)

    print("✔ RAG resources loaded")


# --------------------------------------------------
# Retrieval
# --------------------------------------------------
def retrieve(query, k=3):
    global embedder, index, documents

    q = embedder.encode(query, normalize_embeddings=True)
    q = np.array([q]).astype("float32")

    D, I = index.search(q, k)

    return [documents[i] for i in I[0]]


# --------------------------------------------------
# RAG Answer
# --------------------------------------------------
def rag_answer(question, k=3):
    global embedder, index, documents

    retrieved_chunks = retrieve(question, k)
    context = " ".join(retrieved_chunks)

    if not context.strip():
        return "Answer is not found", []

    # -----------------------------
    # UPDATED OCD RULES (your request)
    # -----------------------------
    prompt = f"""
You are a strict RAG assistant.

RULES:
- Fully understand all 3 retrieved chunks carefully.
- Generate the best, properly detailed answer (up to 500 words).
- If any coding lines or syntax appear in context, show them clearly in a separate formatted block.
- If the answer is not found inside CONTEXT, reply exactly:
  "Answer is not found"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
        {"role": "system", "content": "You are a strict RAG model."},
        {"role": "user", "content": prompt}
          ]
    )

    answer = response.choices[0].message.content.strip()


    # Standardize "no answer" string
    if "Answer is not found" in answer or "No answer found" in answer:
        answer = "Answer is not found"

    return answer, retrieved_chunks

# -----------------------------
# 2. LLM SYSTEM PROMPT
# -----------------------------
SURVEY_SUMMARY_SYSTEM_PROMPT = """
You are a survey-question interpreter.

Your task:
Given any user-written survey question (text extracted from OCR or typed by the user), produce a clean, structured summary.

You must extract and return ONLY the following fields in a YAML-like structure:
understand clearly about the question and extract question label, question text, row text, column text.
QuestionNumber: <number or null>
- Please extract the correct QuestionNumber it wil Q1,Q2 or S1,S2 like that and for info question its follow info1,info2 like that so please make it correct
QuestionText: "<text>"

Comment: "<text or null>" 
-after the question text comment start with "(" and end with ")" and keep. at end of the sentence
- sometimes it was mentioned in question text also so please extract carefully that also

QuestionType: <single-select | multi-select | single-select-grid | multi-select-grid | text | numeric | autosum | drop-down | ranking | information | other>
- in numeric question if there is row so it should be question type of autosum
- info this a information question it contain info1 or etc ..

Rows:
  - text: "<option text>"
    row_label: "<r1 | r2 | r3 ...>"   # The nearest label in text; prefix with 'r'
    anchor: true/false
    exclusive: true/false
    other-specify: true/false
    - read the question properly understand the instruction is belongs to row display or question display logic 
    - if row display logic is found understand and build logic - S5.r1 or S5.r1.c1 or S5.ch1 like that
    - use and and or not operator to build like question display logic
    row-display: "S1.r1"

Columns:
  - text: "<option text>"
    column_label: "<c1 | c2 | c3 ...>"   # The nearest label in text; prefix with 'c' 
    anchor: true/false
    exclusive: true/false
    other-specify: true/false 
  # If the question is not a grid, return: Columns: none

Choice:
   If ranking questions:
  Identify how many ranks are required from the question instruction like rank up to 3 and generate that much choice.
  # Example:
  #   - text: "Rank1"
  #     choice_label: "ch1"
  #   - text: "Rank2"
  #     choice_label: "ch2"
  #     anchor: true/false
  #     exclusive: true/false
  If dropdown question
  text: "<option text>"
  column_label: "<ch1 | ch2 | ch3 ...>
  anchor: true/false

AdditionalInstructions:
  shuffle: true/false
  randomization: true/false
  display_logic: true/false        # If question contains display/skip logic
  termination: true/false          # If question contains termination/end instruction
  note: "Original text contained bold or underlined emphasis."
termination_condition:
 If the question mentions screening / must-select / must-be logic such as:
 - "MUST BE S5/1, IF NOT TERMINATE"
 - “Only continue if respondent selects 1”
 - "Terminate if they do not choose option 2"
 - “Only rows 1 and 3 qualify”

 Then:
 1. Set "termination": true
 2. Identify the exact row(s) required.
 3. Output if row based termination_condition = "Q{question-number}.r{row-number}"
 4. Output if row and col based termination_condition = "Q{question-number}.r{row-number}.c{col-num}"
 5. Output if row and choice termination_condition =Q{question-number}.r{row-number}.ch{choice-num}"
 6. termination logical and, or, not condition based on the python only

 Examples:
 - “If respondant select the S5/1 or if respondant selects the option (something they not provide row name example terminate if respondant selects ad) ”
   → termination_condition = "S5.r1"

 - "Terminate if respondant selects S5/1-3, S5/1,2,3"
  -> termination _condition = "S5.r1 or S5.r2 or S5.r3"

 - "Terminate if NOT selecting S5/r2"
   → termination_condition = "not(S5.r2)"

 - "Terminate if respondant selects row 2 and row 3"
   → termination_condition = "S5.r2 and S5.r3"

  - "Only continue if choosing 1 or 3"
   → termination_condition = "not(S5.r1 or S5.r3)"

 - for numberic type of termination condition is qnum.val -> to extract the value from the question
 - use this keywords build the logic "eq"- equal "gt"-greater than "lt" - less than, "ge" - greater than or equal to , "le" - less than or equal to

 - if you can't able to create the termination condition
   termination_condition = "update here"

DisplayLogic:
- please understand the display logic in question itself they mentioned display if or ask if like that words
-  understand that the question need to display if reponsant selects row and column from what question
- understand the logic and build using and, or, or operator like python condition
-() use this for clear structure of the condition
Example :"[DISPLAY IF Q15/14 IS SELECTED]"
-> DisplayLogic = "Q15.r14"
Example :"[ASK IF S25/2 IS SELECTED]"
-> DisplayLogic = "S25.r2"
Example :"[ASK IF S25_2-4/4 IS SELECTED]"
-> DisplayLogic = "S25.r2.c4 or S25.r3.c4 or S25.r4.c4"
Example :"[DISPLAY IF Q15/2-3 IS SELECTED]"
-> DisplayLogic = "Q15.r2 or Q15.r3 or Q15.r4"
Example :"[DISPLAY MAIN CLIENT’S TO AGENCIES (S15/1,3), DISPLAY COMPANY’S TO MARKETERS (S15/2)]"
-> DisplayLogic = "(S15.r1 or S15.r3) and (S15.r2)
Example : [DISPLAY IF Q15_3/DNE99] DNE99 Its stands for do not equal to r99 understand it.
-> DisplayLogic = "(Q15.r3 or not(Q15.r99))"


RULES:
- avoid print the instruction in the code which contain inside the [] square bracket example [MULTIPLE SELECT; RANDOMIZE]
- avoid spelling mistake
- please more focus on the / and . in text 
- Do NOT generate any survey programming code.
- Do NOT generate XML, HTML, or tags such as <radio>.
- Output ONLY the above YAML-like structure.
- If any field is missing, still include it with null/none.
- The result must be plain text, not JSON.

"""


# -----------------------------
# 3. GEMINI CLIENT (Correct Way)
# -----------------------------




def summarize_question(final_input):
    response = client.chat.completions.create(
    model="openai/gpt-oss-120b",   # recommended stable Groq model
    messages=[
        {"role": "system", "content": SURVEY_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": final_input}
    ])

    summary_text = response.choices[0].message.content
    print(summary_text)    
    return summary_text
    
 # -----------------------------
# 4. PARSE YAML OUTPUT
# -----------------------------
def parse_summary(summary_text):
    documents = list(yaml.safe_load_all(summary_text))
    return documents[0] if documents else None   

def generate_survey_code(summary):
    qtype = summary.get("QuestionType")
    qnum = summary.get("QuestionNumber")
    qtext = summary.get("QuestionText")
    comment = summary.get("Comment")
    rows = summary.get("Rows", [])

        # ---------- SAFE ROWS ----------
    rows = summary.get("Rows")
    if not isinstance(rows, list):
        rows = []
    # ---------- SAFE COLUMNS ----------
    columns = summary.get("Columns")
    if not isinstance(columns, list):
        columns = []

    # ---------- SAFE CHOICES ----------
    choices = summary.get("Choice")
    if not isinstance(choices, list):
        choices = []
    add = summary.get("AdditionalInstructions", {})
    shuffle = add.get("shuffle", False)
    termination = add.get("termination", False)
    randomization = add.get("randomization", False)
    display_logic = add.get("display_logic", False)
    termination = add.get("termination", False)
    term_cond = summary.get("termination_condition")
    display_cond = summary.get("DisplayLogic")
   

    print("Display cond",display_cond)

     
    code =""

    if qtype == "single-select":
        radio_tag = f'<radio label="{qnum}"'

        if display_logic or display_cond:          
            radio_tag += f'cond="{display_cond}"'
        if shuffle or randomization:
            radio_tag += ' shuffle="rows"'
        radio_tag += ">"
        code += radio_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'

            code += row_tag + "\n"
        code += "</radio>\n"
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'        
        return code  

    if qtype == "multi-select":
        checkbox_tag = f'<checkbox label="{qnum}" atleast="1"'
        if shuffle or randomization:
            checkbox_tag += ' shuffle="rows"'
        if display_logic and display_cond:
            checkbox_tag += f' cond="{display_cond}"' 
        checkbox_tag += ">"
        code += checkbox_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'

            code += row_tag + "\n"
        code += "</checkbox>\n"
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code  

    if qtype == "text":
        text_tag = f'<textarea label="{qnum}" optional="0"'
        if display_logic or display_cond:
            text_tag += f' cond="{display_cond}"' 
        text_tag += ">"
        code += text_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        code+=f'<validate>CheckBlank({qnum},1)</validate>' 
        code += "</textarea>\n"         
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code         
    
    
    if qtype == "single-select-grid":
        single_radio_tag = f'<radio label="{qnum}"'

        if display_logic and display_cond:
             single_radio_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            single_radio_tag += ' shuffle="rows"'

        single_radio_tag += ">"
        code += single_radio_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
             code += f'  <comment><i>{comment}</i></comment>\n'

    # --- Rows ---
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"

    # --- Columns ---
    
        for col in columns:
            col_label = col.get("column_label", "cX")
            text = col.get("text", "")
            anchor = col.get("anchor", False)
            exclusive = col.get("exclusive", False)
            other_spec = col.get("other-specify", False)
            col_tag = f'  <col label="{col_label}"'
            if anchor:
                col_tag += ' randomization="0"'
            if other_spec:
                col_tag += ' open="1" openSize="25"'
            if exclusive:
                col_tag += ' exclusive="1"'
            col_tag += f'>{text}</col>'
        
        code += col_tag + "\n"

        code += "</radio>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code


    if qtype == "multi-select-grid":
        multi_check_tag = f'<checkbox label="{qnum}"'

        if display_logic and display_cond:
            multi_check_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            multi_check_tag += ' shuffle="rows"'

        multi_check_tag += ">"
        code += multi_check_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'

    # --- Rows ---
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"

    # --- Columns ---
    
        for col in columns:
            col_label = col.get("column_label", "cX")
            text = col.get("text", "")
            anchor = col.get("anchor", False)
            exclusive = col.get("exclusive", False)
            other_spec = col.get("other-specify", False)
            col_tag = f'  <col label="{col_label}"'
            if anchor:
                col_tag += ' randomization="0"'
            if other_spec:
                col_tag += ' open="1" openSize="25"'
            if exclusive:
                col_tag += ' exclusive="1"'
            col_tag += f'>{text}</col>'
            code += col_tag + "\n"

        code += "</checkbox>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code
    

    if qtype == 'information':
        info_tag = f'<html label="{qnum}"'
        if display_logic and display_cond:
            info_tag += f' cond="{display_cond}"'
        info_tag += ">"
        code += info_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        code += "</html>\n"
        code += "<suspend/>\n"    
        return code        


    

    if qtype == 'ranking':
        select_tag = f'<select uses="ranksort.4" label="{qnum}"'
        if display_logic and display_cond:
            select_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            select_tag += ' shuffle="rows"'
        select_tag += ">"
        code += select_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'

    # --- Rows ---
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"

    # --- Choices ---
        
        for choice in choices:
            cho_label = choice.get("choice_label", "chX")
            text = choice.get("text", "")
            anchor = choice.get("anchor", False)
            exclusive = choice.get("exclusive", False)
            other_spec = choice.get("other-specify", False)
            cho_tag = f'  <choice label="{cho_label}"'
            if anchor:
                cho_tag += ' randomization="0"'
            if exclusive:
                cho_tag += ' exclusive="1"'
            cho_tag += f'>{text}</choice>'
            code += cho_tag + "\n"

        code += "</select>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code

    if qtype == "numeric":
        number_tag = f'<number size="2" verify="range(0,100)" label="{qnum}"'
        if display_logic and display_cond:
            number_tag += f' cond="{display_cond}"'
        number_tag += ">"
        code += number_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        code += "</number>\n"
        code += "<suspend/>\n"   
        if termination or term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code  

    if qtype == 'drop-down':
        select_tag = f'<select label="{qnum}"'
        if display_logic and display_cond:
            select_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            select_tag += ' shuffle="rows"'
        select_tag += ">"
        code += select_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'

          
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"
    # --- Choices ---
        
        for choice in choices:
            cho_label = choice.get("choice_label", "chX")
            text = choice.get("text", "")
            anchor = choice.get("anchor", False)
            exclusive = choice.get("exclusive", False)
            other_spec = choice.get("other-specify", False)
            cho_tag = f'  <choice label="{cho_label}"'
            if anchor:
                cho_tag += ' randomization="0"'
            if exclusive:
                cho_tag += ' exclusive="1"'
            cho_tag += f'>{text}</choice>'
            code += cho_tag + "\n"

        code += "</select>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code      

    if qtype == "autosum":
        num_tag = f'<number amount="100" size="2" autosum:prefill="0" label="{qnum}"'

        if display_logic and display_cond:          
            num_tag += f'cond="{display_cond}"'
        if shuffle or randomization:
            num_tag += ' shuffle="rows"'
        num_tag += ">"
        code += num_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            row_logic = row.get("row-display")
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if row_logic:
                row_tag += f' cond="{row_logic}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'

            code += row_tag + "\n"
        code += "</number>\n"
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'        
        return code  

    if qtype=="other":
        code = "We will update it"
    return code 





    



# ---------------------- GLOBAL SUPER UI CSS ----------------------
super_css = """
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Main background */
body {
    background: linear-gradient(135deg, #e0f2ff 0%, #ffffff 100%);
}

/* Title */
h1 {
    text-align: center !important;
    color: #004e92 !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
}

/* Text input area */
textarea {
    border-radius: 15px !important;
    border: 2px solid #80c7ff !important;
    padding: 12px !important;
    background: #f7fbff !important;
    box-shadow: 0 0 8px rgba(0, 123, 255, 0.15);
}

/* Select box */
.stSelectbox > div {
    border-radius: 12px !important;
    border: 2px solid #80bfff !important;
    background-color: #f0f8ff !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #007bff, #0059b3) !important;
    color: white !important;
    border: none !important;
    padding: 12px 25px !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 10px rgba(0, 123, 255, 0.35) !important;
    transition: 0.2s ease-in-out !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #005ec2, #004089) !important;
    transform: translateY(-2px);
}

/* Info boxes */
.stAlert {
    border-radius: 12px !important;
}

/* Code block */
div.stCodeBlock {
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15) !important;
}

/* Expander */
details {
    border-radius: 10px !important;
    background: #f3faff !important;
    border: 1px solid #cbe7ff !important;
}

summary {
    font-weight: bold !important;
    color: #004e92 !important;
    font-size: 16px !important;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255, 255, 255, 0.65);
    display: flex; 
    justify-content: center;
    align-items: center;
    z-index: 9999;
    font-size: 32px;
    font-weight: 700;
    color: #004e92;
    animation: pulse 1.1s infinite;
}

/* Pulse animation */
@keyframes pulse {
    0%   { transform: scale(1); opacity: 1; }
    50%  { transform: scale(1.05); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
}

</style>
"""
st.markdown(super_css, unsafe_allow_html=True)
# ---------------------------------------------------------------




# ---------------------- LOADING OVERLAY CSS ----------------------
loading_css = """
<style>
.loading-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255, 255, 255, 0.8);
    display: flex; 
    justify-content: center;
    align-items: center;
    z-index: 9999;
    font-size: 28px;
    font-weight: bold;
}
</style>
"""
st.markdown(loading_css, unsafe_allow_html=True)

def show_loading(msg="Loading..."):
    box = st.empty()
    box.markdown(f"<div class='loading-overlay'>{msg}</div>", unsafe_allow_html=True)
    return box
# -----------------------------------------------------------------

st.title("Survey XML Code Generator + RAG QA Bot")

st.write("Paste question text")

# --- MODE SELECTOR ---
mode = st.selectbox("Choose Mode:", ["XML", "QA"])

# --- INPUT TEXT ---
input_text = st.text_area("Paste text here:")

# --------- MAIN BUTTON ----------
if st.button("Generate / Ask"):
    if not input_text:
        st.error("Please enter text first!")
    else:

        # 🌟 SHOW CENTER LOADING MESSAGE 🌟
        loading = show_loading("⚙️ Processing... Please wait...")

        if mode == "XML":
            summary = summarize_question(input_text)
            parsed = parse_summary(summary)
            code = generate_survey_code(parsed)

            loading.empty()   # hide loading UI

            st.info("XML Generation Mode Selected")
            st.subheader("Generated XML Code")
            st.code(code)

        elif mode == "QA":
            load_rag_resources()
            answer, context_docs = rag_answer(input_text)
            
            
            loading.empty()
            st.info("QA Mode Activated")
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("Context Used"):
                for i, doc in enumerate(context_docs, 1):
                    st.write(f"--- Chunk {i} ---")
                    st.write(doc)
