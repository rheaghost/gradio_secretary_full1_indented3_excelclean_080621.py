import gradio as gr
import os
import threading
import time
import json
import sqlite3
import numpy as np
import scipy.io.wavfile as wav
import whisper
from ollama import Client
import pyttsx3
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import yt_dlp
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import Image
import io

import openpyxl

#-- python functions

# --- Add these functions near the top after imports ---
import pandas as pd

def process_excel(file):
    if file is None:
        return "No file uploaded.", None
    
    try:
        df = pd.read_excel(file.name)
        info = f"📊 File loaded: {file.name}\nRows: {df.shape[0]}, Columns: {df.shape[1]}\nColumns: {', '.join(df.columns.tolist())}"
        preview = df.head(10).to_string()
        return info, preview
    except Exception as e:
        return f"❌ Error: {e}", None

def analyze_financials(file):
    if file is None:
        return "No file uploaded.", ""
    
    try:
        df = pd.read_excel(file.name)
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            total = df[col].sum()
            avg = df[col].mean()
            results.append(f"{col}: Sum={total:,.2f}, Avg={avg:,.2f}")
        
        missing = df.isnull().sum()
        missing_str = "\n".join([f"{col}: {count} missing" for col, count in missing.items() if count > 0])
        
        return "\n".join(results), missing_str if missing_str else "No missing values found."
    except Exception as e:
        return f"❌ Error: {e}", ""

def generate_report(file):
    if file is None:
        return "No file uploaded.", ""
    
    try:
        df = pd.read_excel(file.name)
        report = []
        report.append("=" * 50)
        report.append("FINANCIAL DATA REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"File: {file.name}")
        report.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        report.append("")
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            report.append(f"{col}:")
            report.append(f"  Mean: {df[col].mean():,.2f}")
            report.append(f"  Median: {df[col].median():,.2f}")
            report.append(f"  Min: {df[col].min():,.2f}")
            report.append(f"  Max: {df[col].max():,.2f}")
        
        output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(output_file, index=False)
        return "\n".join(report), output_file
    except Exception as e:
        return f"❌ Error: {e}", ""

#-- end python functions


# --- Configuration ---
client = Client(host='http://localhost:11434')
speaker_on = True
MEMORY_FILE = "secretary_learning_log.txt"
DB_FILE = "secretary.db"
MEMORY_LIMIT = 10

# --- Initialize SQLite ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  role TEXT,
                  content TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- Load Whisper Model ---
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# --- RAG Setup ---
rag_model = None
rag_collection = None
rag_persist_dir = "./chroma_db"

# --- Memory Functions ---
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_to_memory(role, content):
    timestamp = get_timestamp()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (timestamp, role, content) VALUES (?, ?, ?)",
              (timestamp, role, content))
    conn.commit()
    conn.close()
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {role}: {content}\n")

def load_memory(limit=20):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, role, content FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

def clear_memory():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

# --- TTS Function ---
def speak(text):
    if not speaker_on:
        return

    def _speak():
        try:
            engine = pyttsx3.init()
            try:
                voice_id = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
                engine.setProperty('voice', voice_id)
            except:
                pass
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
        finally:
            try:
                engine.stop()
            except:
                pass

    thread = threading.Thread(target=_speak)
    thread.daemon = True
    thread.start()

# --- Process Voice Input ---
def process_voice(audio, sound_on):
    if audio is None:
        return "No audio received.", ""

    try:
        sample_rate, audio_data = audio
        temp_file = "temp_voice.wav"
        wav.write(temp_file, sample_rate, audio_data.astype('int16'))

        result = whisper_model.transcribe(temp_file)
        transcript = result["text"]
        print(f"🗣️ User said: {transcript}")

        save_to_memory("user", transcript)

        response = client.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': transcript}]
        )
        answer = response['message']['content']

        save_to_memory("assistant", answer)

        if sound_on:
            speak(answer)

        os.remove(temp_file)
        return transcript, answer

    except Exception as e:
        error_msg = f"❌ Error: {e}"
        print(error_msg)
        return error_msg, ""

# --- RAG Functions ---
def load_rag_model():
    global rag_model
    if rag_model is None:
        rag_model = SentenceTransformer('all-MiniLM-L6-v2')
    return rag_model

def get_rag_collection():
    global rag_collection
    if rag_collection is None:
        try:
            client_db = chromadb.PersistentClient(path=rag_persist_dir)
            rag_collection = client_db.get_collection("documents")
        except:
            rag_collection = None
    return rag_collection

def index_pdf(file):
    global rag_collection, rag_model
    if file is None:
        return "No file uploaded."

    try:
        reader = PdfReader(file.name)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        chunk_size = 500
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

        model = load_rag_model()
        embeddings = model.encode(chunks)

        client_db = chromadb.PersistentClient(path=rag_persist_dir)
        try:
            collection = client_db.get_collection("documents")
            next_id = collection.count()
            ids = [str(next_id + i) for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=ids
            )
            rag_collection = collection
            return f"✅ Added {len(chunks)} chunks to existing index."
        except:
            collection = client_db.create_collection("documents")
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=[str(i) for i in range(len(chunks))]
            )
            rag_collection = collection
            return f"✅ Created new index with {len(chunks)} chunks."
    except Exception as e:
        return f"❌ Error: {e}"

def ask_document(query):
    global rag_collection, rag_model
    if rag_collection is None:
        rag_collection = get_rag_collection()
        if rag_collection is None:
            return "⚠️ No document indexed. Please upload a PDF first."

    try:
        model = load_rag_model()
        query_embedding = model.encode([query])
        results = rag_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        context = "\n\n".join(results['documents'][0])
        prompt = f"Answer the following question based only on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}"
        response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        answer = response['message']['content']
        if speaker_on:
            speak(answer)
        save_to_memory("assistant", f"RAG: {answer}")
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

def clear_rag_index():
    global rag_collection
    try:
        client_db = chromadb.PersistentClient(path=rag_persist_dir)
        client_db.delete_collection("documents")
        rag_collection = None
        return "🗑️ RAG index cleared."
    except:
        return "⚠️ No index to clear."

# --- Web Summary ---
def summarize_web(url):
    if not url.startswith('http'):
        return "⚠️ Invalid URL."

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"❌ Status: {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)[:3000]

        prompt = f"Summarize this webpage in 3-5 sentences:\n\n{text}"
        response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        answer = response['message']['content']

        if speaker_on:
            speak(answer)

        save_to_memory("assistant", f"Web Summary: {answer}")
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# --- YouTube Audio ---
def summarize_youtube_audio(url):
    try:
        # ydl_opts = {
        #    'format': 'bestaudio/best',
        #    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        #    'outtmpl': 'audio.%(ext)s',
        #    'quiet': True,
        #}
      
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'outtmpl': 'audio.%(ext)s',
            'quiet': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'extractor_args': {
                 'youtube': {
                       'skip': ['dash', 'hls'],
                       'player_client': ['android']
                  }
             }
         }


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        model = whisper.load_model("base")
        result = model.transcribe("audio.mp3")
        transcript = result["text"]

        prompt = f"Summarize this transcript in a few sentences:\n\n{transcript}"
        response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        answer = response['message']['content']

        if speaker_on:
            speak(answer)

        save_to_memory("assistant", f"YouTube Summary: {answer}")
        os.remove("audio.mp3")
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# --- Image Analysis ---
def analyze_image_file(file):
    if file is None:
        return "No image uploaded."

    try:
        with open(file.name, 'rb') as img_file:
            response = client.generate(
                model='llava',
                prompt="Describe this image in detail.",
                images=[img_file.read()]
            )
            answer = response['response']
            if speaker_on:
                speak(answer)
            save_to_memory("assistant", f"Image: {answer}")
            return answer
    except Exception as e:
        return f"❌ Error: {e}"

def analyze_image_url(url):
    if not url.startswith('http'):
        return "⚠️ Invalid URL."

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        if response.status_code != 200:
            return f"❌ Status: {response.status_code}"

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image'):
            return "⚠️ Not an image URL."

        img_bytes = response.content
        response = client.generate(
            model='llava',
            prompt="Describe this image in detail.",
            images=[img_bytes]
        )
        answer = response['response']
        if speaker_on:
            speak(answer)
        save_to_memory("assistant", f"Image URL: {answer}")
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

def analyze_image_webcam(image):
    if image is None:
        return "No image captured."

    try:
        pil_image = Image.fromarray(image.astype('uint8'))
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        response = client.generate(
            model='llava',
            prompt="Describe this image in detail.",
            images=[img_bytes]
        )
        answer = response['response']
        if speaker_on:
            speak(answer)
        save_to_memory("assistant", f"Webcam: {answer}")
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# --- Chat Function with Memory ---
def chat_with_memory(message, history):
    if not message:
        return ""

    save_to_memory("user", message)

    if message.lower().startswith("rag:"):
        query = message[4:].strip()
        answer = ask_document(query)
    else:
        recent = load_memory(MEMORY_LIMIT)
        context = [{"role": row[1], "content": row[2]} for row in recent]
        context.append({"role": "user", "content": message})

        response = client.chat(model='llama3', messages=context)
        answer = response['message']['content']

    save_to_memory("assistant", answer)
    return answer

# --- Export Log ---
def export_log_old():
    log_file = MEMORY_FILE
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return content, f"secretary_log_{timestamp}.txt"
    else:
        rows = load_memory(50)
        if rows:
            content = "\n".join([f"[{row[0]}] {row[1]}: {row[2]}" for row in rows])
            return content, f"secretary_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return "No log data found.", None

def export_log():
    log_file = MEMORY_FILE
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"secretary_log_{timestamp}.txt"
        # Write a temporary file that Gradio can pick up
        temp_path = os.path.join(os.getcwd(), filename)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content, temp_path
    else:
        rows = load_memory(50)
        if rows:
            content = "\n".join([f"[{row[0]}] {row[1]}: {row[2]}" for row in rows])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"secretary_log_{timestamp}.txt"
            temp_path = os.path.join(os.getcwd(), filename)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            return content, temp_path
        return "No log data found.", None


# -- show excel code
def show_excel_code_old():
    """Returns the Python code used for the current Excel analysis."""
    code = '''
import pandas as pd

# Load the Excel file portfolio->test_data 2608062034
df = pd.read_excel('test_data.xlsx')

# Basic Info
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns: {', '.join(df.columns.tolist())}")

# Preview
print(df.head())

# Numeric statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

# Group by (example)
# sector_summary = df.groupby('Sector').agg({'Revenue': 'sum', 'Profit': 'mean'})
# print(sector_summary)
'''
    return code

#new show_excel_code 2608062021

def show_excel_code():
    code = '''
import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel('your_file.xlsx')

# Basic Info
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns: {', '.join(df.columns.tolist())}")

# Preview
print(df.head())

# Numeric statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

# --- ADD YOUR CUSTOM CODE BELOW ---
# Group by example
# sector_summary = df.groupby('Sector').agg({'Revenue': 'sum', 'Profit': 'mean'})
# print(sector_summary)

# Filter example
# tech_companies = df[df['Sector'] == 'Tech']
# print(tech_companies)

# Calculate new column example
# df['Growth'] = (df['Revenue_2024'] - df['Revenue_2023']) / df['Revenue_2023'] * 100
'''
    return code

# copy code to clipboard
def copy_code():
    code = show_excel_code()
    # This will be handled by Gradio's gr.Code component
    return code


# --- Gradio UI ---
def respond(message, history, sound_on):
    if not message:
        return "", history

    response = chat_with_memory(message, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    if sound_on:
        speak(response)

    return "", history

# --- Build UI ---
with gr.Blocks(title="Secretary 2026", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Secretary 2026 — Full Edition")

    with gr.Row():
        sound_toggle = gr.Checkbox(label="🔊 Sound", value=True)
        clear_memory_btn = gr.Button("🗑️ Clear Memory")

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg = gr.Textbox(label="Ask your secretary...", placeholder="Type your message...", scale=4, elem_id="msg_textbox")
                send_btn = gr.Button("📤 Send", scale=1)
            clear_chat_btn = gr.Button("🗑️ Clear Chat")

            send_btn.click(respond, [msg, chatbot, sound_toggle], [msg, chatbot])
            msg.submit(respond, [msg, chatbot, sound_toggle], [msg, chatbot])
            clear_chat_btn.click(lambda: ([], []), None, [chatbot, chatbot])

        with gr.TabItem("🎤 Voice"):
            gr.Markdown("Speak into your microphone.")
            with gr.Row():
                voice_input = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Click to Record")
            with gr.Row():
                transcript_output = gr.Textbox(label="📝 Transcript", interactive=False, lines=3)
                response_output = gr.Textbox(label="💬 Response", interactive=False, lines=6)
            voice_input.change(process_voice, inputs=[voice_input, sound_toggle], outputs=[transcript_output, response_output])

        with gr.TabItem("📄 RAG"):
            gr.Markdown("Upload a PDF and ask questions.")
            with gr.Row():
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                index_btn = gr.Button("📥 Index")
                index_output = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                rag_query = gr.Textbox(label="Question")
                rag_btn = gr.Button("🔍 Ask")
                rag_output = gr.Textbox(label="Answer", interactive=False)

            index_btn.click(index_pdf, [pdf_file], [index_output])
            rag_btn.click(ask_document, [rag_query], [rag_output])

        with gr.TabItem("🌐 Web"):
            with gr.Row():
                web_url = gr.Textbox(label="URL")
                web_btn = gr.Button("📄 Summarize")
            web_output = gr.Textbox(label="Summary", interactive=False)
            web_btn.click(summarize_web, [web_url], [web_output])

        with gr.TabItem("🎥 YouTube"):
            with gr.Row():
                yt_url = gr.Textbox(label="YouTube URL")
                yt_btn = gr.Button("📝 Summarize")
            yt_output = gr.Textbox(label="Summary", interactive=False)
            yt_btn.click(summarize_youtube_audio, [yt_url], [yt_output])

        with gr.TabItem("🖼️ Image"):
            with gr.Tabs():
                with gr.TabItem("File"):
                    img_file = gr.File(label="Upload Image")
                    file_btn = gr.Button("🔍 Analyze")
                    file_output = gr.Textbox(label="Description", interactive=False)
                    file_btn.click(analyze_image_file, [img_file], [file_output])

                with gr.TabItem("URL"):
                    img_url = gr.Textbox(label="Image URL")
                    url_btn = gr.Button("🔍 Analyze")
                    url_output = gr.Textbox(label="Description", interactive=False)
                    url_btn.click(analyze_image_url, [img_url], [url_output])

                with gr.TabItem("Webcam"):
                    webcam_input = gr.Image(sources=["webcam"], label="Take a Photo")
                    webcam_btn = gr.Button("📸 Capture")
                    webcam_output = gr.Textbox(label="Description", interactive=False)
                    webcam_btn.click(analyze_image_webcam, [webcam_input], [webcam_output])

        with gr.TabItem("🧠 Memory"):
            gr.Markdown("Recent conversation history.")
            memory_output = gr.Textbox(label="Recent Messages", lines=20, interactive=False)
            refresh_btn = gr.Button("🔄 Refresh")

            def show_memory():
                rows = load_memory(20)
                if not rows:
                    return "No memory yet."
                return "\n".join([f"[{row[0]}] {row[1]}: {row[2]}" for row in rows])

            refresh_btn.click(show_memory, outputs=memory_output)

# excel tab

# === EXCEL SUITE TAB replaced===
    # === EXCEL SUITE TAB ===
        with gr.TabItem("📊 Excel Suite"):
            gr.Markdown("# Excel Financial Analysis Suite")
    
            with gr.Tabs():
               # --- Preview Tab ---
                with gr.TabItem("📤 Upload & Preview"):
                    gr.Markdown("Upload an Excel file to preview its contents")
                    excel_file = gr.File(label="Upload Excel (.xlsx)", file_types=[".xlsx"])
                    excel_btn = gr.Button("📊 Preview")
                    excel_info = gr.Textbox(label="File Info", lines=5, interactive=False)
                    excel_preview = gr.Textbox(label="Preview", lines=10, interactive=False)
                    excel_btn.click(process_excel, [excel_file], [excel_info, excel_preview])
        
              # --- Financial Analysis Tab ---
                with gr.TabItem("📊 Financial Analysis"):
                    gr.Markdown("Upload financial data for statistical analysis")
                    fin_file = gr.File(label="Upload Excel", file_types=[".xlsx"])
                    fin_btn = gr.Button("📊 Analyze Financials")
                    fin_stats = gr.Textbox(label="Statistics", lines=10, interactive=False)
                    fin_missing = gr.Textbox(label="Missing Data", lines=5, interactive=False)
                    fin_btn.click(analyze_financials, [fin_file], [fin_stats, fin_missing])
        
              # --- Show Code Tab (with copy built-in via gr.Code) ---
                with gr.TabItem("📋 Show Code"):
                    gr.Markdown("Click the button below to see the Python code used for analysis.")
                    code_btn = gr.Button("📄 Show Code")
                    code_output = gr.Code(label="Python Code", language="python", lines=20, interactive=False)
                    code_btn.click(show_excel_code, [], [code_output])
        
              # --- Report Generator Tab ---
                with gr.TabItem("📄 Generate Report"):
                    gr.Markdown("Generate a comprehensive financial report")
                    report_file = gr.File(label="Upload Excel", file_types=[".xlsx"])
                    report_btn = gr.Button("📄 Generate Report")
                    report_output = gr.Textbox(label="Report", lines=20, interactive=False)
                    report_download = gr.File(label="Download Report")
                    report_btn.click(generate_report, [report_file], [report_output, report_download])

# end excel tab


# excel example analysis
        with gr.TabItem("📊 Excel Analyzer"):
            gr.Markdown("Upload any Excel file to analyze it")
    
            # Upload file
            file = gr.File(label="Upload Excel file (.xlsx)", file_types=[".xlsx"])
    
            # Analysis options
            with gr.Row():
                 analyze_btn = gr.Button("📊 Run Full Analysis")
                 clear_btn = gr.Button("🗑️ Clear Output")
    
            # Output
            output = gr.Textbox(label="Analysis Results", lines=30, interactive=False)
    
            def analyze_any_excel(file):
                 if file is None:
                     return "Please upload a file."
        
                 try:
                     df = pd.read_excel(file.name)
                     output_lines = []
            
                    # Basic info
                     output_lines.append("=" * 60)
                     output_lines.append(f"📊 FILE: {file.name}")
                     output_lines.append("=" * 60)
                     output_lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                     output_lines.append(f"Columns: {', '.join(df.columns.tolist())}")
                     output_lines.append("")
            
                     # First few rows
                     output_lines.append("📋 PREVIEW:")
                     output_lines.append(df.head(5).to_string())
                     output_lines.append("")
            
                     # Statistics for numeric columns
                     numeric_cols = df.select_dtypes(include=[np.number]).columns
                     if len(numeric_cols) > 0:
                         output_lines.append("📊 NUMERIC STATISTICS:")
                         output_lines.append(df[numeric_cols].describe().to_string())
                         output_lines.append("")
            
                     # Missing values
                     missing = df.isnull().sum()
                     if missing.sum() > 0:
                         output_lines.append("⚠️ MISSING VALUES:")
                         for col, count in missing.items():
                             if count > 0:
                                 output_lines.append(f"  {col}: {count} missing")
                         output_lines.append("")
            
                     # Export
                     output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                     df.to_excel(output_file, index=False)
                     output_lines.append(f"✅ Analysis exported to: {output_file}")
            
                     return "\n".join(output_lines)
                 except Exception as e:
                     return f"❌ Error: {e}"
    
            analyze_btn.click(analyze_any_excel, [file], [output])        

# end excel example analysis





        with gr.TabItem("📋 Logs"):
            log_output = gr.Textbox(label="Log Content", lines=20, interactive=False)
            download_btn = gr.Button("📥 Download Log")

            def get_log():
                content, filename = export_log()
                return content, filename

            download_btn.click(get_log, outputs=[log_output, gr.File()])

    clear_memory_btn.click(clear_memory, outputs=[])

# --- Run ---
if __name__ == "__main__":
    demo.launch()
