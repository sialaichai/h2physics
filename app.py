import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import io
import requests
from PIL import Image
from gtts import gTTS
from duckduckgo_search import DDGS

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="H2 Feynman Bot",
    page_icon="⚛️",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_audio(text):
    """Generates MP3 audio from text, skipping code/image tags."""
    # Clean text so the bot doesn't read code or tags out loud
    clean_text = re.sub(r'```.*?```', 'I have generated a graph.', text, flags=re.DOTALL)
    clean_text = re.sub(r'\[IMAGE:.*?\]', 'Here is a diagram.', clean_text)
    try:
        tts = gTTS(text=clean_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except:
        return None

def google_search_api(query, api_key, cx):
    """Helper: Performs a single Google Search."""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": cx,
            "key": api_key,
            "searchType": "image",
            "num": 3,
            "safe": "active"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code in [403, 429]:
            return None # Failover trigger
            
        data = response.json()
        
        if "items" in data and len(data["items"]) > 0:
            for item in data["items"]:
                link = item["link"]
                if link.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return link
            return data["items"][0]["link"]
            
    except Exception as e:
        print(f"Google API Exception: {e}")
        return None
    return None

def duckduckgo_search_api(query):
    """Helper: Fallback search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=1))
            if results:
                return results[0]['image']
    except Exception as e:
        return f"Search Error: {e}"
    return "No image found."

@st.cache_data(show_spinner=False)
def search_image(query):
    """MASTER FUNCTION: Google Key 1 -> Google Key 2 -> DuckDuckGo"""
    cx = os.environ.get("GOOGLE_CX")
    
    # 1. Try Google Key 1
    key1 = os.environ.get("GOOGLE_SEARCH_KEY")
    if key1 and cx:
        url = google_search_api(query, key1, cx)
        if url: return url

    # 2. Try Google Key 2
    key2 = os.environ.get("GOOGLE_SEARCH_KEY_2")
    if key2 and cx:
        url = google_search_api(query, key2, cx)
        if url: return url

    # 3. Fallback to DuckDuckGo
    return duckduckgo_search_api(query)

def execute_plotting_code(code_snippet):
    try:
        plt.figure()
        local_env = {'plt': plt, 'np': np}
        exec(code_snippet, {}, local_env)
        st.pyplot(plt)
        plt.clf()
    except Exception as e:
        st.error(f"Graph Error: {e}")

def display_message(role, content, enable_voice=False):
    with st.chat_message(role):
        
        text_to_display = content
        
        # 1. Check for Python Code
        code_match = re.search(r'```python(.*?)```', content, re.DOTALL)
        if code_match and role == "assistant":
            text_to_display = text_to_display.replace(code_match.group(0), "")
        
        # 2. Check for [IMAGE: query] Tags
        image_match = re.search(r'\[IMAGE:\s*(.*?)\]', text_to_display, re.IGNORECASE)
        image_result = None
        
        if image_match and role == "assistant":
            search_query = image_match.group(1)
            text_to_display = text_to_display.replace(image_match.group(0), "")
            image_result = search_image(search_query)

        # --- DISPLAY ---
        st.markdown(text_to_display)

        if code_match and role == "assistant":
            with st.expander("Show Graph Code"):
                st.code(code_match.group(1), language='python')
            execute_plotting_code(code_match.group(1))
            
        if image_match and role == "assistant":
            if image_result and "Error" not in image_result:
                st.image(image_result, caption=f"Diagram: {image_match.group(1)}")
                st.markdown(f"[🔗 Open Image in New Tab]({image_result})")
            else:
                st.warning(f"⚠️ Image Search Failed: {image_result}")

        if enable_voice and role == "assistant" and len(text_to_display.strip()) > 0:
            audio_bytes = generate_audio(text_to_display)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')

# -----------------------------------------------------------------------------
# 3. SYSTEM INSTRUCTIONS (UPDATED WITH YOUR CHANGES)
# -----------------------------------------------------------------------------
SEAB_H2_MASTER_INSTRUCTIONS = """
**Identity:**
You are Richard Feynman. Tutor for Singapore H2 Physics (Syllabus 9478).
**CORE TOOLS (MANDATORY):**
1.  ** Constraints:**
    * Strictly limit syllabus to SEAB H2 Physics 9478
    * Do not include any other A-level physics syllabus
    
2.  **Graphs (Python):** If asked to plot/graph, WRITE PYTHON CODE.
    * Use `matplotlib.pyplot`, `numpy`, `scipy`.
    * Enclose in ` ```python ` blocks.
3.  **Diagrams (Web Search):** If you need to show a diagram, YOU MUST USE THE TAG.
    * **Syntax:** `[IMAGE: <concise search query>]`
    * Example: "Here is the setup: [IMAGE: rutherford gold foil experiment diagram]"
    * **Rule:** Do NOT use markdown image links. Use `[IMAGE:...]` ONLY.
4.  **Multimodal:** You can see images and hear audio uploaded by the user.
**PEDAGOGY:**
* Ask **ONE** simple question at a time.
**Feynman-Style Questioning:** Do not just ask for formulas. Use **analogies** to guide their thinking.
* *Bad:* "What is the formula for voltage?"
* *Good (Feynman):* "Think of the wire like a pipe carrying water. What would represent the 'pressure' pushing the water through?"
* **Do not** solve the math immediately. Guide the student with hints or choices in the questions.
**No Hand-Holding:** Do not give the answer too quickly. If they struggle, give a simpler analogy but do not over-simplify. 
* Ask question but not to irritate students by asking too many questions. Try to keep to 3 or less questions.
* **The "I Give Up" Clause:** Only provide the full solution if the student explicitly says "I give up" or "Just tell me the answer."
* Validate them enthusiastically when the student successfully grasps the concept or solves the problem.
* When the student asks to be tested, ask them to explain a specific A-level concept (e.g., "Diffraction") to YOU in simple terms.
* Critique students explanation. Point out exactly where they used jargon to hide a lack of understanding.
* **Summarize:** * When they show some degree of understand, IMMEDIATELY provide a clear, concise **"Summary Note"** of the entire solution or concept.
* Recap the steps you took together.
* State the final formula/answer clearly.
* Use a Markdown blockquote (`>`) for this summary so it stands out.
**Tone & Style:**
* Enthusiastic, curious, encouraging but rigorous, non-condescending and unpretentious.
* Use simple words. Avoid jargon unless you explain it first. Keep answers concise
* "Stop and check": Don't lecture for too long. Explain one concept, then ask the student if they get it.
* **Math:**
* Use LaTeX for ALL math formulas. Enclose them in single dollar signs for inline (e.g., $F=ma$) and double dollar signs for standalone (e.g., $$E = mc^2$$).
* Use **bold** for key terms.
**Common Student Misconceptions to Watch For:**
* Confusion between weight (N) and mass (kg).
* Thinking current gets "used up" in a series circuit.
* Thinking "Centripetal force" is a *new* physical force (rather than just the net force).
* Thinking heavy objects fall faster than light ones (in a vacuum).    
* *Correct these gently but firmly whenever they appear.*
"""

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/42/Richard_Feynman_Nobel.jpg)", width=150)
    
    st.header("⚙️ Settings")
    topic = st.selectbox("Topic:", ["General / Any", "Measurement & Uncertainty", "Kinematics & Dynamics", 
         "Forces & Turnings Effects", "Work, Energy, Power", "Circular Motion", 
         "Gravitational Fields", "Thermal Physics", "Oscillations & Waves", 
         "Electricity & DC Circuits", "Electromagnetism (EMI/AC)", "Modern Physics (Quantum/Nuclear)", 
         "Paper 4: Practical Skills (Spreadsheets)"])
    enable_voice = st.toggle("🗣️ Read Aloud", value=False)

    api_key = None
    if "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]
    if not api_key:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass
    if not api_key:
        api_key = st.text_input("Enter Google API Key", type="password")

    st.divider()
    
    st.markdown("### 📸 Vision & 🎙️ Voice")
    
    tab_upload, tab_cam, tab_mic = st.tabs(["📂 File", "📷 Cam", "🎙️ Voice"])
    
    visual_content = None
    audio_content = None
    
    with tab_upload:
        uploaded_file = st.file_uploader("Upload Image/PDF", type=["jpg", "png", "jpeg", "pdf"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                visual_content = {"mime_type": "application/pdf", "data": uploaded_file.getvalue()}
                st.success(f"📄 PDF: {uploaded_file.name}")
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image Loaded", use_container_width=True)
                visual_content = image

    with tab_cam:
        camera_photo = st.camera_input("Take a photo")
        if camera_photo:
            image = Image.open(camera_photo)
            visual_content = image
            st.image(image, caption="Camera Photo", use_container_width=True)

    with tab_mic:
        voice_recording = st.audio_input("Record a question")
        if voice_recording:
            audio_content = {"mime_type": "audio/wav", "data": voice_recording.read()} 
            st.audio(voice_recording)
            st.success("Audio captured!")

    st.divider()
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 5. MAIN CHAT LOGIC
# -----------------------------------------------------------------------------
mode_label = "Text"
if visual_content: mode_label = "Vision"
if audio_content: mode_label = "Voice"

st.title("⚛️ H2Physics Feynman Bot")
st.caption(f"Topic: **{topic}** | Mode: **{mode_label}**")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello JPJC Physics students! I can **find diagrams**, **plot graphs**, and **see** your work. What can I explain?"})

for msg in st.session_state.messages:
    display_message(msg["role"], msg["content"], enable_voice)

user_input = st.chat_input("Type OR Record/Upload...")

if user_input or audio_content or visual_content:
    
    user_display_text = user_input if user_input else ""
    if audio_content and not user_input: user_display_text = "🎤 *(Sent Audio Message)*"
    elif visual_content and not user_input: user_display_text = "📸 *(Sent Image/PDF)*"
    
    if user_display_text:
        st.session_state.messages.append({"role": "user", "content": user_display_text})
        with st.chat_message("user"):
            st.markdown(user_display_text)

    if not api_key:
        st.error("Key missing.")
        st.stop()

    try:
        genai.configure(api_key=api_key)
        
        # --- MODEL: Using 2.5-flash as requested ---
        model_name = "gemini-2.5-flash" 
        
        model = genai.GenerativeModel(
            model_name=model_name, 
            system_instruction=SEAB_H2_MASTER_INSTRUCTIONS
        )
        
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages if m['role'] != 'system'])
        final_prompt = []
        
        if visual_content:
            final_prompt.append(visual_content)
            final_prompt.append(f"Analyze this image/document. [Context: {topic}]")
        
        if audio_content:
            final_prompt.append(audio_content)
            final_prompt.append(f"Listen to this student's question about {topic}. Respond textually.")

        if user_input:
            final_prompt.append(f"USER TEXT: {user_input}")

        final_prompt.append(f"Conversation History:\n{history_text}\n\nASSISTANT:")

        with st.spinner("Processing..."):
            response = model.generate_content(final_prompt)
        
        display_message("assistant", response.text, enable_voice)
        st.session_state.messages.append({"role": "assistant", "content": response.text})


    except Exception as e:
        # --- ERROR HANDLING ---
        st.error(f"❌ Error: {e}")
        
        # Specific help for "404" errors (Model not found)
        if "404" in str(e) or "not found" in str(e).lower() or "not supported" in str(e).lower():
            st.warning(f"⚠️ Model '{model_name}' failed. Listing available models for your API Key...")
            try:
                # Ask Google: "Which models can I use?"
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                
                # Show the valid list to the user
                if available_models:
                    st.success(f"✅ Your Key works! Available models:")
                    st.code("\n".join(available_models))
                    st.info("Update 'model_name' in line 165 of app.py to one of these.")
                else:
                    st.error("❌ Your API Key has NO access to content generation models.")
            except Exception as inner_e:
                st.error(f"Could not list models: {inner_e}")
