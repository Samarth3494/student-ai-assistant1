import streamlit as st
import requests
import json
import hashlib
import hmac
import secrets
import re
from pathlib import Path
from datetime import datetime
from pypdf import PdfReader

# =========================================================
# SETTINGS
# =========================================================
st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
USERS_FILE = BASE_DIR / "users.json"
LEGACY_CHAT_FILE = BASE_DIR / "chat_history.json"
MIGRATION_FILE = BASE_DIR / "legacy_migration.json"

OLLAMA_URL = "https://ollama.com/api/chat"
MODEL = "gemma4:31b"

MAX_PDF_CHARS = 20000
MAX_PDF_PAGES = 30
PASSWORD_ITERATIONS = 260000


# =========================================================
# HELPERS: JSON, LOGIN, CHAT STORAGE
# =========================================================
def read_json(path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
    except (OSError, json.JSONDecodeError):
        pass
    return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_users():
    data = read_json(USERS_FILE, {})
    return data if isinstance(data, dict) else {}


def save_users(users):
    write_json(USERS_FILE, users)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return salt, hashed


def verify_password(password, salt, expected_hash):
    _, actual_hash = hash_password(password, salt)
    return hmac.compare_digest(actual_hash, expected_hash)


def safe_username(username):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", username.lower())


def chat_file_for(username):
    return BASE_DIR / f"chat_history_{safe_username(username)}.json"


def migrate_legacy_chats_for_first_user(username):
    """Copy old shared chat history once, without deleting the original file."""
    migration = read_json(MIGRATION_FILE, {})
    if migration.get("done"):
        return

    destination = chat_file_for(username)
    if LEGACY_CHAT_FILE.exists() and not destination.exists():
        legacy = read_json(LEGACY_CHAT_FILE, {})
        if isinstance(legacy, dict):
            write_json(destination, legacy)

    write_json(MIGRATION_FILE, {"done": True, "first_user": username})


def load_user_chats(username):
    data = read_json(chat_file_for(username), {})
    return data if isinstance(data, dict) else {}


def save_chats():
    username = st.session_state.get("auth_user")
    if username:
        try:
            write_json(chat_file_for(username), st.session_state.get("chats", {}))
        except OSError as error:
            st.error(f"Chat history save करता आली नाही: {error}")


def create_chat():
    chat_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "pdf_name": "",
        "pdf_text": "",
        "pdf_hash": "",
    }
    st.session_state.current_chat = chat_id
    save_chats()


def get_current_chat():
    chat_id = st.session_state.get("current_chat")
    if chat_id and chat_id in st.session_state.get("chats", {}):
        return st.session_state.chats[chat_id]
    return None


def save_message(role, content):
    chat = get_current_chat()
    if chat is None:
        create_chat()
        chat = get_current_chat()

    chat.setdefault("messages", [])
    chat["messages"].append({"role": role, "content": content})

    if role == "user" and chat.get("title", "New Chat") == "New Chat":
        title = content.strip().replace("\n", " ")
        chat["title"] = title[:35] + ("..." if len(title) > 35 else "")
    save_chats()


def ollama_chat(messages, timeout=180):
    # The key is stored in Streamlit Cloud Secrets, never in this source file.
    try:
        api_key = st.secrets["OLLAMA_API_KEY"]
    except Exception as error:
        raise RuntimeError(
            "OLLAMA_API_KEY set केलेली नाही. Streamlit Cloud → Manage app → "
            "Settings → Secrets मध्ये API key जोडा."
        ) from error

    if not api_key or not str(api_key).strip():
        raise RuntimeError(
            "OLLAMA_API_KEY रिकामी आहे. Streamlit Cloud → Manage app → "
            "Settings → Secrets मध्ये योग्य API key जोडा."
        )

    response = requests.post(
        OLLAMA_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "Ollama Cloud कडून अपेक्षित उत्तर मिळाले नाही. Model नाव आणि API access तपासा."
        ) from error


def make_system_prompt(pdf_text="", pdf_name=""):
    prompt = (
        "You are a helpful Student AI Assistant. "
        "Explain college subjects in simple language with examples. "
        "If the user asks in Marathi, answer in Marathi. "
        "If the user asks in English, answer in English. "
        "Be accurate and do not invent facts."
    )
    if pdf_text:
        prompt += (
            f"\nThe user attached a PDF named {pdf_name}. When the question is about "
            "this PDF, use the extracted text as the primary source. If the answer "
            "is not present in the text, say so clearly. Do not claim the PDF says "
            "something it does not.\n\nPDF TEXT:\n" + pdf_text
        )
    return prompt


def generate_json_from_ai(prompt):
    raw = ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    "Follow the user's instructions carefully. If JSON is requested, "
                    "return only valid JSON with no markdown fences or extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        timeout=240,
    )
    # Be tolerant if the model accidentally wraps JSON in markdown.
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def get_pdf_context():
    chat = get_current_chat()
    if chat and chat.get("pdf_text"):
        return chat.get("pdf_text", ""), chat.get("pdf_name", "uploaded PDF")
    return "", ""


# =========================================================
# SESSION STATE
# =========================================================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "page" not in st.session_state:
    st.session_state.page = "Chat"
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None
if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = ""


# =========================================================
# PROFESSIONAL LIGHT UI
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background: #f7f8fc; color: #202536; }
    [data-testid="stHeader"] { background: rgba(247,248,252,0.96); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6e8f0;
    }
    .block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: #202536; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e6e8f0;
        padding: 16px; border-radius: 14px;
    }
    .stButton > button {
        border-radius: 10px; min-height: 40px;
        border: 1px solid #d9deeb; font-weight: 500;
    }
    .stButton > button[kind="primary"] {
        background: #4f46e5; border-color: #4f46e5; color: white;
    }
    [data-testid="stChatMessage"] {
        background: #ffffff; border: 1px solid #eceef5;
        border-radius: 14px; padding: 12px;
    }
    [data-testid="stChatInput"] {
        background: #ffffff; border: 1px solid #d9deeb; border-radius: 18px;
    }
    .hero {
        padding: 22px 24px; background: white;
        border: 1px solid #e6e8f0; border-radius: 18px; margin-bottom: 18px;
    }
    .muted { color: #697386; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOGIN / SIGNUP SCREEN
# =========================================================
if not st.session_state.auth_user:
    st.markdown(
        "<div class='hero'><h1>🎓 Student AI Assistant</h1>"
        "<p class='muted'>Your personal AI study partner</p></div>",
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.subheader("Login / Signup")
        st.caption("तुझे account वापरून website सुरू कर.")
        mode = st.radio(
            "Account",
            ["Login", "Signup"],
            horizontal=True,
            index=0 if st.session_state.auth_mode == "Login" else 1,
        )
        st.session_state.auth_mode = mode

        with st.form("auth_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="उदा. student123")
            password = st.text_input("Password", type="password")
            if mode == "Signup":
                confirm_password = st.text_input("Confirm password", type="password")
            else:
                confirm_password = ""

            submitted = st.form_submit_button(
                "Login" if mode == "Login" else "Create Account",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            users = load_users()
            username_clean = username.strip().lower()

            if not re.fullmatch(r"[a-zA-Z0-9_-]{3,30}", username_clean):
                st.error("Username 3 ते 30 characters असावा. Letters, numbers, _ आणि - वापर.")
            elif len(password) < 6:
                st.error("Password किमान 6 characters चा असावा.")
            elif mode == "Signup":
                if password != confirm_password:
                    st.error("दोन्ही passwords जुळत नाहीत.")
                elif username_clean in users:
                    st.error("हा username आधीपासून आहे. दुसरा username वापर.")
                else:
                    salt, hashed = hash_password(password)
                    users[username_clean] = {
                        "salt": salt,
                        "password_hash": hashed,
                        "created": datetime.now().isoformat(timespec="seconds"),
                    }
                    save_users(users)
                    migrate_legacy_chats_for_first_user(username_clean)
                    st.session_state.auth_user = username_clean
                    st.session_state.chats = load_user_chats(username_clean)
                    st.session_state.current_chat = None
                    if not st.session_state.chats:
                        create_chat()
                    st.success("Account तयार झाले!")
                    st.rerun()
            else:
                user_record = users.get(username_clean)
                if not user_record or not verify_password(
                    password,
                    user_record.get("salt", ""),
                    user_record.get("password_hash", ""),
                ):
                    st.error("Username किंवा password चुकीचा आहे.")
                else:
                    migrate_legacy_chats_for_first_user(username_clean)
                    st.session_state.auth_user = username_clean
                    st.session_state.chats = load_user_chats(username_clean)
                    st.session_state.current_chat = None
                    if not st.session_state.chats:
                        create_chat()
                    st.rerun()

    st.stop()


# =========================================================
# SIDEBAR NAVIGATION + CHAT LIST
# =========================================================
with st.sidebar:
    st.markdown("# 🎓 Student AI")
    st.caption(f"Welcome, {st.session_state.auth_user}")
    st.divider()

    page_choice = st.radio(
        "Navigation",
        ["Chat", "AI Quiz Generator", "AI Notes Generator", "Dashboard"],
        index=[
            "Chat",
            "AI Quiz Generator",
            "AI Notes Generator",
            "Dashboard",
        ].index(st.session_state.page)
        if st.session_state.page in [
            "Chat",
            "AI Quiz Generator",
            "AI Notes Generator",
            "Dashboard",
        ]
        else 0,
    )
    st.session_state.page = page_choice

    st.divider()
    if st.button("＋ New Chat", use_container_width=True):
        create_chat()
        st.session_state.page = "Chat"
        st.rerun()

    st.markdown("### Your Chats")
    chat_items = list(st.session_state.chats.items())
    chat_items.reverse()

    if not chat_items:
        st.caption("तुझे chats इथे दिसतील.")

    for chat_id, chat_data in chat_items:
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                "💬 " + chat_data.get("title", "New Chat"),
                key="open_" + chat_id,
                use_container_width=True,
            ):
                st.session_state.current_chat = chat_id
                st.session_state.page = "Chat"
                st.rerun()
        with col2:
            if st.button("🗑️", key="delete_" + chat_id, help="Delete this chat"):
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = None
                save_chats()
                st.rerun()

    st.divider()
    st.caption(f"AI Model: {MODEL} · Ollama")
    if st.button("🗑️ Clear All Chats", use_container_width=True):
        st.session_state.chats = {}
        st.session_state.current_chat = None
        save_chats()
        st.rerun()

    if st.button("Log out", use_container_width=True):
        save_chats()
        st.session_state.auth_user = None
        st.session_state.chats = {}
        st.session_state.current_chat = None
        st.session_state.page = "Chat"
        st.session_state.quiz_data = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_result = None
        st.session_state.generated_notes = ""
        st.rerun()


# Make sure there is a current conversation.
if (
    st.session_state.current_chat is None
    or st.session_state.current_chat not in st.session_state.chats
):
    if st.session_state.chats:
        st.session_state.current_chat = list(st.session_state.chats.keys())[-1]
    else:
        create_chat()

chat = get_current_chat()
chat.setdefault("messages", [])
chat.setdefault("pdf_name", "")
chat.setdefault("pdf_text", "")
chat.setdefault("pdf_hash", "")

st.markdown(
    "<div class='hero'><h1>🎓 AI Student Assistant</h1>"
    "<p class='muted'>Study smarter with your notes, AI quizzes and exam preparation.</p></div>",
    unsafe_allow_html=True,
)


# =========================================================
# CHAT PAGE + PDF UPLOAD
# =========================================================
if st.session_state.page == "Chat":
    st.subheader("💬 AI Chat")
    st.caption("प्रश्न विचार किंवा PDF upload करून त्यावर आधारित उत्तरे मिळव.")

    uploaded_pdf = st.file_uploader(
        "📄 PDF upload करा",
        type=["pdf"],
        key="pdf_" + st.session_state.current_chat,
    )

    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if pdf_hash != chat.get("pdf_hash", ""):
            try:
                reader = PdfReader(uploaded_pdf)
                if reader.is_encrypted:
                    st.error("हा PDF password-protected आहे. Unlocked PDF upload कर.")
                else:
                    extracted_pages = []
                    for page in reader.pages[:MAX_PDF_PAGES]:
                        extracted_pages.append(page.extract_text() or "")
                    pdf_text = "\n".join(extracted_pages).strip()
                    pdf_text = pdf_text[:MAX_PDF_CHARS]

                    if not pdf_text:
                        st.error(
                            "PDF मधून readable text मिळाला नाही. हा scanned PDF असू शकतो; "
                            "सध्या OCR feature नाही."
                        )
                    else:
                        chat["pdf_name"] = uploaded_pdf.name
                        chat["pdf_text"] = pdf_text
                        chat["pdf_hash"] = pdf_hash
                        save_chats()
                        st.success(f"PDF तयार आहे: {uploaded_pdf.name}")
                        if len(reader.pages) > MAX_PDF_PAGES:
                            st.info(f"फक्त पहिली {MAX_PDF_PAGES} pages process केली.")
            except Exception as error:
                st.error(f"PDF वाचता आला नाही: {error}")

    if chat.get("pdf_text"):
        st.info(f"📎 या chat ला जोडलेला PDF: {chat['pdf_name']}")
        if st.button("या chat मधून PDF काढा", key="remove_pdf"):
            chat["pdf_name"] = ""
            chat["pdf_text"] = ""
            chat["pdf_hash"] = ""
            save_chats()
            st.rerun()

    st.divider()
    if not chat["messages"]:
        st.markdown("### आज काय शिकायचे आहे?")
        st.write("उदा. `Explain cloud computing in simple Marathi.`")

    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("तुझा प्रश्न इथे लिही...")
    if question:
        save_message("user", question)
        with st.chat_message("user"):
            st.markdown(question)

        # Refresh chat object after save_message.
        chat = get_current_chat()
        messages = [
            {"role": "system", "content": make_system_prompt(
                chat.get("pdf_text", ""), chat.get("pdf_name", "")
            )}
        ]
        messages.extend(chat["messages"])

        try:
            with st.chat_message("assistant"):
                with st.spinner("AI उत्तर तयार करत आहे..."):
                    answer = ollama_chat(messages)
                    st.markdown(answer)
            save_message("assistant", answer)
        except requests.exceptions.ConnectionError:
            st.error("Ollama Cloud शी connection होत नाही. Internet किंवा Ollama API status तपास.")
        except RuntimeError as error:
            st.error(str(error))
        except requests.exceptions.Timeout:
            st.error("AI ला उत्तर द्यायला जास्त वेळ लागला. पुन्हा प्रयत्न कर.")
        except requests.exceptions.RequestException as error:
            st.error(f"AI connection error: {error}")
        except Exception as error:
            st.error(f"काहीतरी error आला: {error}")


# =========================================================
# AI QUIZ GENERATOR
# =========================================================
elif st.session_state.page == "AI Quiz Generator":
    st.subheader("📝 AI Quiz Generator")
    st.write("एखाद्या topic वर किंवा सध्याच्या chat मध्ये जोडलेल्या PDF वरून quiz तयार कर.")

    pdf_text, pdf_name = get_pdf_context()
    if pdf_text:
        st.success(f"Current chat PDF वापरता येईल: {pdf_name}")
    topic = st.text_input(
        "Topic / Subject",
        placeholder="उदा. Cloud Computing, Machine Learning",
        key="quiz_topic",
    )
    num_questions = st.selectbox("प्रश्नांची संख्या", [5, 10, 15], index=0)
    quiz_language = st.selectbox("Quiz language", ["Marathi", "English"], index=0)
    use_pdf = st.checkbox("PDF जोडलेला असेल तर त्यातील माहिती वापर", value=bool(pdf_text))

    if st.button("Generate Quiz", type="primary", use_container_width=True):
        if not topic.strip() and not (use_pdf and pdf_text):
            st.warning("Topic लिही किंवा Chat page वर PDF upload कर.")
        else:
            source = ""
            if use_pdf and pdf_text:
                source = f"\nUse this PDF text as the main source:\n{pdf_text}\n"
            else:
                source = f"\nTopic: {topic.strip()}\n"

            language_instruction = (
                "Write all question and option text in Marathi."
                if quiz_language == "Marathi"
                else "Write all question and option text in English."
            )
            prompt = f"""
Create exactly {num_questions} multiple-choice quiz questions for a college student.
{language_instruction}
{source}
Return valid JSON in this exact structure:
{{
  "title": "Quiz title",
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer_index": 0,
      "explanation": "Short explanation"
    }}
  ]
}}
Rules:
- Each question must have exactly four options.
- answer_index must be an integer 0, 1, 2, or 3.
- Use only source-supported facts if PDF text is supplied.
- Do not include markdown or any text outside JSON.
"""
            try:
                with st.spinner("AI quiz तयार करत आहे..."):
                    quiz = generate_json_from_ai(prompt)
                questions = quiz.get("questions", [])
                valid = (
                    isinstance(questions, list)
                    and len(questions) >= 1
                    and all(
                        isinstance(q, dict)
                        and isinstance(q.get("question"), str)
                        and isinstance(q.get("options"), list)
                        and len(q["options"]) == 4
                        and isinstance(q.get("answer_index"), int)
                        and 0 <= q["answer_index"] < 4
                        for q in questions
                    )
                )
                if not valid:
                    st.error("AI कडून योग्य quiz format मिळाला नाही. पुन्हा Generate Quiz कर.")
                else:
                    st.session_state.quiz_data = quiz
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_result = None
                    st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("Ollama Cloud शी connection होत नाही. Internet किंवा Ollama API status तपास.")
            except RuntimeError as error:
                st.error(str(error))
            except requests.exceptions.RequestException as error:
                st.error(f"AI connection error: {error}")
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                st.error(f"Quiz तयार करताना JSON error आला: {error}. पुन्हा प्रयत्न कर.")
            except Exception as error:
                st.error(f"Quiz तयार करता आला नाही: {error}")

    quiz = st.session_state.quiz_data
    if quiz:
        st.divider()
        st.markdown(f"### {quiz.get('title', 'Your Quiz')}")
        with st.form("quiz_answer_form"):
            answers = {}
            for index, question in enumerate(quiz["questions"]):
                st.markdown(f"**Q{index + 1}. {question['question']}**")
                options = question["options"]
                choice = st.radio(
                    f"Q{index + 1} answer",
                    options,
                    index=None,
                    key=f"quiz_answer_{index}_{hashlib.md5(question['question'].encode()).hexdigest()[:8]}",
                    label_visibility="collapsed",
                )
                answers[str(index)] = choice
                st.write("")
            submitted_quiz = st.form_submit_button("Submit Quiz", type="primary")

        if submitted_quiz:
            score = 0
            for index, question in enumerate(quiz["questions"]):
                selected = answers.get(str(index))
                correct = question["options"][question["answer_index"]]
                if selected == correct:
                    score += 1
            st.session_state.quiz_answers = answers
            st.session_state.quiz_result = score

        if st.session_state.quiz_result is not None:
            score = st.session_state.quiz_result
            total = len(quiz["questions"])
            st.metric("Your Score", f"{score} / {total}")
            st.progress(score / total if total else 0)
            for index, question in enumerate(quiz["questions"]):
                selected = st.session_state.quiz_answers.get(str(index))
                correct = question["options"][question["answer_index"]]
                if selected == correct:
                    st.success(f"Q{index + 1}: बरोबर! {correct}")
                else:
                    st.error(
                        f"Q{index + 1}: तुझे उत्तर: {selected or 'उत्तर निवडले नाही'} | "
                        f"योग्य उत्तर: {correct}"
                    )
                if question.get("explanation"):
                    st.caption(question["explanation"])
            if st.button("Create Another Quiz"):
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_result = None
                st.rerun()


# =========================================================
# AI NOTES GENERATOR
# =========================================================
elif st.session_state.page == "AI Notes Generator":
    st.subheader("📚 AI Notes Generator")
    st.write("Topic किंवा uploaded PDF मधून अभ्यासासाठी notes तयार कर.")

    pdf_text, pdf_name = get_pdf_context()
    if pdf_text:
        st.success(f"Current chat PDF उपलब्ध आहे: {pdf_name}")
    notes_topic = st.text_input(
        "Topic / Unit",
        placeholder="उदा. Unit 1 - Cloud Computing Architecture",
        key="notes_topic",
    )
    notes_type = st.selectbox(
        "Notes type",
        [
            "Short Notes",
            "Detailed Notes",
            "Important Questions and Answers",
            "Exam Revision Points",
        ],
    )
    notes_language = st.selectbox("Notes language", ["Marathi", "English"])
    notes_use_pdf = st.checkbox(
        "PDF मधील माहिती वापर",
        value=bool(pdf_text),
        key="notes_use_pdf",
    )

    if st.button("Generate Notes", type="primary", use_container_width=True):
        if not notes_topic.strip() and not (notes_use_pdf and pdf_text):
            st.warning("Topic लिही किंवा Chat page वर PDF upload कर.")
        else:
            if notes_use_pdf and pdf_text:
                source = f"Use the following PDF text as the main source:\n{pdf_text}\n"
            else:
                source = f"Topic: {notes_topic.strip()}\n"

            lang = (
                "Write the notes in simple Marathi."
                if notes_language == "Marathi"
                else "Write the notes in clear, simple English."
            )
            prompt = f"""
You are a helpful college study-notes assistant.
{lang}
Create {notes_type} for a college student.
{source}
Requirements:
- Use clear headings and bullet points.
- Explain technical terms simply.
- Include examples where useful.
- If PDF text is supplied, do not invent details and clearly mention if a requested detail is absent.
- For important questions and answers, number each question and give a concise answer.
"""
            try:
                with st.spinner("Notes तयार होत आहेत..."):
                    st.session_state.generated_notes = ollama_chat(
                        [
                            {
                                "role": "system",
                                "content": "You create accurate, well-organized college study notes.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        timeout=240,
                    )
            except requests.exceptions.ConnectionError:
                st.error("Ollama Cloud शी connection होत नाही. Internet किंवा Ollama API status तपास.")
            except RuntimeError as error:
                st.error(str(error))
            except requests.exceptions.Timeout:
                st.error("AI ला जास्त वेळ लागला. छोटा topic देऊन पुन्हा प्रयत्न कर.")
            except requests.exceptions.RequestException as error:
                st.error(f"AI connection error: {error}")
            except Exception as error:
                st.error(f"Notes तयार करता आले नाहीत: {error}")

    if st.session_state.generated_notes:
        st.divider()
        st.markdown("### तुझे Notes")
        st.markdown(st.session_state.generated_notes)
        st.download_button(
            "⬇️ Download Notes (.txt)",
            data=st.session_state.generated_notes,
            file_name="student_ai_notes.txt",
            mime="text/plain",
            use_container_width=True,
        )


# =========================================================
# DASHBOARD
# =========================================================
elif st.session_state.page == "Dashboard":
    st.subheader("📊 Student Dashboard")
    total_chats = len(st.session_state.chats)
    total_messages = sum(
        len(item.get("messages", []))
        for item in st.session_state.chats.values()
    )
    chats_with_pdf = sum(
        1 for item in st.session_state.chats.values() if item.get("pdf_text")
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Chats", total_chats)
    col2.metric("Messages", total_messages)
    col3.metric("Chats with PDF", chats_with_pdf)

    st.markdown("### Quick Start")
    st.write("- **Chat:** AI ला अभ्यासाचे प्रश्न विचार.")
    st.write("- **AI Quiz Generator:** Topic किंवा PDF वरून MCQ test तयार कर.")
    st.write("- **AI Notes Generator:** Revision notes आणि exam answers तयार कर.")
    st.info(
        "टीप: AI कडून मिळालेली उत्तरे अभ्यासासाठी मदत म्हणून वापर. "
        "Exam किंवा assignment आधी textbook/teacher notes मधून महत्त्वाची माहिती verify कर."
    )
