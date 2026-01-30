import os
import re
import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Quiz (Qwen2) — MCQ + Scoring",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Powered Quiz — MCQ + Scoring")
st.caption("Qwen2-1.5B-Instruct (CPU) • Generates one MCQ at a time • Select option • Get score")

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    topic = st.text_input("Topic", value="Python Basics", help="e.g., Python Basics, Data Structures, Geography")
    max_new_tokens = st.slider("Max new tokens", 80, 1024, 250, 10)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, help="Lower → more deterministic")
    show_correct_on_submit = st.checkbox("Show correct answer after submit", value=True)
    st.markdown("---")
    st.caption("💡 Tip: If the question truncates, increase *max new tokens*.")

# -----------------------------
# Cache model on first use (CPU-friendly)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_id: str, max_new_tokens: int, temperature: float):
    # Optional: set HF cache
    # os.environ["HF_HOME"] = "D:/huggingface_cache"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
llm = load_model(MODEL_ID, max_new_tokens, temperature)

# -----------------------------
# Prompt template (strict format)
# -----------------------------
template = """<|im_start|>user
Generate 1 multiple choice question about {topic}.
Format the output exactly as follows:
Question: [Your question]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct Answer: [Letter Only]
<|im_end|>
<|im_start|>assistant
"""
PROMPT = PromptTemplate.from_template(template)
OUTPUT_PARSER = StrOutputParser()

# -----------------------------
# Session state
# -----------------------------
if "question" not in st.session_state:
    st.session_state.question = None  # dict: {"q": str, "opts": [A..D], "answer": "A"}
if "raw_output" not in st.session_state:
    st.session_state.raw_output = ""
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None  # "A"/"B"/"C"/"D"
if "total" not in st.session_state:
    st.session_state.total = 0
if "correct" not in st.session_state:
    st.session_state.correct = 0
if "incorrect" not in st.session_state:
    st.session_state.incorrect = 0
if "history" not in st.session_state:
    # list of ({q, opts, answer}, user_choice, is_correct)
    st.session_state.history = []
if "answered" not in st.session_state:
    st.session_state.answered = False  # submitted this question?

# -----------------------------
# Utilities
# -----------------------------
def parse_one_mcq(raw: str):
    """
    Parse a single-Q formatted block:
        Question: ...
        A) ...
        B) ...
        C) ...
        D) ...
        Correct Answer: X
    Returns: {"q": str, "opts": [A,B,C,D], "answer": "A"}
    Raises ValueError if parsing fails.
    """
    text = raw.strip()

    # Extract question line
    m_q = re.search(r"(?im)^question:\s*(.+)$", text)
    if not m_q:
        raise ValueError("Missing 'Question:' line.")
    q_text = m_q.group(1).strip()

    # Extract options A-D
    opts = []
    for lbl in ["A", "B", "C", "D"]:
        m_o = re.search(rf"(?im)^{lbl}\)\s*(.+)$", text)
        if not m_o:
            raise ValueError(f"Missing or malformed option {lbl}).")
        opts.append(m_o.group(1).strip())

    # Extract correct answer (letter)
    m_ans = re.search(r"(?im)^correct answer:\s*([ABCD])\b", text)
    if not m_ans:
        raise ValueError("Missing 'Correct Answer:' letter (A/B/C/D).")
    answer = m_ans.group(1).upper()

    return {"q": q_text, "opts": opts, "answer": answer}

def generate_new_question(current_topic: str):
    chain = PROMPT | llm | OUTPUT_PARSER
    raw = chain.invoke({"topic": current_topic})
    # Store raw for debugging
    st.session_state.raw_output = raw
    # Parse
    item = parse_one_mcq(raw)
    st.session_state.question = item
    st.session_state.user_choice = None
    st.session_state.answered = False

def reset_quiz():
    st.session_state.question = None
    st.session_state.raw_output = ""
    st.session_state.user_choice = None
    st.session_state.total = 0
    st.session_state.correct = 0
    st.session_state.incorrect = 0
    st.session_state.history = []
    st.session_state.answered = False

# -----------------------------
# Controls (top row)
# -----------------------------
c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("🆕 New Question", use_container_width=True):
        try:
            generate_new_question(topic)
            st.toast("Question generated!", icon="✅")
        except Exception as e:
            st.error(f"Generation/Parsing failed: {e}")

with c2:
    if st.button("🔄 Reset Quiz", use_container_width=True, type="secondary"):
        reset_quiz()
        st.toast("Quiz reset.", icon="♻️")

with c3:
    st.metric("Score", f"{st.session_state.correct} / {st.session_state.total}")
    st.caption(f"✅ Correct: {st.session_state.correct} • ❌ Incorrect: {st.session_state.incorrect}")

st.markdown("---")

# -----------------------------
# Render current question
# -----------------------------
if st.session_state.question:
    q = st.session_state.question

    st.subheader("📝 Current Question")
    st.write(q["q"])

    # Radio for options (A-D)
    labels = ["A", "B", "C", "D"]
    display_opts = [f"{labels[i]}) {opt}" for i, opt in enumerate(q["opts"])]

    # Preselect if already chosen
    default_idx = None
    if st.session_state.user_choice in labels:
        default_idx = labels.index(st.session_state.user_choice)

    choice_display = st.radio(
        "Choose one:",
        options=display_opts,
        index=default_idx if default_idx is not None else None,
        key="choice_radio",
    )

    # Store letter only
    if choice_display:
        st.session_state.user_choice = choice_display.split(")")[0]

    c_submit, c_next, c_show = st.columns([1,1,1])

    with c_submit:
        disabled_submit = st.session_state.answered or (st.session_state.user_choice is None)
        if st.button("✅ Submit", disabled=disabled_submit, use_container_width=True):
            # Evaluate
            ua = st.session_state.user_choice
            ans = q["answer"]
            is_correct = (ua == ans)

            st.session_state.total += 1
            if is_correct:
                st.session_state.correct += 1
                st.success("Correct! 🎉")
            else:
                st.error(f"Wrong. The correct answer was **{ans}**.")

            st.session_state.history.append((q, ua, is_correct))
            st.session_state.answered = True

    with c_next:
        disabled_next = not st.session_state.answered
        if st.button("⏭️ Next Question", disabled=disabled_next, use_container_width=True):
            try:
                generate_new_question(topic)
                st.toast("Next question ready!", icon="➡️")
            except Exception as e:
                st.error(f"Generation/Parsing failed: {e}")

    with c_show:
        if show_correct_on_submit and st.session_state.answered:
            st.info(f"✅ Correct Answer: **{q['answer']}**")

    # Show options with highlights after submission
    if st.session_state.answered:
        st.markdown("---")
        st.caption("Breakdown:")
        for lbl, opt in zip(labels, q["opts"]):
            bullet = f"{lbl}) {opt}"
            if lbl == q["answer"]:
                st.markdown(f"- ✅ **{bullet}** *(correct)*")
            elif st.session_state.user_choice == lbl:
                st.markdown(f"- ❌ ~~{bullet}~~ *(your choice)*")
            else:
                st.markdown(f"- {bullet}")

# -----------------------------
# Summary / History
# -----------------------------
st.markdown("---")
with st.expander("📊 Show Attempt History / Summary", expanded=False):
    if not st.session_state.history:
        st.info("No attempts yet.")
    else:
        for i, (item, ua, ok) in enumerate(st.session_state.history, start=1):
            status = "✅ Correct" if ok else "❌ Incorrect"
            st.markdown(f"**Q{i}. {status}** — {item['q']}")
            for lbl, opt in zip(["A", "B", "C", "D"], item["opts"]):
                bullet = f"{lbl}) {opt}"
                if lbl == item["answer"]:
                    st.markdown(f"- ✅ **{bullet}** *(correct)*")
                elif ua == lbl:
                    st.markdown(f"- ❌ ~~{bullet}~~ *(your choice)*")
                else:
                    st.markdown(f"- {bullet}")
            st.markdown("---")

# -----------------------------
# Debug raw output (optional)
# -----------------------------
with st.expander("🛠️ Debug: Raw Model Output"):
    if st.session_state.raw_output:
        st.code(st.session_state.raw_output)
    else:
        st.caption("Generate a question to see raw output.")