import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL   = 'mistralai/Mistral-7B-Instruct-v0.2'
ADAPTER_PATH = r'checkpoints\checkpoint-800'

st.set_page_config(page_title='The Dialectic Assembly', page_icon='🏛️', layout='wide')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background: #f8f7f4; }
.stApp { background: #f8f7f4; }



.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem; font-weight: 600;
    letter-spacing: 4px; text-transform: uppercase;
    color: #1a1a1a; text-align: center;
    padding: 1.5rem 0 0.2rem;
}
.page-sub {
    font-size: 0.78rem; color: #888; text-align: center;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1.5rem;
}
.arena {
    background: #ffffff; border: 1px solid #e8e4df;
    border-radius: 16px; padding: 2rem 2rem 1.5rem; margin-bottom: 1.5rem;
}
.pixel-stage { display: flex; justify-content: center; margin-bottom: 0.5rem; }

/* Loading overlay */
.loading-overlay {
    background: #ffffff; border: 1px solid #e8e4df;
    border-radius: 16px; padding: 2rem;
    text-align: center; margin-bottom: 1.5rem;
}
.loading-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; letter-spacing: 3px;
    text-transform: uppercase; color: #888;
    margin-bottom: 1rem;
}
.loading-bar-track {
    width: 240px; height: 4px; background: #e8e4df;
    border-radius: 2px; margin: 0 auto 0.75rem; overflow: hidden;
}
.loading-bar-fill {
    height: 100%; width: 60%; background: #1a1a1a;
    border-radius: 2px;
    animation: loadpulse 1.4s ease-in-out infinite;
}
@keyframes loadpulse {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(300%); }
}
.loading-sub {
    font-size: 0.72rem; color: #bbb; letter-spacing: 1px;
}

/* Chat feed */
.chat-feed {
    background: #ffffff; border: 1px solid #e8e4df;
    border-radius: 16px; padding: 1.5rem 2rem; margin-top: 0;
}
.round-label {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 3px; color: #bbb;
    text-transform: uppercase;
    border-top: 1px dashed #e8e4df;
    padding-top: 1rem; margin: 1rem 0 0.5rem;
}
.chat-row {
    display: flex; align-items: flex-start; gap: 12px; margin-bottom: 1rem;
}
.chat-row-right {
    display: flex; align-items: flex-start; gap: 12px;
    margin-bottom: 1rem; flex-direction: row-reverse;
}
.avatar-dem {
    width: 36px; height: 36px; border-radius: 6px;
    background: #EBF4FF; border: 1.5px solid #BDDAFF;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.avatar-rep {
    width: 36px; height: 36px; border-radius: 6px;
    background: #FFF0EE; border: 1.5px solid #FFCDC7;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.chat-content-dem { flex: 1; }
.chat-content-rep { flex: 1; text-align: right; }
.speaker-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px;
}
.tag-dem { color: #2563EB; }
.tag-rep { color: #DC2626; text-align: right; }
.bubble-dem {
    background: #EBF4FF; border: 1.5px solid #BDDAFF;
    border-radius: 4px 16px 16px 16px;
    padding: 0.85rem 1rem;
    font-size: 0.92rem; line-height: 1.65; color: #1a2a3a;
    display: inline-block; max-width: 100%; text-align: left;
}
.bubble-rep {
    background: #FFF0EE; border: 1.5px solid #FFCDC7;
    border-radius: 16px 4px 16px 16px;
    padding: 0.85rem 1rem;
    font-size: 0.92rem; line-height: 1.65; color: #2a1a1a;
    display: inline-block; max-width: 100%; text-align: left;
}
.thinking-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 0.75rem; opacity: 0.5;
}
.thinking-row-right {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 0.75rem; opacity: 0.5; flex-direction: row-reverse;
}
.thinking-dots {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #aaa; letter-spacing: 2px;
    
div[data-testid="stStatusWidget"] {
    display: none !important;
}    
}
</style>
""", unsafe_allow_html=True)


def pixel_stage(dem_talking=False, rep_talking=False):
    dem_bubble = '💬' if dem_talking else ''
    rep_bubble = '💬' if rep_talking else ''
    return f"""
<div class="pixel-stage">
<svg width="520" height="210" viewBox="0 0 520 210" xmlns="http://www.w3.org/2000/svg" style="image-rendering:pixelated">
  <rect x="60" y="175" width="400" height="8" fill="#e8e4df"/>
  <rect x="60" y="183" width="400" height="4" fill="#d4cfc9"/>
  <rect x="150" y="145" width="220" height="12" rx="2" fill="#c8b89a"/>
  <rect x="155" y="157" width="210" height="6" rx="1" fill="#b5a488"/>
  <rect x="165" y="163" width="8" height="12" fill="#b5a488"/>
  <rect x="347" y="163" width="8" height="12" fill="#b5a488"/>
  <rect x="215" y="133" width="4" height="14" fill="#888"/>
  <rect x="211" y="129" width="12" height="8" rx="3" fill="#666"/>
  <rect x="295" y="133" width="4" height="14" fill="#888"/>
  <rect x="291" y="129" width="12" height="8" rx="3" fill="#666"/>
  <rect x="230" y="137" width="6" height="10" rx="1" fill="#a8d4f5" opacity="0.7"/>
  <rect x="280" y="137" width="6" height="10" rx="1" fill="#a8d4f5" opacity="0.7"/>
  <rect x="96" y="169" width="8" height="6" rx="1" fill="#1a1a1a"/>
  <rect x="108" y="169" width="8" height="6" rx="1" fill="#1a1a1a"/>
  <rect x="97" y="153" width="7" height="18" fill="#1e3a6e"/>
  <rect x="109" y="153" width="7" height="18" fill="#1e3a6e"/>
  <rect x="92" y="117" width="30" height="40" rx="2" fill="#1e3a6e"/>
  <rect x="104" y="118" width="6" height="20" fill="#f0f0f0"/>
  <rect x="106" y="120" width="2" height="16" fill="#e63946"/>
  <rect x="80" y="119" width="12" height="28" rx="2" fill="#1e3a6e"/>
  <rect x="122" y="119" width="12" height="28" rx="2" fill="#1e3a6e"/>
  <rect x="80" y="145" width="12" height="8" rx="2" fill="#f4c99a"/>
  <rect x="122" y="145" width="12" height="8" rx="2" fill="#f4c99a"/>
  <rect x="103" y="107" width="8" height="12" fill="#f4c99a"/>
  <rect x="94" y="81" width="26" height="28" rx="4" fill="#f4c99a"/>
  <rect x="94" y="81" width="26" height="8" rx="3" fill="#5c3a1e"/>
  <rect x="94" y="81" width="4" height="16" fill="#5c3a1e"/>
  <rect x="100" y="93" width="4" height="4" rx="1" fill="#1a1a1a"/>
  <rect x="110" y="93" width="4" height="4" rx="1" fill="#1a1a1a"/>
  <rect x="101" y="101" width="12" height="3" rx="1" fill="#c0845a"/>
  <rect x="82" y="64" width="48" height="14" rx="3" fill="#EBF4FF" stroke="#BDDAFF" stroke-width="1"/>
  <text x="106" y="75" text-anchor="middle" font-family="monospace" font-size="8" font-weight="bold" fill="#2563EB">DEMOS</text>
  <text x="140" y="78" font-size="16" opacity="{'1' if dem_talking else '0'}">{dem_bubble}</text>
  <rect x="396" y="169" width="8" height="6" rx="1" fill="#1a1a1a"/>
  <rect x="408" y="169" width="8" height="6" rx="1" fill="#1a1a1a"/>
  <rect x="397" y="153" width="7" height="18" fill="#6e1e1e"/>
  <rect x="409" y="153" width="7" height="18" fill="#6e1e1e"/>
  <rect x="392" y="117" width="30" height="40" rx="2" fill="#6e1e1e"/>
  <rect x="404" y="118" width="6" height="20" fill="#f0f0f0"/>
  <rect x="406" y="120" width="2" height="16" fill="#1e3a6e"/>
  <rect x="380" y="119" width="12" height="28" rx="2" fill="#6e1e1e"/>
  <rect x="422" y="119" width="12" height="28" rx="2" fill="#6e1e1e"/>
  <rect x="380" y="145" width="12" height="8" rx="2" fill="#f4c99a"/>
  <rect x="422" y="145" width="12" height="8" rx="2" fill="#f4c99a"/>
  <rect x="403" y="107" width="8" height="12" fill="#f4c99a"/>
  <rect x="394" y="81" width="26" height="28" rx="4" fill="#f4c99a"/>
  <rect x="394" y="81" width="26" height="8" rx="3" fill="#a0a0a0"/>
  <rect x="394" y="81" width="4" height="14" fill="#a0a0a0"/>
  <rect x="416" y="81" width="4" height="14" fill="#a0a0a0"/>
  <rect x="400" y="93" width="4" height="4" rx="1" fill="#1a1a1a"/>
  <rect x="410" y="93" width="4" height="4" rx="1" fill="#1a1a1a"/>
  <rect x="401" y="101" width="12" height="2" rx="1" fill="#c0845a"/>
  <rect x="386" y="64" width="42" height="14" rx="3" fill="#FFF0EE" stroke="#FFCDC7" stroke-width="1"/>
  <text x="407" y="75" text-anchor="middle" font-family="monospace" font-size="8" font-weight="bold" fill="#DC2626">ARES</text>
  <text x="372" y="78" font-size="16" opacity="{'1' if rep_talking else '0'}">{rep_bubble}</text>
  <rect x="248" y="62" width="24" height="24" rx="12" fill="#f8f7f4" stroke="#e8e4df" stroke-width="1"/>
  <text x="260" y="78" text-anchor="middle" font-size="14" fill="#888">⚖</text>
</svg>
</div>"""


def loading_overlay(message="Warming up the debaters…"):
    return f"""
<div class="loading-overlay">
  <div class="loading-title">Please wait</div>
  <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
  <div class="loading-sub">{message}</div>
</div>"""


@st.cache_resource
def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map='auto', trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    mdl = PeftModel.from_pretrained(mdl, ADAPTER_PATH)
    mdl.eval()
    return mdl, tok


def generate(prompt, model, tokenizer, max_tokens=120):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.85,
                             do_sample=True, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).split('[/INST]')[-1].strip()


def agent_speak(name, ideology, topic, history, model, tokenizer):
    party = 'Democratic' if ideology == 'democrat' else 'Republican'
    opponent = 'ARES' if name == 'DEMOS' else 'DEMOS'
    history_text = ''
    if history:
        history_text = '\nDebate so far:\n' + '\n'.join(f"{t['agent']}: {t['text']}" for t in history)
    instruction = (f'Open the debate on this topic: "{topic}". State your position clearly.' if not history
                   else f'Respond directly to {opponent}. Quote what they said and attack it. Be confrontational.')
    prompt = (f'<s>[INST] You are {name}, a {party} senator debating the following topic: "{topic}".\n'
              f'Stay STRICTLY on this topic.\n{history_text}\n\n'
              f'{instruction} Under 80 words. [/INST]')
    return generate(prompt, model, tokenizer)


def dem_row(text):
    return f"""
<div class="chat-row">
  <div class="avatar-dem">🧑</div>
  <div class="chat-content-dem">
    <div class="speaker-tag tag-dem">DEMOS · Democrat</div>
    <div class="bubble-dem">{text}</div>
  </div>
</div>"""

def rep_row(text):
    return f"""
<div class="chat-row-right">
  <div class="avatar-rep">👴</div>
  <div class="chat-content-rep">
    <div class="speaker-tag tag-rep">ARES · Republican</div>
    <div class="bubble-rep">{text}</div>
  </div>
</div>"""

def thinking_dem():
    return """
<div class="thinking-row">
  <div class="avatar-dem" style="opacity:0.4">🧑</div>
  <div class="thinking-dots">typing...</div>
</div>"""

def thinking_rep():
    return """
<div class="thinking-row-right">
  <div class="avatar-rep" style="opacity:0.4">👴</div>
  <div class="thinking-dots">typing...</div>
</div>"""


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">The Dialectic Assembly</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">AI Political Debate Simulator</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")

    # ── Free-text topic input ──────────────────────────────────────────────────
    topic = st.text_area(
        'Debate topic',
        placeholder='e.g. What do you think about immigration policy?',
        height=100,
        help='Type any topic or question you want the two senators to debate.'
    )

    # ── Unlimited rounds number input ──────────────────────────────────────────
    rounds = st.number_input(
        'Debate rounds',
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help='How many back-and-forth rounds after the opening statement.'
    )

    st.markdown("---")
    st.markdown("**DEMOS** 🧑 — Democrat senator")
    st.markdown("**ARES** 👴 — Republican senator")
    st.markdown("---")
    start = st.button('▶ Start debate', use_container_width=True, type='primary',
                      disabled=not topic.strip())

# ── ARENA ─────────────────────────────────────────────────────────────────────
stage_ph = st.empty()
stage_ph.markdown('<div class="arena">' + pixel_stage() + '</div>', unsafe_allow_html=True)

# ── CHAT FEED ─────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-feed">', unsafe_allow_html=True)
chat_ph = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

if start:
    if not topic.strip():
        st.warning("Please enter a debate topic first.")
        st.stop()

    # ── Show styled loading overlay while model loads ──────────────────────────
    stage_ph.markdown(loading_overlay("Loading AI model — this may take a minute…"), unsafe_allow_html=True)
    model, tokenizer = load_model()

    # Restore arena once model is ready
    stage_ph.markdown('<div class="arena">' + pixel_stage() + '</div>', unsafe_allow_html=True)

    history = []
    chat_html = ''

    def update_chat(html):
        chat_ph.markdown(html, unsafe_allow_html=True)

    # Opening — Democrat talking
    stage_ph.markdown('<div class="arena">' + pixel_stage(dem_talking=True) + '</div>', unsafe_allow_html=True)
    update_chat(thinking_dem())
    dem_text = agent_speak('DEMOS', 'democrat', topic, history, model, tokenizer)
    history.append({'agent': 'DEMOS', 'text': dem_text})
    chat_html += dem_row(dem_text)
    stage_ph.markdown('<div class="arena">' + pixel_stage() + '</div>', unsafe_allow_html=True)
    update_chat(chat_html)

    for r in range(rounds):
        round_html = f'<div class="round-label">Round {r + 1}</div>'
        chat_html += round_html
        update_chat(chat_html)

        # Republican
        stage_ph.markdown('<div class="arena">' + pixel_stage(rep_talking=True) + '</div>', unsafe_allow_html=True)
        update_chat(chat_html + thinking_rep())
        rep_text = agent_speak('ARES', 'republican', topic, history, model, tokenizer)
        history.append({'agent': 'ARES', 'text': rep_text})
        chat_html += rep_row(rep_text)
        stage_ph.markdown('<div class="arena">' + pixel_stage() + '</div>', unsafe_allow_html=True)
        update_chat(chat_html)

        # Democrat responds
        stage_ph.markdown('<div class="arena">' + pixel_stage(dem_talking=True) + '</div>', unsafe_allow_html=True)
        update_chat(chat_html + thinking_dem())
        dem_text = agent_speak('DEMOS', 'democrat', topic, history, model, tokenizer)
        history.append({'agent': 'DEMOS', 'text': dem_text})
        chat_html += dem_row(dem_text)
        stage_ph.markdown('<div class="arena">' + pixel_stage() + '</div>', unsafe_allow_html=True)
        update_chat(chat_html)

    chat_html += '<div class="round-label">Debate concluded</div>'
    update_chat(chat_html)