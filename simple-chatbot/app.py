"""
Streamlit UI for POMDP-Controlled Multilingual Chatbot
AA228/CS238 Project

This app demonstrates a chatbot that uses a POMDP to learn user language
preferences and adapt its response language (English, Chinese, mixed).

Features:
- Model selection (Together API or Local)
- Real-time POMDP belief visualization
- Action display showing why the bot chose its response mode
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="POMDP Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 POMDP Multilingual Chatbot")
st.caption("An adaptive chatbot that learns your English-Mandarin code-switching style")

# =============================================================================
# Sidebar: Model Selection & POMDP Display
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.radio(
        "Choose Model:",
        ["Llama 3.1 (Together API)", "Qwen 2.5 (Local)"],
        help="Together API requires TOGETHER_API_KEY. Local model is free but slower."
    )
    
    st.divider()
    
    # POMDP toggle
    use_pomdp = st.checkbox("Enable POMDP", value=True, 
                            help="Use POMDP for adaptive language selection")
    
    # Clear chat button
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if "bot" in st.session_state:
            st.session_state.bot.reset()
        st.rerun()
    
    st.divider()
    
    # -----------------------------------------------------------------
    # POMDP State Display
    # -----------------------------------------------------------------
    st.header("📊 POMDP State")
    
    if "bot" in st.session_state and use_pomdp:
        belief = st.session_state.bot.get_belief()
        last_action = st.session_state.bot.get_last_action()
        last_cmi = st.session_state.bot.get_last_cmi()
        
        if belief:
            st.subheader("Belief Distribution")
            
            # Display belief as progress bars
            for state, prob in belief.items():
                # Color-code by state
                if state == "EN_DOMINANT":
                    label = f"🇺🇸 {state}"
                elif state == "ZH_DOMINANT":
                    label = f"🇨🇳 {state}"
                else:
                    label = f"🔄 {state}"
                
                st.progress(prob, text=f"{label}: {prob:.1%}")
            
            st.divider()
            
            # Show last action
            st.subheader("Last Action")
            action_icons = {
                "RESPOND_EN": "🇺🇸",
                "RESPOND_ZH": "🇨🇳",
                "MIXED_RESPONSE": "🔄",
                "TRANSLATE_QUERY": "🔁",
                "None": "❓"
            }
            icon = action_icons.get(last_action, "❓")
            st.info(f"{icon} **{last_action}**")
            
            # Show last CMI
            st.metric("User CMI", f"{last_cmi:.1f}", 
                     help="Code-Mixing Index: 0=monolingual, 50=balanced")
        else:
            st.info("Chat with the bot to see POMDP updates!")
    else:
        if not use_pomdp:
            st.warning("POMDP disabled")
        else:
            st.info("Start chatting to see POMDP state")
    
    st.divider()
    
    # Project info
    st.caption("**AA228/CS238 Project**")
    st.caption("POMDP-based adaptive multilingual chatbot")

# =============================================================================
# Initialize Bot
# =============================================================================

if "model_type" not in st.session_state:
    st.session_state.model_type = None

if "use_pomdp" not in st.session_state:
    st.session_state.use_pomdp = True

current_model = "together" if "Llama" in model_choice else "local"

# Reinitialize if model or POMDP setting changed
needs_reinit = (
    st.session_state.model_type != current_model or 
    st.session_state.use_pomdp != use_pomdp
)

if needs_reinit:
    if current_model == "together":
        if not os.getenv("TOGETHER_API_KEY"):
            st.error("⚠️ TOGETHER_API_KEY not found. Add it to .env file.")
            st.stop()
        # Import only when needed
        from chatbot import TogetherChat
        with st.spinner("Connecting to Together API..."):
            st.session_state.bot = TogetherChat(use_pomdp=use_pomdp)
    else:
        try:
            # Import only when needed (may fail if transformers broken)
            from chatbot import LocalChat
            with st.spinner("Loading Qwen 2.5 model (first time takes ~1 min)..."):
                st.session_state.bot = LocalChat(use_pomdp=use_pomdp)
        except Exception as e:
            st.error(f"⚠️ Local model failed to load: {e}\n\nTry using Together API instead.")
            st.stop()
    
    st.session_state.model_type = current_model
    st.session_state.use_pomdp = use_pomdp
    st.session_state.messages = []
    st.rerun()

# =============================================================================
# Main Chat Interface
# =============================================================================

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Show POMDP metadata for assistant messages
        if msg["role"] == "assistant" and "pomdp_action" in msg:
            with st.expander("🔍 POMDP Decision", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Action:** {msg['pomdp_action']}")
                    st.write(f"**User CMI:** {msg.get('user_cmi', 0):.1f}")
                with col2:
                    if "belief" in msg:
                        st.write("**Belief:**")
                        for s, p in msg["belief"].items():
                            st.write(f"  {s}: {p:.1%}")

# Chat input
if prompt := st.chat_input("Type your message... (Try mixing English and 中文!)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.chat(prompt)
            st.write(response)
            
            # Store message with POMDP metadata
            msg_data = {"role": "assistant", "content": response}
            
            if use_pomdp:
                msg_data["pomdp_action"] = st.session_state.bot.get_last_action()
                msg_data["user_cmi"] = st.session_state.bot.get_last_cmi()
                msg_data["belief"] = st.session_state.bot.get_belief()
            
            st.session_state.messages.append(msg_data)
    
    # Rerun to update sidebar
    st.rerun()

# =============================================================================
# Footer
# =============================================================================

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💡 Try: 'Hello!' → 'Hi 你好!' → '今天天气怎么样?'")
with col2:
    st.caption("Watch the POMDP belief update in the sidebar →")
with col3:
    st.caption("The bot adapts its response language to match you!")
