import streamlit as st
import agent

st.set_page_config(page_title="Jarvis", page_icon="🤖", layout="centered")
def check_password():
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Password", type="password")
    if pw:
        if pw == st.secrets["APP_PASSWORD"]:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False


if not check_password():
    st.stop()
st.title("🤖 Jarvis")
st.caption("Your personal AI assistant")

if "history" not in st.session_state:
    st.session_state.history = agent.load_history()

with st.sidebar:
    st.header("Dashboard")

    st.subheader("This week")
    st.text(agent.habit_summary(7))

    st.subheader("Facts about you")
    st.text(agent.read_facts() or "No facts saved yet.")
    new_fact = st.text_input("Add a fact")
    if st.button("Save fact") and new_fact.strip():
        agent.add_fact(new_fact.strip())
        st.rerun()

    st.divider()
    if st.button("Clear chat"):
        st.session_state.history = []
        agent.save_history([])
        st.rerun()

for m in st.session_state.history:
    role = "assistant" if m["role"] == "model" else "user"
    with st.chat_message(role):
        st.markdown(m["text"])

if prompt := st.chat_input("Message Jarvis..."):
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = agent.ask(st.session_state.history)
            except Exception as e:
                st.session_state.history.pop()
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.markdown(reply)
    st.session_state.history.append({"role": "model", "text": reply})
    agent.save_history(st.session_state.history)
    st.rerun()