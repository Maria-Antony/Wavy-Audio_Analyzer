import os
import streamlit as st
from mistralai import Mistral, UserMessage

from dotenv import load_dotenv
load_dotenv()

st.subheader("💬 Ask Anything About This Signal")

if "signal_stats" in st.session_state:
    stats = st.session_state["signal_stats"]
    user_question = st.text_input("Your question:", placeholder="e.g., Is this speech or noise?")
    if st.button("Ask"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key and user_question.strip():
            client = Mistral(api_key=api_key)
            prompt = f"Given: {stats}\\nQuestion: {user_question}"
            messages = [UserMessage(content=prompt)]
            response = client.chat.complete(model="mistral-large-latest", messages=messages)
            st.success(response.choices[0].message.content)
        else:
            st.warning("Missing API key or question.")
