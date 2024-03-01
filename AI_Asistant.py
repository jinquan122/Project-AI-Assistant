import streamlit as st
from app.controllers.llamaindex.query_pipeline.nlp_qp import init_agent


def main():
    st.set_page_config(page_title="Natural Language Processing Expert Agent, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Natural Language Processing Expert Agent, powered by LlamaIndex 💬🦙")
    # st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
    with st.sidebar:
        st.markdown("LlamaIndex")
        # if st.button("🔗Refresh Knowledge", type="primary"):
        #     st.rerun()
        st.info("This apps is built with LLamaIndex, the RAG framework for LLM knowledge augmentation.\nEnjoy the DEMO!")

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm a Natural Language Expert. You're welcomed to ask me NLP related questions.💬"}
        ]

    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            agent = init_agent()
            return agent

    agent = load_data()

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chat_engine = agent

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # create streaming response from a response where response is a text
                response = st.session_state.chat_engine.query(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

if __name__ == "__main__":
    main()