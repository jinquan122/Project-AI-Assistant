import streamlit as st

def main():
    if 'response' in st.session_state:
        response = st.session_state['response']
        st.title("Retrieved Nodes")
        st.write("Please check the retrieved node:")
        for index, node in enumerate(response.source_nodes):
            with st.container(border=True):
                st.subheader(f"Node {index}")
                st.write(f"Similarity score: {node.score}")
                st.write(f"Text: {node.text}")
                st.divider()
    else:
        st.write("No valid response was found!")

if __name__ == "__main__":
    main()
