import streamlit as st
from Interactive_Storytelling import storytelling_agent, child_user_agent

def main():
    st.title("Interactive Storytelling")
    
    # Input for the user to enter their request
    user_input = st.text_input("Enter a story prompt or request:")
    
    if st.button("Tell me a story"):
        if user_input:
            try:
                # Use LangChain to fetch the story based on user input
                response = storytelling_agent.request_story(user_input)
                st.write(response)
            except Exception as e:
                st.error(f"Error fetching story: {e}")
        else:
            st.warning("Please enter a prompt or request.")

if __name__ == "__main__":
    main()
