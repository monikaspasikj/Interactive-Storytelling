import streamlit as st
from Interactive_Storytelling import storytelling_agent, child_user_agent

def main():
    st.title("Interactive Storytelling")
    
    # Input for the user to enter their request
    user_input = st.text_input("Enter a story prompt or request:")

    if st.button("Tell me a story"):
        # Send the request to the child_user_agent
        response = child_user_agent.request_story(user_input)
        # Display the story from the storytelling_agent
        st.write(response)

if __name__ == "__main__":
    main()
