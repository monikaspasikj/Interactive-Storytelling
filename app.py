import streamlit as st
from Interactive_Storytelling import generate_story  # Import the story generation function

def main():
    st.title("Interactive Storytelling")
    
    # Input for the user to enter their request
    user_input = st.text_input("Enter a story prompt or request:")

    if st.button("Tell me a story"):
        if user_input:
            try:
                # Generate story using the function from Interactive_Storytelling.py
                story = generate_story(user_input)
                st.write(story)
                # Debug: Print the generated story to console
                print(f"Generated story: {story}")
            except Exception as e:
                st.error(f"Error fetching story: {e}")
        else:
            st.warning("Please enter a prompt or request.")

if __name__ == "__main__":
    main()