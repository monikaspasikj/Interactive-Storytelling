import streamlit as st
from Interactive_Storytelling import generate_story, generate_story_direct

def main():
    st.title("Interactive Storytelling")
    
    # Input for the user to enter their request
    user_input = st.text_input("Enter a story prompt or request:")

    if st.button("Tell me a story"):
        if user_input:
            try:
                # Generate story using the function from Interactive_Storytelling.py
                story = generate_story(user_input)

                # If the agent didn't respond, try direct story generation
                if "No response from the storytelling agent" in story:
                    story = generate_story_direct(user_input)

                st.write(story)
                print(f"Generated story: {story}")
            except Exception as e:
                st.error(f"Error fetching story: {e}")
        else:
            st.warning("Please enter a prompt or request.")

if __name__ == "__main__":
    main()
