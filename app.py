import streamlit as st
from Interactive_Storytelling import generate_story  # Import the story generation function

def main():
    st.title("Interactive Storytelling")
    
    # Input for the user to enter their request
    user_input = st.text_input("Enter a story prompt or request:")

    if st.button("Tell me a story"):
        if user_input:
            try:
                # Debug: Print the user input for troubleshooting
                print(f"User input: {user_input}")
                
                # Generate story using the function from Interactive_Storytelling.py
                story = generate_story(user_input)
                
                if story:
                    st.write(story)
                    # Debug: Print the generated story to the console
                    print(f"Generated story: {story}")
                else:
                    st.error("No story was generated. Please try again.")
            except Exception as e:
                st.error(f"Error fetching story: {e}")
                # Debug: Print the error message to the console
                print(f"Error: {e}")
        else:
            st.warning("Please enter a prompt or request.")
            # Debug: Inform user input was missing
            print("Warning: No user input provided.")

if __name__ == "__main__":
    main()