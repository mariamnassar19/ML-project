import streamlit as st

def render():
    st.header("Feedback")
    feedback_text = st.text_area("Please provide your feedback here:")
    if st.button("Submit Feedback"):
        if feedback_text:
            st.success("Thank you for your feedback!")
        else:
            st.error("Please enter your feedback before submitting.")
