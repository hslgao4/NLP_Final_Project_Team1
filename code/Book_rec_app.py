# To Run: streamlit run Book_rec_app.py --server.port=8888
import streamlit as st

#%%
st.set_page_config(
    page_title="AI Book Recommendation",
    page_icon=":books:",
)
#%%
st.title('AI Book Recommendation App')
with st.expander(":open_book: Welcome to AI Book Recommendation!", expanded=False):
    st.write(
        """     
    - Provide a short book review and receive tailored recommendations for your next read.
    - Use the intuitive controls on the left to tailor your experience to your preferences.
        """
    )

# Step-by-Step Guide
with st.expander(":open_book: How to Use", expanded=False):
    st.write(
        """
    1. **Enter Book Review:**
        - Type or paste a 3-5 sentence review of a book you enjoyed to find more like it.
        - Hint: Tell us what you liked most about a recent read.
        - Be descriptive, get creative! :stuck_out_tongue_winking_eye:

    2. **Choose Features:**
        - Toggle the switch on the left sidebar to choose between SVD and Transformers4Rec models.
        - Which model gives you better results?
        """
    )

#%%
# Main function to run the Streamlit app
def main():
    st.title('Your next story starts here...')

    # Sidebar to select model type
    model_type = st.sidebar.radio("Select Model Type", ("SVD", "Transformers4Rec"))

    # Text input area for book review
    input_text = st.text_area("Enter your book review (3-5 sentences):")

    if st.button("Get Recommendations"):
        if model_type == "SVD":
            #recommendations = svd_main(input_text)
            st.write("SVD recommendations coming soon!")
            #st.write("**Recommendations:**")
            #st.table(recommendations)
        elif model_type == "Transformers4Rec":
            # recommendations = get_transformer4rec_recommendations(input_text, data)
            st.write("Transformers4Rec recommendations coming soon!")


        # Output recommendations
        # st.write("**Recommendations:**")
        # st.table(recommendations)


if __name__ == "__main__":
    main()