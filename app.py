from cProfile import label

import streamlit as st
from transformers import pipeline

#Page config
st.set_page_config(page_title="Sentiment Analyser", page_icon=":smiley:", layout="centered")


#Load model (with caching so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')  

classifier = load_model()

#App title and description
st.title("Movie Review Sentiment Analyser")
st.write("Enter a movie review below to analyze its sentiment.")

#Text input for movie review
user_input = st.text_area("Your Review",placeholder="Enter your movie review here..." ,height=150)

#Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if user_input:
        #Make prediction
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]

        #Display results
        st.divider()

        #Sentiment with color
        if result['label'] == 'POSITIVE':
            st.success(f"**Sentiment:** {result['label']}")
        else:
            st.error(f"**Sentiment:** {result['label']}")

        #Confidence
        st.metric(label="Confidence", value=f"{result['score']:.2%}")

        #Progress bar for confidence
        st.progress(result['score'])

    else:
        st.warning("Please enter a valid movie review to analyze.")

#Sidebar with examples
st.sidebar.header("Try these examples:")
st.sidebar.write("Click to copy to clipboard:")

examples = [
    "This movie was fantastic! I loved every minute of it.",
    "The plot was boring and the acting was terrible.", 
    "An absolute masterpiece with stunning visuals and a gripping story.",
    "I wouldn't recommend this movie to anyone. It was a waste of time.",
    "A decent film with some good moments, but overall it was just okay."
]

for example in examples:
    if st.sidebar.button(example[:50] + "...", key=example):
        st.rerun() #This would set the text(simplified here)
        