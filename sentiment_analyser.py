from transformers import pipeline


#Load a pre-trained sentiment analysis model
print("Loading sentiment analysis model...")
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english' )

print("Model loaded successfully.\n")

#Interactive testing

def analyse_custom_text():
    print("="*50)
    print("Sentiment Analysis - Interactive Mode")
    print("="*50)
    print("Type you rmovie reviews below.")
    print("Type 'quit' to exit the program.\n")

    while True:
        user_input = input("Enter your movie review: ").strip()

        if user_input.lower() == 'quit':
            print("\nThanks for using the Sentiment Analyser. Goodbye!")
            break
        if user_input:
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']

            print(f"Sentiment: {label} \n Confidence: {score:.2%}\n")
        else:
            print("Please enter a valid movie review.\n")


#Run the interactive sentiment analysis
analyse_custom_text()
