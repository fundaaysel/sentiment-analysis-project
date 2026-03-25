from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# Path to your extracted fine-tuned model
MODEL_PATH = "C:\\Users\\funda\\AIEngineerCourse\\sentiment-analysis\\fine_tuned_model\\fine_tuned_model"  # Update this path!


# Load the model
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Create pipeline with your fine-tuned model
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

print("✅ Fine-tuned model loaded successfully!\n")

#Interactive testing
def analyse_custom_text():
    print("="*50)
    print("Sentiment Analysis - Interactive Mode")
    print("Using fine-tuned model.")
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
