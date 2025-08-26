"""
Financial Sentiment Analysis - Quick Demo
Run this to see the model in action!
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import sys

def load_model():
    """Load the pre-trained FinBERT model"""
    try:
        model_path = "peejm/finbert-financial-sentiment"
        print("Loading FinBERT model from Hugging Face Hub...")
        print("Note: The first run will download the model (~440MB), which may take a few minutes.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the model 'peejm/finbert-financial-sentiment' is public on the Hugging Face Hub.")
        sys.exit(1)

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a single text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = int(torch.argmax(predictions, dim=-1).item())
    
    labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    confidence = predictions[0][predicted_class].item()
    
    return labels[predicted_class], confidence

def interactive_mode(model, tokenizer):
    """Interactive demo mode"""
    print("\nInteractive Demo Mode")
    print("Enter financial headlines to analyze (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            text = input("\nEnter headline: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                print("Thanks for trying the demo!")
                break
            if not text:
                continue
                
            sentiment, confidence = predict_sentiment(text, model, tokenizer)
            
            # Color-coded output
            if sentiment == 'positive':
                color = "\033[92m"  # Green
            elif sentiment == 'negative':
                color = "\033[91m"  # Red
            else:
                color = "\033[93m"  # Yellow
                
            print(f"Sentiment: {color}{sentiment.upper()}\033[0m (confidence: {confidence:.1%})")
            
        except KeyboardInterrupt:
            print("\nDemo ended!")
            break
        except Exception as e:
            print(f"Error: {e}")

def demo_examples(model, tokenizer):
    """Run predefined demo examples"""
    examples = [
        "Apple stock surges 15% after strong earnings report",
        "Market crashes as inflation fears mount",
        "Fed announces unchanged interest rates",
        "Tesla beats delivery expectations in Q3",
        "Oil prices plummet on demand concerns"
    ]
    
    print("\nDemo Examples")
    print("=" * 50)
    
    for i, text in enumerate(examples, 1):
        sentiment, confidence = predict_sentiment(text, model, tokenizer)
        print(f"{i}. {text}")
        print(f"   {sentiment.upper()} (confidence: {confidence:.1%})")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Financial Sentiment Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_demo.py                    # Run interactive demo
  python quick_demo.py --examples         # Show predefined examples
  python quick_demo.py --text "Apple stock rises"  # Analyze single text
        """
    )
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--examples", action="store_true", help="Run predefined examples")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (default)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model()
    
    # Run appropriate mode
    if args.text:
        sentiment, confidence = predict_sentiment(args.text, model, tokenizer)
        print(f"\nText: {args.text}")
        print(f"Sentiment: {sentiment.upper()} (confidence: {confidence:.1%})")
    elif args.examples:
        demo_examples(model, tokenizer)
    else:
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()