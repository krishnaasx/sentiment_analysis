import argparse
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from textwrap import wrap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 10
class_names = ['negative', 'neutral', 'positive']

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Sentiment Classifier model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# Custom Dataset class
class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# DataLoader creation function
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)  

# Training function
def train_model(model, train_loader, val_loader, epochs, device):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss, correct_preds, total_examples = 0, 0, 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, dim=1)
            total_loss += loss.item()
            correct_preds += torch.sum(preds == targets).item()
            total_examples += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_preds / total_examples
        print(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["targets"].to(device)
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, targets)
                    _, preds = torch.max(outputs, dim=1)
                    val_loss += loss.item()
                    val_correct += torch.sum(preds == targets).item()
                    val_total += targets.size(0)
            val_avg_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    torch.save(model.state_dict(), "best_model_state.bin")
    print("Model trained and saved as 'best_model_state.bin'")

# Prediction function
def get_predictions(model, data_loader):
    model.eval()
    review_texts, predictions, prediction_probs, real_values = [], [], [], []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    return (
        review_texts,
        torch.stack(predictions).cpu(),
        torch.stack(prediction_probs).cpu(),
        torch.stack(real_values).cpu()
    )

# Confusion matrix visualization
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.show()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis with BERT")
    parser.add_argument('--train', action='store_true',
                        help="Train the model from scratch instead of using pre-trained weights")
    args = parser.parse_args()

    # Initialize the model
    model = SentimentClassifier(len(class_names))

    # Model path
    model_path = "best_model_state.bin"

    if args.train:
        csv_file = "reviews.csv"
        if not os.path.exists(csv_file):
            print(f"Error: '{csv_file}' not found! Please provide a CSV with 'content' and 'score' columns.")
            exit(1)
        
        df = pd.read_csv(csv_file)
        df['sentiment'] = df['score'].apply(lambda x: 0 if int(x) <= 2 else (1 if int(x) == 3 else 2))
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
        val_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
        
        print("Starting training...")
        train_model(model, train_loader, val_loader, EPOCHS, device)
    else:
        if not os.path.exists(model_path):
            print(f"Error: '{model_path}' not found! Use --train to train first.")
            exit(1)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            print(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    # # # Evaluate on test data if available
    # csv_file = "reviews.csv"
    # if os.path.exists(csv_file):
    #     df = pd.read_csv(csv_file)
    #     df['sentiment'] = df['score'].apply(lambda x: 0 if int(x) <= 2 else (1 if int(x) == 3 else 2))
    #     test_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)
    #     y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_loader)
        
    #     print("\nClassification Report:")
    #     print(classification_report(y_test, y_pred, target_names=class_names))
    #     cm = confusion_matrix(y_test, y_pred)
    #     df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    #     show_confusion_matrix(df_cm)
        
    #     idx = 6
    #     review_text = y_review_texts[idx]
    #     true_sentiment = y_test[idx]
    #     pred_df = pd.DataFrame({'class_names': class_names, 'values': y_pred_probs[idx].cpu().numpy()})
    #     print("\nExample Review:")
    #     print("\n".join(wrap(review_text)))
    #     print(f'True sentiment: {class_names[true_sentiment]}')
    #     sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
    #     plt.ylabel('sentiment')
    #     plt.xlabel('probability')
    #     plt.xlim([0, 1])
    #     plt.show()

    # Predict on raw text
    review_text = "I am satisfieid with the result!!"   
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
    print(f'\nReview text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')