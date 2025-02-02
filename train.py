import torch
import torch.nn as nn
import torch.nn.functional as F
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import numpy as np
import os
os.environ["WANDB_DISABLED"] = "true"

# Multi-Head Classifier
class MultiHeadClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes_intent, num_classes_domain, num_classes_hitl):
        super().__init__()
        self.intent_head = nn.Sequential(nn.Linear(embedding_size, 50),nn.ReLU(),nn.Linear(50, num_classes_intent))
        self.domain_head = nn.Sequential(nn.Linear(embedding_size, 50),nn.ReLU(),nn.Linear(50, num_classes_domain))
        self.hitl_head = nn.Sequential(nn.Linear(embedding_size, 50),nn.ReLU(),nn.Linear(50, num_classes_hitl))

    def forward(self, features):
        x = features["sentence_embedding"]
        logits_intent = self.intent_head(x)
        logits_domain = self.domain_head(x)
        logits_hitl = self.hitl_head(x)
        return {"logits": (logits_intent, logits_domain, logits_hitl)}

    def predict(self, embeddings):
        logits_intent, logits_domain, logits_hitl = self.forward({"sentence_embedding": embeddings})["logits"]
        return (
            torch.argmax(logits_intent, dim=-1),
            torch.argmax(logits_domain, dim=-1),
            torch.argmax(logits_hitl, dim=-1),
        )

    def predict_proba(self, embeddings):
        logits_intent, logits_domain, logits_hitl = self.forward({"sentence_embedding": embeddings})["logits"]
        return (
            F.softmax(logits_intent, dim=-1),
            F.softmax(logits_domain, dim=-1),
            F.softmax(logits_hitl, dim=-1),
        )

    def get_loss_fn(self):
        # Custom loss for multi-head classification
        def loss_fn(logits, labels):
            # Unpack labels into intent, domain, and hitl
            intent_labels = labels[:, 0].long()
            domain_labels = labels[:, 1].long()
            hitl_labels = labels[:, 2].long()

            intent_logits, domain_logits, hitl_logits = logits

            # Calculate loss for each head
            loss_intent = F.cross_entropy(intent_logits, intent_labels)
            loss_domain = F.cross_entropy(domain_logits, domain_labels)
            loss_hitl = F.cross_entropy(hitl_logits, hitl_labels)

            return loss_intent + loss_domain + loss_hitl  # Total loss from all heads

        return loss_fn
        
# Initialize Model
embedding_size = 384  # BAAI/bge-small-en-v1.5 embedding size
num_classes_intent = 4
num_classes_domain = 2
num_classes_hitl = 2

model_body = SentenceTransformer("BAAI/bge-small-en-v1.5")
model_head = MultiHeadClassifier(embedding_size, num_classes_intent, num_classes_domain, num_classes_hitl)
model = SetFitModel(model_body, model_head)

intent_map = {0:"Booking/cancellation",1:"Support",2:"Payment",3:"Others"}
domain_map = {0:"domain",1:"out of domain"}
hitl_map = {0:False,1:True}

data = {
    "text": [
        "Book a flight to New York", "Cancel my flight.", "I think i need to cancel the hotel booking.",
        "Find me the best hotels","cheaper flights?","How can i reach Varanasi?","I am unable to book",
        "I can't understand it","how to make payment", "do you accept debit cards","what all payment methods do you accept?",
        "I want my money back","i cannot pay","it doesn't work","I need help",
        "need a refund","I haven't received my refund", "Legal consultation service", "AI in robotics",
        "Turn on the lights", "Play some music", "financial analysis"
    ],
    "intent": [0,0,0,0,0,0,0,1,1,2,2,2,2,1,1,2,2,3,3,3,3,3],
    "domain": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    "hitl":   [0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,1,0,0,0,0,0],
}

# Create a composite label (tuple of intent, domain, hitl)
composite_labels = [[intent, domain, hitl] for intent, domain, hitl in zip(data["intent"], data["domain"], data["hitl"])]


dataset = Dataset.from_dict({
    "text": data["text"],
    "label": composite_labels
})

dataset = dataset.shuffle()


args = TrainingArguments(batch_size=4, num_epochs=(2,10))


trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=dataset
)
trainer.train()

model.save_pretrained("query_preprocessing")
