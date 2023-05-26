import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLSTM(nn.Module):
    def __init__(self, sequence_len, vocab_size, embed_size, num_layers, hidden_dim, batch_size):
        super(CharLSTM, self).__init__()
        self.sequence_len = sequence_len
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):
        embeds = self.embedding(x)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embeds)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[:, -1, :]
        logits = self.fc(output)
        return logits


def preprocess_domain(domain):
    # Preprocess the domain by converting characters to ASCII codes
    domain = domain.lower()
    ascii_codes = [ord(char) for char in domain]
    return ascii_codes


def predict_domain(domain):
    # Preprocess the domain
    processed_domain = preprocess_domain(domain)

    # Convert the domain to tensor and calculate its length
    domain_tensor = torch.tensor(processed_domain, dtype=torch.long).unsqueeze(0)
    domain_length = torch.tensor([len(processed_domain)], dtype=torch.int64)

    # Load the trained model
    model = CharLSTM(sequence_len=128, vocab_size=257, embed_size=256, num_layers=2, hidden_dim=512, batch_size=1)
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model.pth")
    # Map the keys from the saved state dictionary to the model's state dictionary
    mapped_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(mapped_dict)
    model.load_state_dict(model_dict)
    model.eval()

    # Make the prediction
    with torch.no_grad():
        output = model(domain_tensor, domain_length)
        probabilities = F.softmax(output, dim=1)
        _, predicted_label = torch.max(probabilities, dim=1)
        return predicted_label[0].item()  # Extract the predicted label for the first sample in the batch


# Example usage
domain = "example.com"
prediction = predict_domain(domain)
print("Domain:", domain)
print("Prediction:", "Real" if prediction == 1 else "Fake")
