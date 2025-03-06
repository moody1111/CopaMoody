import json
import os
import tensorflow as tf
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer

# Set cache directory
cache_dir = "CACHE_DIRECTORY_PLACEHOLDER"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Load datasets from drivers, and new files
dataset_dir_d = "DATASET_DIR_D_PLACEHOLDER"
chatbot_dataset_path = "CHATBOT_DATASET_PATH_PLACEHOLDER"
data_dir_e = "DATA_DIRECTORY_PLACEHOLDER"
new_data_dir = "NEW_DATA_DIRECTORY_PLACEHOLDER"  # Update this path with the directory of new files

# Function to load text files from a directory and its subdirectories
def load_text_files_from_directory(directory):
    text_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_data.append(f.read())
    return text_data

# Load dataset from D: drive
# if it was in D:\
#d_dataset = load_text_files_from_directory(dataset_dir_d)

# Load dataset from E: drive
# if it was in E:\
#e_dataset = load_text_files_from_directory(data_dir_e)

# Load new dataset from the new files
new_dataset = load_text_files_from_directory(new_data_dir)

# Load chatbot dataset from C: drive
with open(chatbot_dataset_path, 'r', encoding='utf-8') as file:
    chatbot_dataset = json.load(file)

# Transform chatbot dataset to list of text entries
chatbot_texts = []
for key, value in chatbot_dataset.items():
    chatbot_texts.append(f"{key}: {value}")

# Combine all datasets into a single list
combined_dataset = d_dataset + e_dataset + new_dataset + chatbot_texts

# Convert combined dataset to Hugging Face Dataset
data = {"text": combined_dataset}
dataset = Dataset.from_dict(data)

# Split dataset into train, validation, and test sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_val_split = train_test_split['train'].train_test_split(test_size=0.25)

# Create a DatasetDict to hold the splits
datasets = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': train_test_split['test']
})

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize datasets
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4)

# Save tokenized datasets to disk
tokenized_datasets.save_to_disk("CACHE_DIRECTORY_PLACEHOLDER\\tokenized_datasets")

# Model Architecture in TensorFlow
class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            for _ in range(num_layers)
        ]
        self.norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.ffn_layers = [tf.keras.layers.Dense(dim_feedforward, activation='relu') for _ in range(num_layers)]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout_rate) for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(vocab_size)
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        for i in range(len(self.attention_layers)):
            attn_output = self.attention_layers[i](x, x)
            x = self.norm_layers[i](x + attn_output)
            ffn_output = self.ffn_layers[i](x)
            x = self.dropout_layers[i](ffn_output, training=training)
        return self.output_layer(x)

# Initialize model with hyperparameters
vocab_size = tokenizer.vocab_size
d_model = 768
num_heads = 12
num_layers = 12
dim_feedforward = 3072
dropout_rate = 0.1

model = TransformerModel(vocab_size, d_model, num_heads, num_layers, dim_feedforward, dropout_rate)

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Prepare datasets for TensorFlow
def encode_examples(ds, tokenizer):
    input_ids = []
    attention_masks = []
    for example in ds:
        encoded = tokenizer.encode_plus(
            example['text'],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, input_ids))

train_dataset = encode_examples(tokenized_datasets['train'], tokenizer)
val_dataset = encode_examples(tokenized_datasets['validation'], tokenizer)

train_dataset = train_dataset.shuffle(10000).batch(4).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(4).prefetch(tf.data.experimental.AUTOTUNE)

# Training Loop
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Save the trained model
model.save_weights("CACHE_DIRECTORY_PLACEHOLDER\\copamoody_model.h5")
