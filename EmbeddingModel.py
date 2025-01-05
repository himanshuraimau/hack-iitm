import os
import tensorflow as tf
import warnings
from sentence_transformers import SentenceTransformer
import singlestoredb as s2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning messages

# Suppress specific TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to create a new connection
def create_connection():
    return s2.connect('')

# Load the BGE-M3 model using Sentence Transformers
model = SentenceTransformer('BAAI/bge-m3')

# Sample sentences for which we want to generate embeddings
sentences = [
    "Spicy food is unhealthy",
    "Good food is love"
]

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Insert embeddings into SingleStoreDB
try:
    with create_connection() as conn:  # Create a new connection for this block
        with conn.cursor() as cur:
            # Create a table to store embeddings if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    content TEXT,
                    embedding BLOB
                )
            """)

            # Insert each sentence and its corresponding embedding into the table
            for sentence, embedding in zip(sentences, embeddings):
                cur.execute("""
                    INSERT INTO embeddings (content, embedding) VALUES (%s, %s)
                """, (sentence, embedding.tobytes()))  # Convert numpy array to bytes

            # Commit the transaction
            conn.commit()
            print("Insertion successful.")
except Exception as e:
    print(f"Insertion failed: {e}")

# Performing a similarity search can be done here (not shown in this example)
