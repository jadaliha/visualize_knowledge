import uvicorn
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from sklearn.decomposition import PCA
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duckdb import connect
from langchain_community.vectorstores import DuckDB
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer 
from langchain_text_splitters import MarkdownTextSplitter
from tqdm import tqdm
from langchain_core.embeddings import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer


class GTEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'Supabase/gte-small'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> (List[List[float]]):
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> (List[float]):
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.model.encode(text).tolist()
    
embeddings = GTEmbeddings()

CHUNK_SIZE = 1024
html2text = Html2TextTransformer()
spliter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE)
# use local file for memory store
conn = connect(".data/blogs.db")


class DuckStore(DuckDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ensure_table()
        self.sources = self._load_sources_from_db()

    def _ensure_table(self):
        super()._ensure_table()
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                url VARCHAR UNIQUE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE
            )
        """)  # {{ edit_3 }}
        self._sources = self._connection.table("resources")

    def _load_sources_from_db(self):
        # Use fetchall instead of fetchnumpy to handle nulls properly
        result = self._connection.execute("SELECT url FROM resources").fetchall()  # {{ edit_1 }}
        # Extract URLs, ensuring no None values are included
        return set(row[0] for row in result if row[0] is not None)  # {{ Reason: This comprehension ensures that only non-None URLs are added to the set, preventing the inclusion of unhashable types. }}

    def add_source(self, urls: str | set[str]):
        if isinstance(urls, str):
            urls = {urls}
        
        new_urls = urls - self.sources
        
        for url in new_urls:
            self.sources.add(url)
            self._connection.execute("INSERT OR IGNORE INTO resources (url) VALUES (?)", [url])

    def remove_source(self, url):
        if url in self.sources:
            self.sources.remove(url)
            self._connection.execute("DELETE FROM resources WHERE url = ?", [url])

    def get_sources(self):
        return list(self.sources)

vectorstore = DuckStore(connection=conn, embedding=embeddings)

def consume(urls: set[str], vectorstore: DuckStore, browser=AsyncChromiumLoader):
    urls = urls - vectorstore.sources
    
    if not urls:
        print("No new URLs to consume.")
        return
    
    loader = browser(urls)
    print("Loading URLs...")
    index = {}
    
    for document in tqdm(loader.lazy_load(), total=len(urls), desc="Loading URLs", unit="url"):
        if not document.page_content.startswith("Error"):
                       
            # Transform HTML to text
            doc_transformed = html2text.transform_documents([document])
            
            # Split text into chunks
            chunks = spliter.split_documents(doc_transformed)
            
            # Update metadata and index for each chunk
            current_index = 0
            for chunk in chunks:
                chunk.metadata.update({
                    "app": "consume",
                    "index": current_index
                })
                current_index += 1
            
            # Add chunks to vectorstore
            vectorstore.add_documents(chunks)
            
            # Update index
            index[document.metadata["source"]] = current_index
            
            # Add source to vectorstore
            vectorstore.add_source(document.metadata["source"])
    
    print("Consumption complete.")
    return index

# convert a duckdb table to a

pca = PCA(n_components=3)
app = FastAPI()
# Removing HTTPS redirect for local development
# app.add_middleware(HTTPSRedirectMiddleware)

# Fetch data from DuckDB
data = vectorstore._table.select('id', 'text', 'embedding', 'metadata').to_df()

# use tensorboard to visualize the embeddings
# https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin
# tensorboard --logdir logs

# use PCA to reduce the dimensionality of the embeddings


# Convert list embeddings to numpy array

embeddings_array = np.array(data['embedding'].tolist())

# Apply PCA
reduced_embeddings = pca.fit_transform(embeddings_array)

# Assign the reduced embeddings back to the dataframe
data['embedding'] = reduced_embeddings.tolist()
data['x'] = data['embedding'].apply(lambda x: x[0])
data['y'] = data['embedding'].apply(lambda x: x[1])
data['z'] = data['embedding'].apply(lambda x: x[2])

data['source'] = data['metadata'].apply(lambda x: json.loads(x)['source'].split('/')[2])
data['text'] = data['text'].apply(lambda x: x[:100])
data = data[['id','text', 'x', 'y', 'z', 'source']].to_dict(orient='records')

# Convert data to JSON
json_data = json.dumps(data)

def fetch_address_from_database(id):
    # Find the data with the id in a duckdb query
    meta = vectorstore._table.filter(f"id == '{id}'").select('metadata').fetchone()[0]
    json_meta = json.loads(meta)
    address  = json_meta['source']
    chuncks = vectorstore._table.filter(f"metadata LIKE '%{address}%'").select('id').fetchall()

    address = {
        'address': json_meta['source'],
        'chunk_ids': [item for sublist in chuncks for item in sublist]
    }
    return address

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/source-address/{id}")
async def get_source_address(id: str):
    print(id)
    # Fetch the full address based on the id
    address = fetch_address_from_database(id)
    if address:
        return address
    return {"error": "Address not found"}

@app.get("/")
async def visualize(request: Request):
    return templates.TemplateResponse("visualize.html", {"request": request})

@app.get("/data")
async def get_data():
    return data

@app.post("/prompt")
async def process_prompt(prompt: str = Form()):
    # Embed the prompt using the same model
    prompt_embedding = embeddings.embed_query(prompt)
    
    # Convert to numpy array for PCA transformation
    prompt_embedding_array = np.array([prompt_embedding])
    
    # Transform the prompt embedding using the same PCA model
    # that was used for the original data
    prompt_reduced = pca.transform(prompt_embedding_array)[0]
    
    # Calculate distances to find relevant chunks
    distances = []
    for i, point in enumerate(data):
        # Calculate Euclidean distance between prompt and each point
        point_coords = np.array([point['x'], point['y'], point['z']])
        prompt_coords = np.array([prompt_reduced[0], prompt_reduced[1], prompt_reduced[2]])
        distance = np.linalg.norm(point_coords - prompt_coords)
        distances.append((i, distance))
    
    # Sort by distance and get the closest points (most relevant)
    distances.sort(key=lambda x: x[1])
    relevant_indices = [d[0] for d in distances[:10]]  # Get top 10 relevant chunks
    
    # Prepare response with prompt position and relevant chunks
    response = {
        'prompt': {
            'text': prompt,
            'x': float(prompt_reduced[0]),
            'y': float(prompt_reduced[1]),
            'z': float(prompt_reduced[2])
        },
        'relevant_chunks': [data[i]['id'] for i in relevant_indices],
        'relevance_radius': float(distances[9][1]) if len(distances) >= 10 else 1.0  # Use distance to 10th point as radius
    }
    
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=3030,
    )

# use tensorboard to visualize the embeddings
# https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin
# tensorboard --logdir logs
