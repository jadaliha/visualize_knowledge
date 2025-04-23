# Knowledge Visualization

A 3D interactive visualization tool for exploring text embeddings from blog posts. This application demonstrates how information retrieval works by visualizing text chunks in three-dimensional space and allowing users to query the knowledge base.

## Overview

This project creates a web application that:

1. Loads blog post content from a DuckDB database
2. Converts text chunks into embeddings using sentence transformers
3. Reduces dimensionality to 3D using PCA
4. Visualizes the text chunks as interactive points in 3D space
5. Allows users to query the knowledge base with natural language prompts
6. Highlights relevant information based on semantic similarity

The visualization helps demonstrate how AI-based information retrieval systems work by showing how semantically similar content clusters together in the embedding space.

## Features

- **3D Interactive Visualization**: Navigate through the embedding space using mouse controls (rotate, pan, zoom)
- **Color-Coded Sources**: Different colors represent different blog sources
- **Tooltip Information**: Hover over points to see text previews and source information
- **Natural Language Queries**: Ask questions about the blog content
- **Relevance Visualization**: See a blinking point for your query and a radius sphere showing relevant information
- **Source Viewing**: Shift+Click on points to view the original source in an iframe

## How It Works

1. **Text Processing**: Blog posts are split into chunks and converted to vector embeddings
2. **Dimensionality Reduction**: PCA reduces high-dimensional embeddings to 3D for visualization
3. **Interactive Visualization**: Three.js renders the points in 3D space
4. **Query Processing**: 
   - User enters a question
   - The question is converted to an embedding
   - The embedding is transformed using the same PCA model
   - Euclidean distances are calculated to find the most relevant chunks
   - Relevant chunks are highlighted and a sphere shows the relevance radius

## Setup and Installation

### Prerequisites

- Python 3.8+
- DuckDB with blog data in `.data/blogs.db`

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd visualize_knowledge
   ```

2. Install dependencies:
   ```
   # On Windows
   setup.bat
   
   # On Linux/Mac
   bash setup.bash
   
   # Or manually with pip
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
   
2. Open your browser and navigate to:
   ```
   http://localhost:3030
   ```

3. Interact with the visualization:
   - Use mouse to rotate, pan, and zoom
   - Hover over points to see text previews
   - Hold Shift and click on points to view source content
   - Enter questions in the prompt box to find relevant information

## Technologies Used

- **Backend**:
  - FastAPI: Web framework
  - DuckDB: Database for storing blog content
  - Langchain: Document processing and transformations
  - Sentence Transformers: Text embedding generation
  - Scikit-learn: PCA for dimensionality reduction

- **Frontend**:
  - Three.js: 3D visualization
  - HTML/CSS/JavaScript: User interface

## Project Structure

- `app.py`: Main application file with FastAPI routes and embedding logic
- `templates/visualize.html`: HTML template with Three.js visualization
- `static/`: Static files directory
- `.data/blogs.db`: DuckDB database containing blog posts
- `requirements.txt`: Python dependencies
- `setup.bash`/`setup.bat`: Setup scripts for different platforms

## Demo

To see the visualization in action:

1. Start the application
2. Explore the 3D space by rotating and zooming
3. Enter a question in the prompt box (e.g., "What is machine learning?")
4. Observe how the system highlights relevant information
5. Click on highlighted points to view the source content

