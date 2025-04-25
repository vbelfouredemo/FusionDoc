# FusionDoc

FusionDoc is a RAG (Retrieval-Augmented Generation) solution that can analyze integration JSON files and generate human-readable documentation. It helps understand what an integration does by explaining the source systems, data transformations, and target systems in simple terms.

## Architecture

This solution consists of three main components, all running in Docker containers:

1. **Ollama with Mistral**: The LLM (Large Language Model) that powers the natural language understanding and generation.
2. **ChromaDB**: The vector database used to store and query embeddings.
3. **LangChain API**: A FastAPI application that orchestrates the document processing, vector storage, and generation of documentation.

## Features

- **Document Generation**: Upload JSON integration files and generate human-readable documentation in various formats (HTML, Markdown, PDF)
- **Smart Document Generation**: Use RAG to generate more context-aware documentation based on training examples
- **Model Training**: Upload examples of integration JSON with corresponding documentation to train the model
- **Web Interface**: Easy-to-use interface for both document generation and model training

## Getting Started

### Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of RAM available for Docker

### Running the Solution

1. Clone the repository
2. Navigate to the project directory
3. Start the Docker containers:

```
docker-compose up -d
```

On first run, it will download the necessary Docker images and build the LangChain container. This may take some time depending on your internet connection.

### Using the Web Interface

Once the containers are running, you can access the web interface at:
```
http://localhost:5000
```

The interface has two main tabs:

1. **Generate Documentation**: 
   - Upload a JSON file by dragging and dropping or using the file browser
   - Select output format (HTML, Markdown, PDF)
   - Choose between standard or smart documentation generation
   - View generated documentation and download if needed

2. **Train the Model**:
   - Add examples of integration JSON with expected documentation
   - Upload JSON files by drag and drop or paste directly in the text area
   - Submit training data to improve smart documentation generation
   - View training status and statistics

### Using the API

The solution also exposes a REST API with the following endpoints:

1. **Upload a JSON file**:
   ```
   POST http://localhost:5000/upload
   ```
   This endpoint accepts a file upload in JSON format.

2. **Analyze a JSON file with format selection**:
   ```
   POST http://localhost:5000/analyze-with-format
   ```
   Provide the filename of a previously uploaded JSON file and desired output format using form data.

3. **Smart Analysis**:
   ```
   POST http://localhost:5000/analyze-smart
   ```
   Uses the trained vector store to generate enhanced documentation.

4. **Training endpoints**:
   ```
   POST http://localhost:5000/train
   ```
   Submit training examples to improve smart document generation.
   ```
   GET http://localhost:5000/training/status
   ```
   Check the status of training data.

5. **Check API status**:
   ```
   GET http://localhost:5000/
   ```
   Returns a status message confirming the API is running.

## Example API Usage

1. Upload an integration JSON file:
   ```bash
   curl -X POST -F "file=@path/to/your/integration.json" http://localhost:5000/upload
   ```

2. Analyze the uploaded file with format selection:
   ```bash
   curl -X POST -F "filename=integration.json" -F "output_format=markdown" http://localhost:5000/analyze-with-format
   ```

3. Submit training data:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"examples":[{"integration_json":{...},"documentation":"..."}]}' http://localhost:5000/train
   ```

## Customization

You can customize the solution by:

1. Modifying the prompt in `app.py` to generate different types of documentation.
2. Changing the Ollama model by updating the model name in `app.py`.
3. Adjusting the ChromaDB configuration for different embedding characteristics.
4. Adding more output formats or visualization options.

## Troubleshooting

- If you encounter issues with Ollama, you may need to pull the Mistral model manually:
  ```bash
  docker exec -it ollama ollama pull mistral
  ```

- If the LangChain container exits, check the logs:
  ```bash
  docker logs langchain
  ```

- If the web interface doesn't load properly, check if all containers are running:
  ```bash
  docker-compose ps
  ```

## Development

To contribute to this project:

1. Fork the repository
2. Make your changes
3. Submit a pull request

Please use the provided `.gitignore` file to avoid committing unnecessary files to the repository.