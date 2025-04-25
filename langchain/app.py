from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import json
import uvicorn
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import os
import tempfile
import time
import requests
import logging
import markdown
import pdfkit
from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionDoc")

class OutputFormat(str, Enum):
    html = "html"
    markdown = "markdown"
    pdf = "pdf"

class TrainingExample(BaseModel):
    integration_json: Dict
    documentation: str

class TrainingData(BaseModel):
    examples: List[TrainingExample]
    description: Optional[str] = None

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
os.makedirs("/data", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        # Save the uploaded JSON file
        file_path = f"/data/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process the JSON file
        data = json.loads(content)
        return {"message": "File processed successfully", "filename": file.filename}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON file"}
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post("/analyze")
async def analyze_file(filename: str = Form(...)):
    try:
        logger.info(f"Starting analysis of file: {filename}")
        # Load the JSON file
        file_path = f"/data/{filename}"
        logger.info(f"Loading file from path: {file_path}")
        
        # Check if Ollama service is available
        try:
            logger.info("Checking if Ollama service is available...")
            # Ollama doesn't have a /api/health endpoint, use the root endpoint instead
            response = requests.get("http://ollama:11434", timeout=5)
            logger.info(f"Ollama connection check response: {response.status_code}")
            if response.status_code >= 400:
                logger.error(f"Ollama service returned status code: {response.status_code}")
                return JSONResponse(
                    status_code=503,
                    content={"error": "Ollama service is not responding correctly. Please check if the service is running properly."}
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama service: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Cannot connect to Ollama service: {str(e)}. Please ensure the Ollama container is running and that the Mistral model is installed."}
            )
        
        # Check if Mistral model is available
        try:
            logger.info("Checking if Mistral model is available...")
            response = requests.post(
                "http://ollama:11434/api/generate",
                json={"model": "mistral", "prompt": "test", "stream": False},
                timeout=5
            )
            logger.info(f"Mistral model check response: {response.status_code}")
            if response.status_code == 404:
                logger.error("Mistral model not found")
                return JSONResponse(
                    status_code=503,
                    content={"error": "The Mistral model is not installed in Ollama. Please install it with 'docker exec -it ollama ollama pull mistral'"}
                )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not check Mistral model: {str(e)}")
            # If we can't check model, we'll try to use it anyway
        
        # Initialize document loader for JSON
        try:
            logger.info("Loading and flattening JSON document...")
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmp:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Convert the JSON to a more flat structure if needed
                flat_json = flatten_json(json_data)
                json.dump(flat_json, tmp)
                tmp.flush()
                
                # Load the flattened JSON
                loader = JSONLoader(
                    file_path=tmp.name,
                    jq_schema='.',
                    text_content=False
                )
                documents = loader.load()
                logger.info(f"Successfully loaded {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error loading JSON: {str(e)}"}
            )
        
        # Setup embeddings with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Embedding attempt {attempt + 1}: Initializing OllamaEmbeddings...")
                embeddings = OllamaEmbeddings(
                    base_url="http://ollama:11434",
                    model="mistral"
                )
                
                # Test embeddings with a simple input
                logger.info("Testing embeddings with a simple query...")
                embeddings.embed_query("test")
                logger.info("Embedding test successful")
                break
            except Exception as e:
                logger.error(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Failed to initialize embeddings after {max_retries} attempts: {str(e)}. Please ensure the Ollama service is running and the Mistral model is installed."}
                    )
                logger.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        # Create vector store
        try:
            logger.info("Creating vector store with ChromaDB...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="/data/chromium"
            )
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Error creating vector store: {str(e)}. Please check ChromaDB connection."}
            )
        
        # Setup retrieval chain with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM attempt {attempt + 1}: Setting up retrieval chain...")
                llm = OllamaLLM(
                    base_url="http://ollama:11434", 
                    model="mistral"
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )
                
                # Generate documentation
                logger.info("Generating documentation...")
                query = "Document this integration. Explain what it does in simple terms, including the source systems, transformations, and target systems."
                result = qa_chain.invoke(query)["result"]
                logger.info("Documentation generated successfully")
                
                return {"documentation": result}
            except Exception as e:
                logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Failed to generate documentation after {max_retries} attempts: {str(e)}"}
                    )
                logger.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

@app.post("/analyze-with-format")
async def analyze_file_with_format(filename: str = Form(...), output_format: OutputFormat = Form(OutputFormat.html)):
    try:
        logger.info(f"Starting analysis of file: {filename} with output format: {output_format}")
        # Load the JSON file
        file_path = f"/data/{filename}"
        logger.info(f"Loading file from path: {file_path}")
        
        # Check if Ollama service is available
        try:
            logger.info("Checking if Ollama service is available...")
            # Ollama doesn't have a /api/health endpoint, use the root endpoint instead
            response = requests.get("http://ollama:11434", timeout=5)
            logger.info(f"Ollama connection check response: {response.status_code}")
            if response.status_code >= 400:
                logger.error(f"Ollama service returned status code: {response.status_code}")
                return JSONResponse(
                    status_code=503,
                    content={"error": "Ollama service is not responding correctly. Please check if the service is running properly."}
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama service: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Cannot connect to Ollama service: {str(e)}. Please ensure the Ollama container is running and that the Mistral model is installed."}
            )
        
        # Check if Mistral model is available
        try:
            logger.info("Checking if Mistral model is available...")
            response = requests.post(
                "http://ollama:11434/api/generate",
                json={"model": "mistral", "prompt": "test", "stream": False},
                timeout=5
            )
            logger.info(f"Mistral model check response: {response.status_code}")
            if response.status_code == 404:
                logger.error("Mistral model not found")
                return JSONResponse(
                    status_code=503,
                    content={"error": "The Mistral model is not installed in Ollama. Please install it with 'docker exec -it ollama ollama pull mistral'"}
                )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not check Mistral model: {str(e)}")
            # If we can't check model, we'll try to use it anyway
        
        # Initialize document loader for JSON
        try:
            logger.info("Loading and flattening JSON document...")
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmp:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Convert the JSON to a more flat structure if needed
                flat_json = flatten_json(json_data)
                json.dump(flat_json, tmp)
                tmp.flush()
                
                # Load the flattened JSON
                loader = JSONLoader(
                    file_path=tmp.name,
                    jq_schema='.',
                    text_content=False
                )
                documents = loader.load()
                logger.info(f"Successfully loaded {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error loading JSON: {str(e)}"}
            )
        
        # Setup embeddings with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Embedding attempt {attempt + 1}: Initializing OllamaEmbeddings...")
                embeddings = OllamaEmbeddings(
                    base_url="http://ollama:11434",
                    model="mistral"
                )
                
                # Test embeddings with a simple input
                logger.info("Testing embeddings with a simple query...")
                embeddings.embed_query("test")
                logger.info("Embedding test successful")
                break
            except Exception as e:
                logger.error(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Failed to initialize embeddings after {max_retries} attempts: {str(e)}. Please ensure the Ollama service is running and the Mistral model is installed."}
                    )
                logger.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        # Create vector store
        try:
            logger.info("Creating vector store with ChromaDB...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="/data/chromium"
            )
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Error creating vector store: {str(e)}. Please check ChromaDB connection."}
            )
        
        # Setup retrieval chain with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM attempt {attempt + 1}: Setting up retrieval chain...")
                llm = OllamaLLM(
                    base_url="http://ollama:11434", 
                    model="mistral"
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )
                
                # Generate documentation
                logger.info("Generating documentation...")
                query = "Document this integration. Explain what it does in simple terms, including the source systems, transformations, and target systems."
                result = qa_chain.invoke(query)["result"]
                logger.info("Documentation generated successfully")
                
                return {"documentation": result}
            except Exception as e:
                logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Failed to generate documentation after {max_retries} attempts: {str(e)}"}
                    )
                logger.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

@app.post("/analyze-smart")
async def analyze_file_with_training(filename: str = Form(...), output_format: OutputFormat = Form(OutputFormat.html)):
    """Generate documentation using training data for improved accuracy"""
    try:
        logger.info(f"Starting smart analysis of file: {filename} with output format: {output_format}")
        # Load the JSON file
        file_path = f"/data/{filename}"
        logger.info(f"Loading file from path: {file_path}")
        
        # Check if Ollama service is available
        try:
            logger.info("Checking if Ollama service is available...")
            response = requests.get("http://ollama:11434", timeout=5)
            if response.status_code >= 400:
                logger.error(f"Ollama service returned status code: {response.status_code}")
                return JSONResponse(
                    status_code=503,
                    content={"error": "Ollama service is not responding correctly. Please check if the service is running properly."}
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama service: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Cannot connect to Ollama service: {str(e)}. Please ensure the Ollama container is running and that the Mistral model is installed."}
            )
        
        # Initialize embeddings
        try:
            embeddings = OllamaEmbeddings(
                base_url="http://ollama:11434",
                model="mistral"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Failed to initialize embeddings: {str(e)}"}
            )
        
        # Process the document
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmp:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Convert the JSON to a flat structure
                flat_json = flatten_json(json_data)
                json.dump(flat_json, tmp)
                tmp.flush()
                
                # Load the flattened JSON
                loader = JSONLoader(
                    file_path=tmp.name,
                    jq_schema='.',
                    text_content=False
                )
                documents = loader.load()
                logger.info(f"Successfully loaded {len(documents)} documents from the input file")
        except Exception as e:
            logger.error(f"Error processing input file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing input file: {str(e)}"}
            )
        
        # Set up vector stores for the document and training data
        try:
            # Create vector store for the document
            doc_vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=f"/data/temp_vectorstore_{int(time.time())}",
            )
            
            # Check if training vector store exists
            training_vectorstore_path = "/data/training/vectorstore"
            if not os.path.exists(training_vectorstore_path):
                logger.warning("No training data available. Using standard analysis method.")
                # Fall back to regular analysis if no training data
                return await analyze_file_with_format(filename, output_format)
            
            # Load training vector store
            training_vectorstore = Chroma(
                persist_directory=training_vectorstore_path,
                embedding_function=embeddings,
                collection_name="training_examples"
            )
            logger.info("Loaded training vector store")
            
            # Find similar examples in the training data
            doc_text = " ".join([doc.page_content for doc in documents])
            similar_examples = training_vectorstore.similarity_search_with_score(
                doc_text, 
                k=3
            )
            
            if not similar_examples:
                logger.warning("No similar examples found in training data. Using standard analysis method.")
                # Fall back to regular analysis if no similar examples
                return await analyze_file_with_format(filename, output_format)
            
            logger.info(f"Found {len(similar_examples)} similar examples in training data")
            
            # Extract the similar documents and their scores
            examples_with_scores = []
            for doc, score in similar_examples:
                if hasattr(doc, 'metadata') and 'expected_documentation' in doc.metadata:
                    examples_with_scores.append({
                        'document': doc.page_content,
                        'documentation': doc.metadata['expected_documentation'],
                        'similarity_score': score
                    })
            
            # Set up LLM and retrieval
            llm = OllamaLLM(
                base_url="http://ollama:11434", 
                model="mistral"
            )
            
            # Construct a few-shot prompt with examples
            examples_text = ""
            for i, example in enumerate(examples_with_scores):
                examples_text += f"\nExample {i+1}:\n"
                examples_text += f"Integration Data: {example['document'][:500]}...\n" # Truncate for brevity
                examples_text += f"Documentation: {example['documentation']}\n"
            
            # Create enhanced prompt
            enhanced_prompt = f"""
            You are analyzing an integration JSON file to create documentation.
            
            I will provide you with examples of similar integrations and their documentation to guide you.
            
            {examples_text}
            
            Now, document the following integration in a similar style:
            
            Integration Data: {doc_text[:2000]}...
            
            Document this integration. Explain what it does in simple terms, including the source systems, transformations, and target systems.
            Be as specific as possible about the actual systems, data fields, and business logic involved.
            """
            
            # Generate the documentation using the enhanced prompt
            result = llm.invoke(enhanced_prompt)
            logger.info("Generated documentation using training examples")
            
            # Format and return the result based on requested output format
            if output_format == OutputFormat.html:
                return {"documentation": result, "format": "html"}
            
            # Create file name based on input file
            base_filename = filename.rsplit('.', 1)[0]
            
            if output_format == OutputFormat.markdown:
                # Save the markdown content
                md_filename = f"{base_filename}.md"
                md_filepath = f"/data/{md_filename}"
                with open(md_filepath, "w") as f:
                    f.write(f"# Integration Documentation: {base_filename}\n\n{result}")
                
                return {
                    "documentation": result,
                    "format": "markdown",
                    "filename": md_filename,
                    "download_url": f"/download/{md_filename}"
                }
            
            elif output_format == OutputFormat.pdf:
                # Convert to PDF
                pdf_filename = f"{base_filename}.pdf"
                pdf_filepath = f"/data/{pdf_filename}"
                
                # Create HTML content for PDF
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Integration Documentation: {base_filename}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                        h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                        .content {{ margin-top: 20px; white-space: pre-wrap; }}
                    </style>
                </head>
                <body>
                    <h1>Integration Documentation: {base_filename}</h1>
                    <div class="content">{result}</div>
                </body>
                </html>
                """
                
                try:
                    # Generate PDF file
                    pdfkit.from_string(html_content, pdf_filepath)
                    
                    return {
                        "documentation": result,
                        "format": "pdf",
                        "filename": pdf_filename,
                        "download_url": f"/download/{pdf_filename}"
                    }
                except Exception as e:
                    logger.error(f"Error generating PDF: {str(e)}")
                    # Fall back to HTML if PDF generation fails
                    return {
                        "documentation": result,
                        "format": "html",
                        "error": f"Failed to generate PDF: {str(e)}"
                    }
                
        except Exception as e:
            logger.error(f"Error in smart analysis: {str(e)}")
            logger.warning("Falling back to standard analysis method.")
            # Fall back to regular analysis if an error occurs
            return await analyze_file_with_format(filename, output_format)
            
    except Exception as e:
        logger.error(f"Unexpected error in smart analysis: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

@app.post("/train")
async def train_model(training_data: TrainingData):
    """Add training examples to improve documentation generation"""
    try:
        logger.info(f"Received {len(training_data.examples)} training examples")
        
        # Create training data directory if it doesn't exist
        os.makedirs("/data/training", exist_ok=True)
        
        # Save the training data to a file
        training_file = f"/data/training/training_data_{int(time.time())}.json"
        with open(training_file, "w") as f:
            f.write(training_data.json())
        
        # Setup embeddings
        try:
            embeddings = OllamaEmbeddings(
                base_url="http://ollama:11434",
                model="mistral"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": f"Failed to initialize embeddings: {str(e)}"}
            )
        
        # Process each training example
        for i, example in enumerate(training_data.examples):
            try:
                # Create a document for the training example
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmp:
                    # Flatten the JSON and add the expected documentation
                    flat_json = flatten_json(example.integration_json)
                    # Add the documentation as metadata
                    enriched_json = {
                        **flat_json,
                        "expected_documentation": example.documentation
                    }
                    json.dump(enriched_json, tmp)
                    tmp.flush()
                    
                    # Load the JSON file with metadata
                    loader = JSONLoader(
                        file_path=tmp.name,
                        jq_schema='.',
                        text_content=False
                    )
                    documents = loader.load()
                    logger.info(f"Processed training example {i+1} into {len(documents)} documents")
                
                # Add to the training vector store
                training_vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory="/data/training/vectorstore",
                    collection_name="training_examples"
                )
                logger.info(f"Added example {i+1} to training vector store")
                
            except Exception as e:
                logger.error(f"Error processing training example {i+1}: {str(e)}")
                # Continue with other examples even if one fails
        
        return {"message": f"Successfully processed {len(training_data.examples)} training examples"}
    except Exception as e:
        logger.error(f"Error in training endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing training data: {str(e)}"}
        )

@app.get("/training/status")
async def get_training_status():
    """Get information about the training data and examples"""
    try:
        # Create training data directory if it doesn't exist
        os.makedirs("/data/training", exist_ok=True)
        
        # Count training files
        training_files = [f for f in os.listdir("/data/training") if f.startswith("training_data_") and f.endswith(".json")]
        
        # Count examples in each file
        total_examples = 0
        example_details = []
        
        for training_file in training_files:
            file_path = f"/data/training/{training_file}"
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    examples_count = len(data.get("examples", []))
                    total_examples += examples_count
                    
                    timestamp = int(training_file.replace("training_data_", "").replace(".json", ""))
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                    
                    example_details.append({
                        "file": training_file,
                        "examples": examples_count,
                        "created": formatted_time
                    })
            except Exception as e:
                logger.error(f"Error reading training file {training_file}: {str(e)}")
        
        # Check if training vector store exists
        vectorstore_exists = os.path.exists("/data/training/vectorstore")
        
        return {
            "total_files": len(training_files),
            "total_examples": total_examples,
            "files": example_details,
            "vectorstore_exists": vectorstore_exists
        }
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting training status: {str(e)}"}
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serve a file for download"""
    file_path = f"/data/{filename}"
    
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"File {filename} not found"}
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

def flatten_json(json_data, prefix=""):
    """Flatten a nested JSON structure into a flat dictionary."""
    result = {}
    for key, value in json_data.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flatten_json(value, f"{new_key}."))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    result.update(flatten_json(item, f"{new_key}[{i}]."))
                else:
                    result[f"{new_key}[{i}]"] = str(item)
        else:
            result[new_key] = str(value)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)