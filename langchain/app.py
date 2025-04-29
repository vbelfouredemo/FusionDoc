from typing import List, Optional, Dict, Any, Union
import requests
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin, urlparse
import traceback
import base64
from io import BytesIO
from PIL import Image
import uuid
import time
import os
import re
import json
import zipfile
import glob
import logging
import markdown
import pdfkit
import tempfile

# Update Playwright imports to use async version
from playwright.async_api import async_playwright

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from enum import Enum
from contextlib import asynccontextmanager

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

class Integration(BaseModel):
    id: str
    name: str
    file_path: str

class WebScrapingConfig(BaseModel):
    url: str
    max_depth: int = Field(default=2, description="Maximum depth of links to follow from the starting URL")
    max_pages: int = Field(default=20, description="Maximum number of pages to crawl")
    include_patterns: Optional[List[str]] = Field(default=None, description="URL patterns to include (e.g. '/docs/')")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="URL patterns to exclude (e.g. '/blog/')")
    title: str = Field(default="Web Documentation", description="Title for this training material")

class WebCrawlResult(BaseModel):
    title: str
    url: str
    text_content: str
    crawled_at: str
    
class WebCrawlBatchResult(BaseModel):
    title: str
    pages_crawled: int
    urls: List[str]
    training_id: str

class ImageAnalysisRequest(BaseModel):
    image_path: str = Field(..., description="Path to the image file to analyze")
    format: Optional[OutputFormat] = Field(OutputFormat.markdown, description="Output format for the documentation")

class ImageUploadResponse(BaseModel):
    filename: str
    file_path: str
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI application startup and shutdown events."""
    # Startup code - runs before the application starts accepting requests
    # Retry a few times to accommodate Ollama startup time
    max_retries = 5
    for attempt in range(max_retries):
        logger.info(f"Startup check {attempt+1}/{max_retries}: Verifying Ollama and Mistral availability")
        try:
            # Check if Ollama is up
            response = requests.get("http://ollama:11434", timeout=5)
            if response.status_code < 400:
                # Ollama is up, now ensure Mistral is installed
                if ensure_mistral_model():
                    logger.info("Startup completed: Ollama is running and Mistral model is available")
                    break
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama service not ready yet: {str(e)}")
        
        # If we reach here, either Ollama isn't up or Mistral installation failed
        if attempt < max_retries - 1:
            delay = 10  # Wait 10 seconds between attempts
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            logger.error("Failed to ensure Mistral model availability after maximum retries")
    
    # Yield control back to FastAPI
    yield
    
    # Shutdown code - runs after the application finishes serving requests
    logger.info("Application shutting down")

# Define the app with lifespan
app = FastAPI(lifespan=lifespan)

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
os.makedirs("/data/uploads", exist_ok=True)
os.makedirs("/data/extracted", exist_ok=True)

def ensure_mistral_model():
    """
    Check if Mistral model is installed in Ollama, and attempt to install it if not present.
    Returns True if the model is available or was successfully installed, False otherwise.
    """
    logger.info("Checking if Mistral model is available in Ollama...")
    
    try:
        # Check available models
        response = requests.get("http://ollama:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            
            # Check if Mistral is already installed
            if any(m.get("name") == "mistral" for m in models):
                logger.info("Mistral model is already installed")
                return True
            
            # If not installed, try to pull it
            logger.info("Mistral model not found. Attempting to pull it...")
            
            pull_response = requests.post(
                "http://ollama:11434/api/pull",
                json={"name": "mistral"},
                timeout=600  # Longer timeout for model download
            )
            
            if pull_response.status_code == 200:
                logger.info("Successfully pulled Mistral model")
                return True
            else:
                logger.error(f"Failed to pull Mistral model: {pull_response.status_code}, {pull_response.text}")
                return False
        else:
            logger.error(f"Failed to check available models: {response.status_code}, {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama service: {str(e)}")
        return False

# Try to ensure Mistral model is installed at startup
@app.on_event("startup")
async def startup_event():
    # Retry a few times to accommodate Ollama startup time
    max_retries = 5
    for attempt in range(max_retries):
        logger.info(f"Startup check {attempt+1}/{max_retries}: Verifying Ollama and Mistral availability")
        try:
            # Check if Ollama is up
            response = requests.get("http://ollama:11434", timeout=5)
            if response.status_code < 400:
                # Ollama is up, now ensure Mistral is installed
                if ensure_mistral_model():
                    logger.info("Startup completed: Ollama is running and Mistral model is available")
                    break
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama service not ready yet: {str(e)}")
        
        # If we reach here, either Ollama isn't up or Mistral installation failed
        if attempt < max_retries - 1:
            delay = 10  # Wait 10 seconds between attempts
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            logger.error("Failed to ensure Mistral model availability after maximum retries")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Keep the original upload endpoint for backward compatibility
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

@app.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    """
    Upload a ZIP archive containing integration files and extract integrations.
    Returns a list of integrations found in the archive.
    """
    if not file.filename.lower().endswith('.zip'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only ZIP files are accepted"}
        )
    
    try:
        logger.info(f"Receiving ZIP file: {file.filename}")
        
        # Generate a unique ID for this upload
        upload_id = str(uuid.uuid4())
        zip_filename = f"{upload_id}_{file.filename}"
        zip_path = f"/data/uploads/{zip_filename}"
        
        # Save the uploaded zip file
        content = await file.read()
        with open(zip_path, "wb") as f:
            f.write(content)
        
        logger.info(f"ZIP file saved to {zip_path}")
        
        # Create extraction directory
        extract_dir = f"/data/extracted/{upload_id}"
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"ZIP file extracted to {extract_dir}")
        
        # Find all service files in the project structure
        # Pattern: projects/[ProjectName]/[ProjectVersion]/services/*.json
        integrations = find_integrations_in_archive(extract_dir)
        
        logger.info(f"Found {len(integrations)} integrations in the archive")
        
        return {
            "filename": zip_filename,
            "integrations": integrations
        }
    except zipfile.BadZipFile:
        logger.error(f"Bad ZIP file: {file.filename}")
        return JSONResponse(
            status_code=400,
            content={"error": "The file is not a valid ZIP archive"}
        )
    except Exception as e:
        logger.error(f"Error processing ZIP file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing ZIP file: {str(e)}"}
        )

def find_integrations_in_archive(extract_dir: str) -> List[Dict[str, str]]:
    """
    Find all integration JSON files in the extracted archive.
    Integration files are located in projects/[ProjectName]/[ProjectVersion]/services/
    The name of each integration is extracted from the filename pattern: [Name]_[UUID].json
    """
    # Find the projects directory within the extracted ZIP
    projects_paths = glob.glob(f"{extract_dir}/**/projects", recursive=True)
    
    if not projects_paths:
        logger.warning("No 'projects' directory found in the archive")
        return []
    
    integrations = []
    
    # For each projects directory, look for service files
    for projects_path in projects_paths:
        logger.info(f"Searching for integrations in {projects_path}")
        
        # Find all JSON files in any services directory
        service_files = glob.glob(f"{projects_path}/**/services/*.json", recursive=True)
        
        for file_path in service_files:
            try:
                # Extract integration name from filename
                filename = os.path.basename(file_path)
                
                # The pattern is typically: [Integration Name]_[UUID].json
                match = re.match(r"(.*?)_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.json", filename)
                
                if match:
                    integration_name = match.group(1)
                    integration_id = match.group(2)
                    
                    integrations.append({
                        "id": integration_id,
                        "name": integration_name,
                        "file_path": file_path
                    })
                    logger.info(f"Found integration: {integration_name} ({integration_id})")
                else:
                    logger.warning(f"File doesn't match expected pattern: {filename}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
    
    return integrations

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
                timeout=15
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
        retry_delay = 5
        
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
                llm = Ollama(
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
        logger.error(f"Error in analyze: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

# Fix duplicate code in analyze-with-format function
@app.post("/analyze-with-format")
async def analyze_file_with_format(
    filename: str = Form(...), 
    output_format: OutputFormat = Form(OutputFormat.html),
    integrations: Optional[str] = Form(None)
):
    """
    Analyze integration files from a ZIP archive and generate documentation.
    """
    try:
        logger.info(f"Starting analysis of file: {filename} with output format: {output_format}")
        
        # Check if this is a ZIP file analysis with selected integrations
        if integrations:
            selected_integrations = json.loads(integrations)
            
            if not selected_integrations:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No integrations selected for documentation"}
                )
            
            # Extract the upload ID from the filename
            upload_id = filename.split('_')[0]
            extract_dir = f"/data/extracted/{upload_id}"
            
            if not os.path.exists(extract_dir):
                return JSONResponse(
                    status_code=404,
                    content={"error": "Extracted archive not found. Please upload the file again."}
                )
            
            # Get file paths for selected integrations
            all_integrations = find_integrations_in_archive(extract_dir)
            selected_files = []
            selected_names = []
            selected_ids = []
            
            for integration in all_integrations:
                if integration["id"] in selected_integrations:
                    selected_files.append(integration["file_path"])
                    selected_names.append(integration["name"])
                    selected_ids.append(integration["id"])
            
            if not selected_files:
                return JSONResponse(
                    status_code=404,
                    content={"error": "No matching integration files found for the selected integrations"}
                )
            
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
            
            # Initialize embeddings with retry logic
            max_retries = 3
            retry_delay = 5
            
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
            
            # Initialize LLM
            llm = Ollama(
                base_url="http://ollama:11434", 
                model="mistral"
            )
            
            # Process each selected integration file and generate documentation
            all_docs = []
            for i, file_path in enumerate(selected_files):
                integration_name = selected_names[i]
                integration_id = selected_ids[i]
                logger.info(f"Processing integration: {integration_name} ({file_path})")
                
                try:
                    # Analyze the integration flow to get comprehensive information
                    flow_analysis = analyze_integration_flow(integration_id, extract_dir)
                    
                    # Load and process the JSON file
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Convert the JSON to a more flat structure for processing
                    flat_json = flatten_json(json_data)
                    
                    # Add the flow analysis information to the flat JSON
                    flow_json = {
                        "flow_analysis": flow_analysis,
                        "basic_data": flat_json
                    }
                    
                    # Create a temporary file for document loading
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmp:
                        json.dump(flow_json, tmp)
                        tmp.flush()
                        
                        # Load the enriched JSON using the document loader
                        loader = JSONLoader(
                            file_path=tmp.name,
                            jq_schema='.',
                            text_content=False
                        )
                        documents = loader.load()
                        logger.info(f"Successfully loaded {len(documents)} documents from {integration_name}")
                except Exception as e:
                    logger.error(f"Error processing integration {integration_name}: {str(e)}")
                    continue
                
                # Create a vector store for retrieval
                try:
                    logger.info("Creating vector store with ChromaDB...")
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        persist_directory=f"/data/temp_vectorstore_{integration_name}_{int(time.time())}",
                    )
                    logger.info("Vector store created successfully")
                except Exception as e:
                    logger.error(f"Error creating vector store: {str(e)}")
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Error creating vector store: {str(e)}. Please check ChromaDB connection."}
                    )
                
                # Create a retrieval chain for this integration
                try:
                    logger.info(f"LLM attempt {attempt + 1}: Setting up retrieval chain...")
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever()
                    )
                    
                    # Generate component structure description
                    component_descriptions = []
                    if flow_analysis and "components" in flow_analysis:
                        # Sort components by order
                        components = sorted(
                            flow_analysis["components"], 
                            key=lambda x: float(x["order"]) if x["order"] is not None else float('inf')
                        )
                        
                        for component in components:
                            component_desc = f"- {component['name']}"
                            if component["order"]:
                                component_desc += f" (Order: {component['order']})"
                            if component["description"]:
                                component_desc += f": {component['description']}"
                            if component.get("parent_id"):
                                parent_name = next(
                                    (c["name"] for c in components if c["id"] == component["parent_id"]), 
                                    component["parent_id"]
                                )
                                component_desc += f" [Child of: {parent_name}]"
                            component_descriptions.append(component_desc)
                    
                    # Generate documentation with a specific prompt for this integration
                    flow_structure = "\n".join(component_descriptions) if component_descriptions else "No component data available"
                    
                    query = f"""Document this integration called '{integration_name}'. 
                    
                    The integration flow structure is as follows:
                    {flow_structure}
                    
                    Explain what this integration does in simple terms, including:
                    1. The source systems and how data enters the flow
                    2. The flow of data through components, especially any branching or transformation logic
                    3. The target systems where data is sent
                    4. Any business purpose this integration serves
                    
                    Be specific about what data fields are mapped, and any routing or conditional logic."""
                    
                    result = qa_chain.invoke(query)["result"]
                    
                    # Add to the documentation collection
                    all_docs.append({
                        "name": integration_name,
                        "documentation": result,
                        "flow_structure": flow_structure
                    })
                except Exception as e:
                    logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        return JSONResponse(
                            status_code=503,
                            content={"error": f"Failed to generate documentation after {max_retries} attempts: {str(e)}"}
                        )
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            # Combine all documentation into a single document
            combined_doc = ""
            for doc in all_docs:
                combined_doc += f"## {doc['name']}\n\n"
                if doc.get("flow_structure"):
                    combined_doc += f"### Integration Components\n\n{doc['flow_structure']}\n\n"
                combined_doc += f"### Integration Description\n\n{doc['documentation']}\n\n"
                combined_doc += "---\n\n"
            
            # Format and return the results based on requested format
            base_filename = f"integration_docs_{int(time.time())}"
            
            if output_format == OutputFormat.html:
                # Convert markdown to HTML for display
                html_doc = markdown.markdown(combined_doc)
                return {"documentation": html_doc, "format": "html"}
            
            elif output_format == OutputFormat.markdown:
                # Save the markdown content
                md_filename = f"{base_filename}.md"
                md_filepath = f"/data/{md_filename}"
                
                with open(md_filepath, "w") as f:
                    f.write(f"# Integration Documentation\n\n")
                    f.write(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(combined_doc)
                
                return {
                    "documentation": combined_doc,
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
                    <title>Integration Documentation</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                        h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                        h2 {{ color: #3498db; margin-top: 30px; }}
                        h3 {{ color: #2980b9; margin-top: 20px; }}
                        hr {{ border: 0; height: 1px; background: #eee; margin: 30px 0; }}
                        .content {{ margin-top: 20px; }}
                        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Integration Documentation</h1>
                    <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div class="content">
                        {markdown.markdown(combined_doc)}
                    </div>
                </body>
                </html>
                """
                
                try:
                    # Generate PDF file
                    pdfkit.from_string(html_content, pdf_filepath)
                    
                    return {
                        "documentation": markdown.markdown(combined_doc),
                        "format": "pdf",
                        "filename": pdf_filename,
                        "download_url": f"/download/{pdf_filename}"
                    }
                except Exception as e:
                    logger.error(f"Error generating PDF: {str(e)}")
                    # Fall back to HTML if PDF generation fails
                    return {
                        "documentation": markdown.markdown(combined_doc),
                        "format": "html",
                        "error": f"Failed to generate PDF: {str(e)}"
                    }
    
    except Exception as e:
        logger.error(f"Error in analyze-with-format: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

@app.post("/analyze-smart")
async def analyze_file_with_training(
    filename: str = Form(...), 
    output_format: OutputFormat = Form(OutputFormat.html),
    integrations: Optional[str] = Form(None)
):
    """Generate documentation using training data for improved accuracy"""
    try:
        logger.info(f"Starting smart analysis of file: {filename} with output format: {output_format}")

        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            base_url="http://ollama:11434",
            model="mistral"
        )

        # Load the training vectorstore
        training_vectorstore = None
        training_path = "/data/training/vectorstore"
        if os.path.exists(training_path):
            logger.info("Loading training vectorstore...")
            try:
                training_vectorstore = Chroma(
                    persist_directory=training_path,
                    embedding_function=embeddings,
                    collection_name="training_examples"
                )
                logger.info("Training vectorstore loaded successfully")
            except Exception as e:
                logger.error(f"Error loading training vectorstore: {str(e)}")
                # Continue without training data
                
        # If no training vectorstore exists, create an empty one
        if training_vectorstore is None:
            logger.warning("No training vectorstore found. Using empty vectorstore.")
            # Create an empty vectorstore or load examples from langchain directory
            langchain_examples = []
            if os.path.exists("/data/langchain"):
                example_files = [f for f in os.listdir("/data/langchain") if f.endswith(".json")]
                if example_files:
                    logger.info(f"Found {len(example_files)} example files in langchain directory")
                    # Load a few examples to create a minimal training vectorstore
                    documents = []
                    for example_file in example_files[:5]:  # Limit to 5 examples
                        try:
                            with open(f"/data/langchain/{example_file}", "r") as f:
                                example_data = json.load(f)
                                if "integration_json" in example_data and "documentation" in example_data:
                                    # Create metadata with expected documentation
                                    metadata = {"expected_documentation": example_data["documentation"]}
                                    # Create document text from integration JSON
                                    if isinstance(example_data["integration_json"], dict):
                                        doc_text = json.dumps(example_data["integration_json"])
                                    else:
                                        doc_text = str(example_data["integration_json"])
                                    documents.append(Document(page_content=doc_text, metadata=metadata))
                        except Exception as e:
                            logger.error(f"Error loading example file {example_file}: {str(e)}")
                    
                    if documents:
                        logger.info(f"Creating training vectorstore with {len(documents)} documents")
                        training_vectorstore = Chroma.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            persist_directory="/data/training/vectorstore_temp",
                            collection_name="training_examples"
                        )
            
            # If still no training data, use fallback approach
            if training_vectorstore is None:
                logger.warning("No examples found for training vectorstore. Creating empty vectorstore.")
                # Create an empty training vectorstore (will prevent errors but won't help with examples)
                training_vectorstore = Chroma(
                    persist_directory="/data/training/vectorstore_temp",
                    embedding_function=embeddings,
                    collection_name="training_examples"
                )

        # Check if this is a ZIP file analysis with selected integrations
        if integrations:
            selected_integrations = json.loads(integrations)
            
            if not selected_integrations:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No integrations selected for documentation"}
                )
            
            # Extract the upload ID from the filename
            upload_id = filename.split('_')[0]
            extract_dir = f"/data/extracted/{upload_id}"
            
            if not os.path.exists(extract_dir):
                return JSONResponse(
                    status_code=404,
                    content={"error": "Extracted archive not found. Please upload the file again."}
                )
            
            # Get file paths for selected integrations
            all_integrations = find_integrations_in_archive(extract_dir)
            selected_files = []
            selected_names = []
            selected_ids = []
            
            for integration in all_integrations:
                if integration["id"] in selected_integrations:
                    selected_files.append(integration["file_path"])
                    selected_names.append(integration["name"])
                    selected_ids.append(integration["id"])
            
            if not selected_files:
                return JSONResponse(
                    status_code=404,
                    content={"error": "No matching integration files found for the selected integrations"}
                )
            
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
            
            # Initialize embeddings with retry logic
            max_retries = 3
            retry_delay = 5
            
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
            
            # Initialize LLM
            llm = Ollama(
                base_url="http://ollama:11434", 
                model="mistral"
            )
            
            # Process each selected integration file and generate documentation
            all_docs = []
            for i, file_path in enumerate(selected_files):
                integration_name = selected_names[i]
                integration_id = selected_ids[i]
                logger.info(f"Processing integration: {integration_name} ({file_path})")
                
                try:
                    # Analyze the integration flow to get comprehensive information
                    flow_analysis = analyze_integration_flow(integration_id, extract_dir)
                    
                    # Load and process the JSON file
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Convert the JSON to a more flat structure for processing
                    flat_json = flatten_json(json_data)
                    
                    # Add the flow analysis information to the flat JSON
                    flow_json = {
                        "flow_analysis": flow_analysis,
                        "basic_data": flat_json
                    }
                    
                    # Create a temporary file for document loading
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmp:
                        json.dump(flow_json, tmp)
                        tmp.flush()
                        
                        # Load the enriched JSON using the document loader
                        loader = JSONLoader(
                            file_path=tmp.name,
                            jq_schema='.',
                            text_content=False
                        )
                        documents = loader.load()
                        logger.info(f"Successfully loaded {len(documents)} documents from {integration_name}")
                except Exception as e:
                    logger.error(f"Error processing integration {integration_name}: {str(e)}")
                    continue
                
                # Create a vector store for retrieval
                try:
                    logger.info("Creating vector store with ChromaDB...")
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        persist_directory=f"/data/temp_vectorstore_{integration_name}_{int(time.time())}",
                    )
                    logger.info("Vector store created successfully")
                except Exception as e:
                    logger.error(f"Error creating vector store: {str(e)}")
                    return JSONResponse(
                        status_code=503,
                        content={"error": f"Error creating vector store: {str(e)}. Please check ChromaDB connection."}
                    )
                
                # Create a retrieval chain for this integration
                try:
                    logger.info(f"LLM attempt {attempt + 1}: Setting up retrieval chain...")
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever()
                    )
                    
                    # Generate component structure description
                    component_descriptions = []
                    if flow_analysis and "components" in flow_analysis:
                        # Sort components by order
                        components = sorted(
                            flow_analysis["components"], 
                            key=lambda x: float(x["order"]) if x["order"] is not None else float('inf')
                        )
                        
                        for component in components:
                            component_desc = f"- {component['name']}"
                            if component["order"]:
                                component_desc += f" (Order: {component['order']})"
                            if component["description"]:
                                component_desc += f": {component['description']}"
                            if component.get("parent_id"):
                                parent_name = next(
                                    (c["name"] for c in components if c["id"] == component["parent_id"]), 
                                    component["parent_id"]
                                )
                                component_desc += f" [Child of: {parent_name}]"
                            component_descriptions.append(component_desc)
                    
                    # Generate documentation with a specific prompt for this integration
                    flow_structure = "\n".join(component_descriptions) if component_descriptions else "No component data available"
                    
                    query = f"""Document this integration called '{integration_name}'. 
                    
                    The integration flow structure is as follows:
                    {flow_structure}
                    
                    Explain what this integration does in simple terms, including:
                    1. The source systems and how data enters the flow
                    2. The flow of data through components, especially any branching or transformation logic
                    3. The target systems where data is sent
                    4. Any business purpose this integration serves
                    
                    Be specific about what data fields are mapped, and any routing or conditional logic."""
                    
                    result = qa_chain.invoke(query)["result"]
                    
                    # Add to the documentation collection
                    all_docs.append({
                        "name": integration_name,
                        "documentation": result,
                        "flow_structure": flow_structure
                    })
                except Exception as e:
                    logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        return JSONResponse(
                            status_code=503,
                            content={"error": f"Failed to generate documentation after {max_retries} attempts: {str(e)}"}
                        )
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            # Combine all documentation into a single document
            combined_doc = ""
            for doc in all_docs:
                combined_doc += f"## {doc['name']}\n\n"
                if doc.get("flow_structure"):
                    combined_doc += f"### Integration Components\n\n{doc['flow_structure']}\n\n"
                combined_doc += f"### Integration Description\n\n{doc['documentation']}\n\n"
                combined_doc += "---\n\n"
            
            # Format and return the results based on requested format
            base_filename = f"integration_docs_{int(time.time())}"
            
            if output_format == OutputFormat.html:
                # Convert markdown to HTML for display
                html_doc = markdown.markdown(combined_doc)
                return {"documentation": html_doc, "format": "html"}
            
            elif output_format == OutputFormat.markdown:
                # Save the markdown content
                md_filename = f"{base_filename}.md"
                md_filepath = f"/data/{md_filename}"
                
                with open(md_filepath, "w") as f:
                    f.write(f"# Integration Documentation\n\n")
                    f.write(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(combined_doc)
                
                return {
                    "documentation": combined_doc,
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
                    <title>Integration Documentation</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                        h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                        h2 {{ color: #3498db; margin-top: 30px; }}
                        h3 {{ color: #2980b9; margin-top: 20px; }}
                        hr {{ border: 0; height: 1px; background: #eee; margin: 30px 0; }}
                        .content {{ margin-top: 20px; }}
                        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Integration Documentation</h1>
                    <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div class="content">
                        {markdown.markdown(combined_doc)}
                    </div>
                </body>
                </html>
                """
                
                try:
                    # Generate PDF file
                    pdfkit.from_string(html_content, pdf_filepath)
                    
                    return {
                        "documentation": markdown.markdown(combined_doc),
                        "format": "pdf",
                        "filename": pdf_filename,
                        "download_url": f"/download/{pdf_filename}"
                    }
                except Exception as e:
                    logger.error(f"Error generating PDF: {str(e)}")
                    # Fall back to HTML if PDF generation fails
                    return {
                        "documentation": markdown.markdown(combined_doc),
                        "format": "html",
                        "error": f"Failed to generate PDF: {str(e)}"
                    }
    
    except Exception as e:
        logger.error(f"Error in analyze-smart: {str(e)}")
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
        
        # Also check the langchain directory for individual training examples
        langchain_examples = []
        if os.path.exists("/data/langchain"):
            langchain_examples = [f for f in os.listdir("/data/langchain") if f.endswith(".json")]
        
        # Count examples in each file
        total_examples = 0
        example_details = []
        
        # Process traditional training files
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
                        "created": formatted_time,
                        "type": "batch"
                    })
            except Exception as e:
                logger.error(f"Error reading training file {training_file}: {str(e)}")
        
        # Process individual examples from langchain directory
        langchain_file_details = {}
        for example_file in langchain_examples:
            if not example_file.endswith(".json"):
                continue
                
            file_path = f"/data/langchain/{example_file}"
            try:
                # Extract integration name from filename pattern: Name_UUID.json
                name_match = re.match(r"(.*?)_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.json", example_file)
                if name_match:
                    integration_name = name_match.group(1)
                    integration_id = name_match.group(2)
                    
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        
                        # Calculate quality metrics based on documentation length and content
                        doc_text = data.get("documentation", "")
                        doc_length = len(doc_text)
                        
                        # Simple quality score calculation
                        quality_score = min(1.0, doc_length / 500)  # Normalize to max of 1.0
                        
                        # Additional quality factors
                        has_structure = 1 if "##" in doc_text or "#" in doc_text else 0
                        has_details = 1 if len(doc_text.split()) > 100 else 0
                        
                        # Combined quality metrics (range 0.0 - 1.0)
                        combined_quality = (quality_score * 0.6) + (has_structure * 0.2) + (has_details * 0.2)
                        
                        # Get file creation timestamp
                        file_timestamp = os.path.getctime(file_path)
                        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_timestamp))
                        
                        # Group by integration name for UI display
                        if integration_name not in langchain_file_details:
                            langchain_file_details[integration_name] = {
                                "count": 0,
                                "examples": [],
                                "total_quality": 0
                            }
                        
                        langchain_file_details[integration_name]["count"] += 1
                        langchain_file_details[integration_name]["total_quality"] += combined_quality
                        langchain_file_details[integration_name]["examples"].append({
                            "id": integration_id,
                            "filename": example_file,
                            "created": formatted_time,
                            "quality": combined_quality
                        })
                        
                        total_examples += 1
            except Exception as e:
                logger.error(f"Error reading langchain example file {example_file}: {str(e)}")
        
        # Convert langchain file details to list format for response
        for integration_name, details in langchain_file_details.items():
            avg_quality = details["total_quality"] / details["count"] if details["count"] > 0 else 0
            example_details.append({
                "file": integration_name,
                "examples": details["count"],
                "created": details["examples"][0]["created"] if details["examples"] else "Unknown",
                "type": "integration",
                "quality": avg_quality,
                "individual_examples": details["examples"]
            })
        
        # Check if training vector store exists
        vectorstore_exists = os.path.exists("/data/training/vectorstore")
        
        # Calculate quality distribution for visualization
        quality_ranges = {
            "high": 0,    # 0.8 - 1.0
            "medium": 0,  # 0.5 - 0.79
            "low": 0      # 0.0 - 0.49
        }
        
        total_with_quality = 0
        for detail in example_details:
            if "quality" in detail:
                quality = detail["quality"]
                count = detail["examples"]
                total_with_quality += count
                
                if quality >= 0.8:
                    quality_ranges["high"] += count
                elif quality >= 0.5:
                    quality_ranges["medium"] += count
                else:
                    quality_ranges["low"] += count
        
        # Convert to percentages
        if total_with_quality > 0:
            quality_ranges["high_percent"] = round((quality_ranges["high"] / total_with_quality) * 100)
            quality_ranges["medium_percent"] = round((quality_ranges["medium"] / total_with_quality) * 100)
            quality_ranges["low_percent"] = round((quality_ranges["low"] / total_with_quality) * 100)
        else:
            quality_ranges["high_percent"] = 0
            quality_ranges["medium_percent"] = 0
            quality_ranges["low_percent"] = 0
        
        return {
            "total_files": len(training_files) + len(langchain_file_details),
            "total_examples": total_examples,
            "files": example_details,
            "vectorstore_exists": vectorstore_exists,
            "quality_metrics": quality_ranges,
            "quality_available": total_with_quality > 0
        }
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting training status: {str(e)}"}
        )

@app.get("/training/list")
async def list_training_examples():
    """List all individual training examples that can be managed"""
    try:
        examples = []
        
        # Check the langchain directory for individual training examples
        if os.path.exists("/data/langchain"):
            for example_file in os.listdir("/data/langchain"):
                if example_file.endswith(".json"):
                    file_path = f"/data/langchain/{example_file}"
                    try:
                        # Get file stats for creation time
                        file_stats = os.stat(file_path)
                        created_time = time.strftime("%Y-%m-%d %H:%M:%S", 
                                                   time.localtime(file_stats.st_ctime))
                        
                        # Extract integration name from filename pattern: Name_UUID.json
                        parts = example_file.split('_')
                        integration_name = '_'.join(parts[:-1]) if len(parts) > 1 else example_file
                        integration_id = parts[-1].replace(".json", "") if len(parts) > 1 else ""
                        
                        # Get a snippet of the documentation for preview
                        doc_preview = ""
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if "documentation" in data:
                                # Get first 100 chars of documentation as preview
                                doc_preview = data["documentation"][:100] + "..." if len(data["documentation"]) > 100 else data["documentation"]
                        
                        examples.append({
                            "id": integration_id,
                            "filename": example_file,
                            "integration_name": integration_name,
                            "created": created_time,
                            "preview": doc_preview,
                            "file_path": file_path
                        })
                    except Exception as e:
                        logger.error(f"Error reading example file {example_file}: {str(e)}")
        
        # Sort by creation time, newest first
        examples.sort(key=lambda x: x["created"], reverse=True)
        
        return {"examples": examples}
    except Exception as e:
        logger.error(f"Error listing training examples: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error listing training examples: {str(e)}"}
        )

@app.delete("/training/delete/{example_id}")
async def delete_training_example(example_id: str):
    """Delete a specific training example by ID"""
    try:
        # Find the example file with this ID
        example_found = False
        example_file = None
        markdown_file = None
        
        if os.path.exists("/data/langchain"):
            for file in os.listdir("/data/langchain"):
                if file.endswith(f"{example_id}.json"):
                    example_file = f"/data/langchain/{file}"
                    example_found = True
                    # Also look for accompanying markdown file
                    md_file = file.replace(".json", ".md")
                    if os.path.exists(f"/data/langchain/{md_file}"):
                        markdown_file = f"/data/langchain/{md_file}"
                    break
        
        if not example_found:
            return JSONResponse(
                status_code=404,
                content={"error": f"Training example with ID {example_id} not found"}
            )
        
        # Delete the example file
        os.remove(example_file)
        logger.info(f"Deleted training example file: {example_file}")
        
        # Delete accompanying markdown file if it exists
        if markdown_file and os.path.exists(markdown_file):
            os.remove(markdown_file)
            logger.info(f"Deleted training example markdown: {markdown_file}")
        
        # This will require retraining the vector store
        # Mark the vector store as needing rebuild
        with open("/data/training/rebuild_required", "w") as f:
            f.write("Training examples have been deleted. Vectorstore needs rebuild.")
        
        return {"message": f"Successfully deleted training example with ID {example_id}"}
    except Exception as e:
        logger.error(f"Error deleting training example: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error deleting training example: {str(e)}"}
        )

@app.post("/training/rebuild-vectorstore")
async def rebuild_vectorstore():
    """Rebuild the training vectorstore from existing examples"""
    try:
        if not os.path.exists("/data/langchain"):
            return JSONResponse(
                status_code=400,
                content={"error": "No training examples found to rebuild vectorstore"}
            )
        
        # Get list of example files
        example_files = [f for f in os.listdir("/data/langchain") if f.endswith(".json")]
        
        if not example_files:
            return JSONResponse(
                status_code=400,
                content={"error": "No training examples found to rebuild vectorstore"}
            )
        
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            base_url="http://ollama:11434",
            model="mistral"
        )
        
        # Create a fresh vectorstore directory
        vectorstore_dir = "/data/training/vectorstore"
        os.makedirs(vectorstore_dir, exist_ok=True)
        
        # Check if existing vectorstore
        if os.path.exists(f"{vectorstore_dir}/chroma.sqlite3"):
            # Load existing vectorstore
            logger.info("Loading existing training vectorstore")
            vectorstore = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embeddings,
                collection_name="training_examples"
            )
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
        else:
            # Create new vectorstore
            logger.info("Creating new training vectorstore")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=vectorstore_dir,
                collection_name="training_examples"
            )
        
        # Persist changes
        vectorstore.persist()
        logger.info(f"Successfully added web crawl data to training vectorstore")
        
    except Exception as e:
        logger.error(f"Error creating vectorstore from web crawl data: {str(e)}")
        raise

@app.post("/train/web-crawl")
async def web_crawl_for_training(config: WebScrapingConfig):
    """
    Crawl a website for documentation to use as training data.
    The content from each page will be extracted and used to train the model.
    """
    logger.info(f"Starting web crawl for training data from: {config.url}")
    
    try:
        # Create a unique ID for this training batch
        training_id = str(uuid.uuid4())
        
        # Track crawled URLs to avoid duplicates
        crawled_urls = set()
        crawl_results = []
        
        # Set up the initial URL to crawl
        urls_to_crawl = [(config.url, 0)]  # (url, depth)
        
        # Start crawling
        logger.info(f"Beginning web crawl with max depth {config.max_depth} and max pages {config.max_pages}")
        
        # Try to use Playwright for JavaScript-heavy sites
        try:
            # Use async Playwright API correctly
            async with async_playwright() as playwright:
                logger.info("Starting Playwright browser for web crawling")
                browser = await playwright.chromium.launch()
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
                )
                page = await context.new_page()
                
                # Process URLs in queue
                while urls_to_crawl and len(crawled_urls) < config.max_pages:
                    # Get the next URL and its depth
                    current_url, current_depth = urls_to_crawl.pop(0)
                    
                    # Skip if we've already crawled this URL
                    if current_url in crawled_urls:
                        continue
                        
                    logger.info(f"Crawling URL with Playwright: {current_url} (depth: {current_depth})")
                    
                    try:
                        # Visit the page and wait for it to load (async)
                        await page.goto(current_url, wait_until="networkidle", timeout=30000)
                        
                        # Wait for potential content to load
                        await page.wait_for_timeout(3000)
                        
                        # Get the page title
                        title = await page.title()
                        
                        # Get the page content
                        content = await page.content()
                        
                        # Parse with BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # First try to extract content using standard methods
                        main_content = extract_main_content(soup)
                        
                        # If content is still limited, try direct text extraction from Playwright
                        if len(main_content.strip()) < 100:
                            logger.info("Attempting specialized extraction with Playwright selectors")
                            
                            # Try various selectors that might contain the main content
                            content_selectors = [
                                "article", 
                                ".topic-content", 
                                "#topic-content", 
                                ".documentation",
                                ".content-wrapper",
                                ".main-content",
                                "main",
                                ".content",
                                "#content"
                            ]
                            
                            for selector in content_selectors:
                                try:
                                    elements = await page.query_selector_all(selector)
                                    if elements:
                                        extracted_texts = []
                                        for element in elements:
                                            text = await element.inner_text()
                                            if text and len(text.strip()) > 50:
                                                extracted_texts.append(text)
                                        
                                        if extracted_texts:
                                            combined_text = "\n\n".join(extracted_texts)
                                            if len(combined_text.strip()) > 100:
                                                main_content = combined_text
                                                logger.info(f"Found content using Playwright selector: {selector}")
                                                break
                                except Exception as e:
                                    logger.warning(f"Error extracting with selector {selector}: {str(e)}")
                        
                        # Skip if no significant content was extracted
                        if len(main_content.strip()) < 50:
                            logger.warning(f"Skipping {current_url}: insufficient content")
                            continue
                        
                        # Store the crawl result
                        crawl_result = WebCrawlResult(
                            title=title,
                            url=current_url,
                            text_content=main_content,
                            crawled_at=time.strftime('%Y-%m-%d %H:%M:%S')
                        )
                        crawl_results.append(crawl_result)
                        
                        # Add to crawled set
                        crawled_urls.add(current_url)
                        
                        # If we're not at max depth, find new links to crawl
                        if current_depth < config.max_depth:
                            # Find links using Playwright
                            hrefs = []
                            links = await page.query_selector_all('a')
                            
                            for link in links:
                                try:
                                    href = await link.get_attribute('href')
                                    if href:
                                        hrefs.append(href)
                                except Exception as e:
                                    continue
                            
                            # Process collected links
                            for href in hrefs:
                                # Normalize URL
                                next_url = urljoin(current_url, href)
                                
                                # Skip URLs outside the original domain
                                if not is_same_domain(config.url, next_url):
                                    continue
                                    
                                # Skip URLs that have already been crawled or queued
                                if next_url in crawled_urls or any(next_url == u for u, _ in urls_to_crawl):
                                    continue
                                
                                # Check include/exclude patterns
                                if should_crawl_url(next_url, config.include_patterns, config.exclude_patterns):
                                    urls_to_crawl.append((next_url, current_depth + 1))
                    
                    except Exception as e:
                        logger.error(f"Error crawling {current_url} with Playwright: {str(e)}")
                        continue
                
                # Close browser (async)
                await browser.close()
                logger.info("Playwright browser closed")
        
        except Exception as e:
            logger.error(f"Failed to initialize or use Playwright: {str(e)}")
            logger.info("Falling back to requests-based crawling")
            
            # Fallback to simple requests-based crawling
            while urls_to_crawl and len(crawled_urls) < config.max_pages:
                # Get the next URL and its depth
                current_url, current_depth = urls_to_crawl.pop(0)
                
                # Skip if we've already crawled this URL
                if current_url in crawled_urls:
                    continue
                    
                logger.info(f"Crawling URL with requests: {current_url} (depth: {current_depth})")
                
                try:
                    # Make the request with headers to look more like a browser
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Referer': 'https://www.google.com/',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Cache-Control': 'max-age=0'
                    }
                    
                    response = requests.get(current_url, headers=headers, timeout=30)
                    
                    # Skip if not successful
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch {current_url}: status code {response.status_code}")
                        continue
                        
                    # Parse the HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else "No Title"
                    
                    # Extract the content
                    main_content = extract_main_content(soup)
                    
                    # Skip if no significant content was extracted
                    if len(main_content.strip()) < 50:
                        logger.warning(f"Skipping {current_url}: insufficient content")
                        continue
                    
                    # Store the crawl result
                    crawl_result = WebCrawlResult(
                        title=title,
                        url=current_url,
                        text_content=main_content,
                        crawled_at=time.strftime('%Y-%m-%d %H:%M:%S')
                    )
                    crawl_results.append(crawl_result)
                    
                    # Add to crawled set
                    crawled_urls.add(current_url)
                    
                    # If we're not at max depth, find new links to crawl
                    if current_depth < config.max_depth:
                        # Find all links
                        links = soup.find_all('a', href=True)
                        
                        for link in links:
                            # Normalize URL
                            next_url = urljoin(current_url, link['href'])
                            
                            # Skip URLs outside the original domain
                            if not is_same_domain(config.url, next_url):
                                continue
                                
                            # Skip URLs that have already been crawled or queued
                            if next_url in crawled_urls or any(next_url == u for u, _ in urls_to_crawl):
                                continue
                                
                            # Check include/exclude patterns
                            if should_crawl_url(next_url, config.include_patterns, config.exclude_patterns):
                                urls_to_crawl.append((next_url, current_depth + 1))
                                
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {str(e)}")
                    continue
        
        # Process and save the crawled content as training data
        if crawl_results:
            logger.info(f"Successfully crawled {len(crawl_results)} pages")
            
            try:
                # Create training directory if it doesn't exist
                os.makedirs("/data/training/web", exist_ok=True)
                
                # Save all crawled content to a single file
                training_data_path = f"/data/training/web/web_training_{training_id}.json"
                
                # Convert crawl results to training data
                training_data = {
                    "title": config.title,
                    "source_url": config.url,
                    "crawled_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "pages": [
                        {
                            "url": result.url,
                            "title": result.title,
                            "content": result.text_content
                        }
                        for result in crawl_results
                    ]
                }
                
                with open(training_data_path, "w") as f:
                    json.dump(training_data, f, indent=2)
                
                # Add to vector store
                create_vectorstore_from_web_crawl(training_data_path, training_id)
                
                # Return success with stats
                return WebCrawlBatchResult(
                    title=config.title,
                    pages_crawled=len(crawl_results),
                    urls=[result.url for result in crawl_results],
                    training_id=training_id
                )
                
            except Exception as e:
                logger.error(f"Error processing crawled data: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing crawled data: {str(e)}"
                )
        else:
            logger.warning("No pages were successfully crawled")
            return JSONResponse(
                status_code=400,
                content={"error": "No pages were successfully crawled. Check the URL and crawling patterns."}
            )
    
    except Exception as e:
        logger.error(f"Web crawling error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Web crawling error: {str(e)}"
        )

def extract_main_content(soup):
    """Extract the main content from a BeautifulSoup object."""
    # Remove script and style elements
    for element in soup(['script', 'style']):
        element.extract()
    
    # Try to find main content areas with more potential selectors
    main_content = soup.find('main') or soup.find(id='content') or soup.find(id='topic-content') or \
                  soup.find(class_='content') or soup.find('article') or soup.find(class_='article-content') or \
                  soup.find(class_='portal-body') or soup.find(class_='documentation')
    
    # If we found a main content area, use it
    if main_content:
        # Convert to text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        return h.handle(str(main_content))
    else:
        # Fall back to the body
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        return h.handle(str(soup.body)) if soup.body else ""

def flatten_json(json_data, prefix=''):
    """
    Flatten a nested JSON structure into a flat dictionary.
    """
    flattened = {}
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, (dict, list)):
                flattened.update(flatten_json(value, f"{new_key}_"))
            else:
                flattened[new_key] = value
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            flattened.update(flatten_json(item, f"{prefix}{i}_"))
    else:
        flattened[prefix.rstrip('_')] = json_data
    
    return flattened

def analyze_integration_flow(integration_id, extract_dir):
    """
    Analyze the flow structure of an integration, extracting information about components,
    their relationships, and other metadata that's useful for documentation.
    
    Args:
        integration_id: The UUID of the integration to analyze
        extract_dir: The directory containing the extracted project files
    
    Returns:
        A dictionary containing analysis of the integration flow
    """
    logger.info(f"Analyzing integration flow for: {integration_id}")
    
    # Find the integration JSON file
    integration_files = glob.glob(f"{extract_dir}/**/services/*_{integration_id}.json", recursive=True)
    
    if not integration_files:
        logger.warning(f"Integration file for ID {integration_id} not found")
        return None
    
    integration_file = integration_files[0]
    
    try:
        # Load the integration JSON
        with open(integration_file, 'r') as f:
            integration_data = json.load(f)
        
        # Extract basic information
        flow_info = {
            "id": integration_data.get("id", ""),
            "name": integration_data.get("name", ""),
            "description": integration_data.get("description", ""),
            "components": []
        }
        
        # Find all component configurations that are used in this integration
        component_ids = []
        
        # Extract component IDs from the integration data
        if "components" in integration_data:
            for component in integration_data["components"]:
                component_ids.append(component.get("id", ""))
        
        # Find component configuration files
        components = []
        for component_id in component_ids:
            if not component_id:
                continue
                
            component_files = glob.glob(f"{extract_dir}/**/component_configs/{component_id}.json", recursive=True)
            
            if component_files:
                with open(component_files[0], 'r') as f:
                    component_data = json.load(f)
                    components.append(component_data)
        
        # Process each component to extract useful information
        for component in components:
            component_info = {
                "id": component.get("id", ""),
                "name": component.get("name", ""),
                "description": component.get("description", ""),
                "type": component.get("type", ""),
                "parent_id": component.get("parentId", None),
                "order": None,
                "properties": {}
            }
            
            # Extract component properties
            if "properties" in component:
                for prop_name, prop_value in component["properties"].items():
                    # Only include non-complex properties or those useful for documentation
                    if isinstance(prop_value, (str, int, float, bool)) or prop_name in [
                        "httpMethod", "url", "path", "expression", "inputFormat", "outputFormat",
                        "targetField", "conditions", "mappings", "query", "queryParams"
                    ]:
                        component_info["properties"][prop_name] = prop_value
            
            flow_info["components"].append(component_info)
        
        # Determine component order if possible
        # This is often defined in the integration JSON rather than component configs
        if "flows" in integration_data:
            for flow in integration_data["flows"]:
                if "nodes" in flow:
                    for node in flow["nodes"]:
                        node_id = node.get("componentId", "")
                        node_order = node.get("order", None)
                        
                        # Update the order in our component info
                        for component in flow_info["components"]:
                            if component["id"] == node_id:
                                component["order"] = node_order
                                break
        
        # Extract data mappings between components
        if "mappings" in integration_data:
            flow_info["mappings"] = integration_data["mappings"]
        
        # Look for API definitions if this integration uses API endpoints
        api_ids = set()
        for component in flow_info["components"]:
            if component["type"] in ["apiEventSource", "apiInvoker", "apiEventEmitter"]:
                if "properties" in component and "apiId" in component["properties"]:
                    api_ids.add(component["properties"]["apiId"])
        
        # Get API details if available
        if api_ids:
            flow_info["apis"] = []
            for api_id in api_ids:
                api_files = glob.glob(f"{extract_dir}/**/apis/*_{api_id}.json", recursive=True)
                
                if api_files:
                    with open(api_files[0], 'r') as f:
                        api_data = json.load(f)
                        
                        # Extract key API information
                        api_info = {
                            "id": api_data.get("id", ""),
                            "name": api_data.get("name", ""),
                            "description": api_data.get("description", ""),
                            "endpoints": []
                        }
                        
                        # Extract endpoint information
                        if "paths" in api_data:
                            for path, methods in api_data["paths"].items():
                                for method, details in methods.items():
                                    endpoint_info = {
                                        "path": path,
                                        "method": method.upper(),
                                        "summary": details.get("summary", ""),
                                        "description": details.get("description", "")
                                    }
                                    api_info["endpoints"].append(endpoint_info)
                        
                        flow_info["apis"].append(api_info)
        
        return flow_info
        
    except Exception as e:
        logger.error(f"Error analyzing integration flow: {str(e)}")
        return None

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for analysis.
    Returns the file path for further processing.
    """
    try:
        logger.info(f"Receiving image file: {file.filename}")
        
        # Generate a unique ID for this upload
        upload_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        image_filename = f"{upload_id}{file_extension}"
        image_path = f"/data/uploads/{image_filename}"
        
        # Save the uploaded image file
        content = await file.read()
        with open(image_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Image file saved to {image_path}")
        
        return ImageUploadResponse(
            filename=file.filename,
            file_path=image_path,
            message="Image uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image file: {str(e)}"
        )

@app.post("/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze an integration flow diagram image and generate structured documentation about the integration.
    Uses a multimodal model that can actually process images.
    """
    try:
        logger.info(f"Starting analysis of integration flow diagram: {request.image_path}")
        
        # Check if the image file exists
        if not os.path.exists(request.image_path):
            raise HTTPException(
                status_code=404,
                detail="Image file not found. Please upload the image again."
            )
        
        # Check if Ollama service is available
        try:
            logger.info("Checking if Ollama service is available...")
            response = requests.get("http://ollama:11434", timeout=5)
            if response.status_code >= 400:
                logger.error(f"Ollama service returned status code: {response.status_code}")
                raise HTTPException(
                    status_code=503,
                    detail="Ollama service is not responding correctly. Please check if the service is running properly."
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama service: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama service: {str(e)}. Please ensure the Ollama container is running."
            )
        
        # Check if llava model is available
        try:
            logger.info("Checking if llava model is available...")
            response = requests.get("http://ollama:11434/api/tags", timeout=5)
            models = []
            
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    models = [model.get("name") for model in models_data["models"]]
            
            # Look for llava model
            if "llava" not in models:
                logger.warning("llava model not found. Attempting to pull it...")
                # Try to pull the llava model
                pull_response = requests.post(
                    "http://ollama:11434/api/pull",
                    json={"name": "llava"},
                    timeout=300  # 5 minutes timeout for model download
                )
                
                if pull_response.status_code != 200:
                    logger.error(f"Failed to pull llava model: {pull_response.status_code}")
                    raise HTTPException(
                        status_code=503,
                        detail="Could not find or download llava model. Please run 'docker exec -it ollama ollama pull llava' to install it."
                    )
                logger.info("Successfully pulled llava model")
            else:
                logger.info("llava model is available")
        except Exception as e:
            logger.error(f"Error checking/pulling llava model: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Error setting up llava model: {str(e)}. Try running 'docker exec -it ollama ollama pull llava' manually."
            )
        
        # Load the image
        try:
            image = Image.open(request.image_path)
            
            # Resize if the image is too large - balance between size and quality
            max_size = (800, 800)  # Multimodal models can handle larger images than text-only models
            if image.width > max_size[0] or image.height > max_size[1]:
                image.thumbnail(max_size, Image.LANCZOS)
            
            # Convert image to base64 for model processing
            buffered = BytesIO()
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Use good quality for multimodal models
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info(f"Image processed: {image.width}x{image.height}, base64 length: {len(img_str)}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )
        
        # Initialize LLM with retry logic
        max_retries = 3
        retry_delay = 10  # Seconds between retries
        
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM attempt {attempt + 1}: Setting up llava for flow diagram analysis...")
                
                # Specialized prompt for integration flow diagrams
                prompt = f"""Analyze this integration flow diagram image in detail. 

This is an integration flow diagram showing how data moves between systems. Document it by providing:
1. An overview of the integration's purpose
2. A detailed description of all components visible in the diagram 
3. The data flow between components showing the sequence of operations
4. Any error handling or conditional logic shown

Format your response as structured markdown with headings and bullet points.
"""
                
                # Call Ollama API with increased timeout and optimized configuration
                logger.info("Sending request to Ollama with llava model for flow diagram analysis")
                
                # Use a longer timeout to accommodate processing
                llm_response = requests.post(
                    "http://ollama:11434/api/generate",
                    json={
                        "model": "llava",  # Use llava multimodal model instead of mistral
                        "prompt": prompt,
                        "images": [img_str],  # Pass image separately for multimodal models
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "top_k": 40,
                            "top_p": 0.9,
                            "num_predict": 2048
                        }
                    },
                    timeout=300  # 5 minute timeout for complex diagram analysis
                )
                
                if llm_response.status_code == 200:
                    response_data = llm_response.json()
                    analysis_text = response_data.get("response", "")
                    
                    if not analysis_text:
                        raise Exception("Empty response from language model")
                    
                    # Check for common error responses
                    if "I cannot see any image" in analysis_text or "I don't see any image" in analysis_text:
                        logger.error("llava model couldn't process the image properly")
                        raise Exception("The llava model couldn't process the image properly. It may be corrupted or in an unsupported format.")
                    
                    logger.info("Integration flow documentation generated successfully")
                    
                    # Format and return the result based on requested output format
                    if request.format == OutputFormat.html:
                        # Convert markdown to HTML for display
                        html_doc = markdown.markdown(analysis_text)
                        return {"documentation": html_doc, "format": "html"}
                    
                    # Create file name based on original image
                    base_filename = os.path.basename(request.image_path).rsplit('.', 1)[0]
                    
                    if request.format == OutputFormat.markdown:
                        # Save the markdown content
                        md_filename = f"{base_filename}_integration_doc.md"
                        md_filepath = f"/data/{md_filename}"
                        with open(md_filepath, "w") as f:
                            f.write(f"# Integration Flow Documentation: {base_filename}\n\n{analysis_text}")
                        
                        return {
                            "documentation": analysis_text,
                            "format": "markdown",
                            "filename": md_filename,
                            "download_url": f"/download/{md_filename}"
                        }
                    
                    elif request.format == OutputFormat.pdf:
                        # Convert to PDF
                        pdf_filename = f"{base_filename}_integration_doc.pdf"
                        pdf_filepath = f"/data/{pdf_filename}"
                        
                        # Create HTML content for PDF with side-by-side image and documentation
                        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Integration Flow Documentation: {base_filename}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                                .content {{ margin-top: 20px; }}
                                .image-container {{ text-align: center; margin-bottom: 20px; }}
                                img {{ max-width: 100%; max-height: 500px; border: 1px solid #ddd; }}
                                h2 {{ color: #3498db; margin-top: 20px; }}
                                h3 {{ color: #2980b9; }}
                                .component {{ margin-bottom: 15px; }}
                                .component-details {{ margin-left: 20px; }}
                                pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
                            </style>
                        </head>
                        <body>
                            <h1>Integration Flow Documentation: {base_filename}</h1>
                            
                            <div class="image-container">
                                <img src="data:image/jpeg;base64,{img_str}" alt="Integration Flow Diagram">
                                <p><em>Integration Flow Diagram</em></p>
                            </div>
                            
                            <div class="content">
                                {markdown.markdown(analysis_text)}
                            </div>
                        </body>
                        </html>
                        """
                        
                        try:
                            # Generate PDF file
                            pdfkit.from_string(html_content, pdf_filepath)
                            
                            return {
                                "documentation": markdown.markdown(analysis_text),
                                "format": "pdf",
                                "filename": pdf_filename,
                                "download_url": f"/download/{pdf_filename}"
                            }
                        except Exception as e:
                            logger.error(f"Error generating PDF: {str(e)}")
                            # Fall back to HTML if PDF generation fails
                            return {
                                "documentation": markdown.markdown(analysis_text),
                                "format": "html",
                                "error": f"Failed to generate PDF: {str(e)}"
                            }
                else:
                    logger.error(f"LLM API error: {llm_response.status_code}, {llm_response.text}")
                    raise Exception(f"LLM API returned status code {llm_response.status_code}: {llm_response.text}")
                
            except requests.exceptions.Timeout as e:
                logger.error(f"LLM attempt {attempt + 1} timed out: {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=503,
                        detail="Image analysis is taking longer than expected. This could be due to the complexity of the image or the server load. Please try again later."
                    )
                logger.warning(f"Request timed out. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to analyze integration flow diagram: {str(e)}"
                    )
                logger.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze-image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing integration flow diagram: {str(e)}"
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated file (markdown or PDF)"""
    file_path = f"/data/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path, 
        filename=filename,
        media_type="application/octet-stream"
    )

def is_same_domain(base_url, check_url):
    """
    Check if the check_url is in the same domain as the base_url.
    """
    try:
        base_domain = urlparse(base_url).netloc
        check_domain = urlparse(check_url).netloc
        return base_domain == check_domain
    except:
        return False

def should_crawl_url(url, include_patterns=None, exclude_patterns=None):
    """
    Determine if a URL should be crawled based on include and exclude patterns.
    """
    # Skip media files, backend URLs, etc.
    skip_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz', '.svg', '.mp4', '.mp3', '.mov']
    for ext in skip_extensions:
        if url.lower().endswith(ext):
            return False
    
    # Skip URLs with fragment identifiers (anchors)
    if '#' in url:
        url = url.split('#')[0]
    
    # Apply include patterns if specified
    if include_patterns and len(include_patterns) > 0:
        should_include = False
        for pattern in include_patterns:
            if pattern in url:
                should_include = True
                break
        if not should_include:
            return False
    
    # Apply exclude patterns if specified
    if exclude_patterns and len(exclude_patterns) > 0:
        for pattern in exclude_patterns:
            if pattern in url:
                return False
    
    return True

async def create_vectorstore_from_web_crawl(training_data_path, training_id):
    """
    Create a vectorstore from web crawl data.
    """
    try:
        # Load the training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            base_url="http://ollama:11434",
            model="mistral"
        )
        
        # Create documents from the crawled pages
        documents = []
        
        for page in training_data.get('pages', []):
            title = page.get('title', 'No Title')
            content = page.get('content', '')
            url = page.get('url', '')
            
            if content:
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "url": url,
                        "source": "web_crawl",
                        "training_id": training_id
                    }
                )
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents from web crawl data")
        
        if not documents:
            logger.warning("No documents created from web crawl data")
            return
        
        # Create vectorstore directory
        vectorstore_dir = "/data/training/vectorstore"
        os.makedirs(vectorstore_dir, exist_ok=True)
        
        # Check if existing vectorstore
        if os.path.exists(f"{vectorstore_dir}/chroma.sqlite3"):
            # Load existing vectorstore
            logger.info("Loading existing training vectorstore")
            vectorstore = Chroma(
                persist_directory=vectorstore_dir,
                embedding_function=embeddings,
                collection_name="training_examples"
            )
            # Add documents to existing vectorstore
            vectorstore.add_documents(documents)
        else:
            # Create new vectorstore
            logger.info("Creating new training vectorstore")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=vectorstore_dir,
                collection_name="training_examples"
            )
        
        # Persist changes
        vectorstore.persist()
        logger.info(f"Successfully added web crawl data to training vectorstore")
        
    except Exception as e:
        logger.error(f"Error creating vectorstore from web crawl data: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)