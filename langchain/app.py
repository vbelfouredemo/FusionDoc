#!/usr/bin/env python3

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
import shutil  # Add import for directory cleanup

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

# Get the LLM model from environment variable, or use default model
# This allows switching models easily without changing code
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "mistral")
logger.info(f"Using LLM model: {DEFAULT_LLM_MODEL}")

# Define supported models and their requirements
SUPPORTED_MODELS = {
    "mistral": {
        "description": "Fast medium-sized model with good results",
        "min_ram": "8GB",
        "quality": "Medium"
    },
    "llama3": {
        "description": "High quality, recent model from Meta",
        "min_ram": "16GB",
        "quality": "High"
    },
    "llama3:70b": {
        "description": "Very high quality large model, requires significant resources",
        "min_ram": "32GB+",
        "quality": "Very High"
    },
    "mixtral": {
        "description": "High quality mixture-of-experts model",
        "min_ram": "16GB",
        "quality": "High"
    }
}

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

# Default query template to use if no custom template is provided
DEFAULT_QUERY_TEMPLATE = """Generate detailed documentation for the '{integration_name}' integration.

The integration has the following structure and components:

{flow_structure}

Your task is to write comprehensive documentation that includes:

## 1. INTRODUCTION
Start with a clear, concise overview of what this integration does.

## 2. COMPLETE FLOW DESCRIPTION
* Identify the trigger/source that initiates this flow
* Explain how data flows through EACH component in sequence
* For system steps, clearly explain the logic and contents of each
* Describe what happens at each step, being specific about connections used

## 3. TECHNICAL DETAILS
* Mention specific connection names and plugs if applicable
* Include queue names, endpoints, or database details where available
* Explain what data transformations occur at each step

Format your response as a professional technical document with clear sections, numbered headers (## 1., ## 2., etc.), bullet points (using * for items), and concise language.
IMPORTANT: Be specific and detailed about each component and how they connect - don't just provide a generic overview.
"""

def init_default_query_template():
    """Initialize the default query template file if it doesn't exist."""
    template_dir = "/data/templates"
    os.makedirs(template_dir, exist_ok=True)
    
    default_template_path = os.path.join(template_dir, "default_query_template.txt")
    
    # Only create the file if it doesn't exist
    if not os.path.exists(default_template_path):
        logger.info(f"Creating default query template at {default_template_path}")
        with open(default_template_path, 'w') as f:
            f.write(DEFAULT_QUERY_TEMPLATE)
    else:
        logger.info(f"Default query template already exists at {default_template_path}")

def get_query_template():
    """
    Get the query template to use for LLM prompts.
    Checks for a custom template file first, falls back to the default template if not found.
    """
    template_dir = "/data/templates"
    custom_template_path = os.path.join(template_dir, "custom_query_template.txt")
    default_template_path = os.path.join(template_dir, "default_query_template.txt")
    
    # Use the custom template if it exists
    if os.path.exists(custom_template_path):
        try:
            logger.info(f"Using custom query template from {custom_template_path}")
            with open(custom_template_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading custom template: {str(e)}")
            # Fall back to default template on error
    
    # Use the default template file if it exists
    if os.path.exists(default_template_path):
        try:
            logger.info(f"Using default query template from {default_template_path}")
            with open(default_template_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading default template file: {str(e)}")
            # Fall back to hardcoded template on error
    
    # Fall back to hardcoded constant
    logger.info("Using hardcoded default query template")
    return DEFAULT_QUERY_TEMPLATE

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
                if ensure_model():
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

def ensure_model(model=None):
    """
    Check if requested model is installed in Ollama, and attempt to install it if not present.
    Returns True if the model is available or was successfully installed, False otherwise.
    
    Args:
        model: The model name to ensure is available. If None, uses DEFAULT_LLM_MODEL.
    """
    model_to_check = model or DEFAULT_LLM_MODEL
    logger.info(f"Checking if {model_to_check} model is available in Ollama...")
    
    try:
        # Check available models
        response = requests.get("http://ollama:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            # Check if model is already installed
            if any(m == model_to_check for m in model_names):
                logger.info(f"{model_to_check} model is already installed")
                return True
            
            # If not installed, try to pull it
            logger.info(f"{model_to_check} model not found. Attempting to pull it...")
            
            pull_response = requests.post(
                "http://ollama:11434/api/pull",
                json={"name": model_to_check},
                timeout=600  # Longer timeout for model download
            )
            
            if pull_response.status_code == 200:
                logger.info(f"Successfully pulled {model_to_check} model")
                return True
            else:
                logger.error(f"Failed to pull {model_to_check} model: {pull_response.status_code}, {pull_response.text}")
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
    # Initialize the default query template
    init_default_query_template()
    
    # Retry a few times to accommodate Ollama startup time
    max_retries = 5
    for attempt in range(max_retries):
        logger.info(f"Startup check {attempt+1}/{max_retries}: Verifying Ollama and Mistral availability")
        try:
            # Check if Ollama is up
            response = requests.get("http://ollama:11434", timeout=5)
            if response.status_code < 400:
                # Ollama is up, now ensure Mistral is installed
                if ensure_model():
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
                
                # Extract integration name from the data if available, or from filename
                integration_name = json_data.get("name", os.path.basename(filename).split('.')[0])
                logger.info(f"Integration name: {integration_name}")
                
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
                    model=DEFAULT_LLM_MODEL  # Use the configurable model instead of hard-coded "mistral"
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
                    model=DEFAULT_LLM_MODEL  # Use the configurable model instead of hard-coded "mistral"
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )
                
                # Generate a basic flow structure if none is available
                flow_structure = "No detailed structure information available for this integration."
                
                # Try to extract structure information from the JSON data
                if "components" in json_data:
                    components = json_data.get("components", [])
                    component_details = []
                    
                    for component in components:
                        component_name = component.get("name", "Unnamed Component")
                        component_type = component.get("type", "Unknown")
                        component_desc = component.get("description", "")
                        
                        component_details.append(f"- Component: {component_name} (Type: {component_type})")
                        if component_desc:
                            component_details.append(f"  Description: {component_desc}")
                    
                    if component_details:
                        flow_structure = "Components:\n" + "\n".join(component_details)
                
                # Generate documentation with a specific prompt for this integration
                # Use the custom query template or default if not available
                query_template = get_query_template()
                query = query_template.format(
                    integration_name=integration_name,
                    flow_structure=flow_structure
                )
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

@app.post("/analyze-hybrid")
async def analyze_file_with_hybrid_approach(
    filename: str = Form(...), 
    output_format: OutputFormat = Form(OutputFormat.html),
    integrations: Optional[str] = Form(None),
    use_realtime_lookup: bool = Form(True)
):
    """Generate documentation using both pre-trained data and real-time lookups for improved accuracy"""
    try:
        logger.info(f"Starting hybrid analysis of file: {filename} with output format: {output_format}")
        logger.info(f"Real-time lookups enabled: {use_realtime_lookup}")

        # Initialize the core components as in the regular /analyze-smart endpoint
        # ...existing code...

        # Check if this is a ZIP file analysis with selected integrations
        if integrations:
            selected_integrations = json.loads(integrations)
            
            # ...existing code for extraction, validation, etc...
            
            # Initialize a hybrid documentation generator if requested
            hybrid_generator = None
            if use_realtime_lookup:
                from hybrid_approach import HybridDocumentationGenerator
                hybrid_generator = HybridDocumentationGenerator(model_name=DEFAULT_LLM_MODEL)
            
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
                    with open(file_path, "r") as f:
                        json_data = json.load(f)
                    
                    # Choose between hybrid approach or standard approach
                    if use_realtime_lookup and hybrid_generator:
                        # Use the hybrid approach with real-time lookups
                        documentation_result = await hybrid_generator.generate_documentation(
                            integration_json=json_data,
                            integration_name=integration_name,
                            flow_analysis=flow_analysis,
                            component_names=[comp.get("name") for comp in flow_analysis.get("components", [])
                                           if comp.get("name") and "Unnamed" not in comp.get("name")]
                        )
                        
                        # Extract the documentation and metadata
                        documentation = documentation_result["documentation"]
                        sources = documentation_result.get("sources", [])
                        lookup_quality = documentation_result.get("lookup_quality", "low")
                        
                        all_docs.append({
                            "name": integration_name,
                            "documentation": documentation,
                            "sources": sources,
                            "lookup_quality": lookup_quality,
                            "realtime_enhanced": True
                        })
                    else:
                        # Use the standard approach without real-time lookups
                        # ... existing code for standard processing ...
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing integration {integration_name}: {str(e)}")
                    continue
            
            # Combine all documentation into a single document
            # ... existing code ...
            
            # Format and return the results based on requested format
            # ... existing code ...
            
    except Exception as e:
        logger.error(f"Error in analyze-hybrid: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

@app.post("/analyze-smart")
async def analyze_file_with_training(
    filename: str = Form(...), 
    output_format: OutputFormat = Form(OutputFormat.html),
    integrations: Optional[str] = Form(None),
    verbose: Optional[bool] = Form(False)
):
    """Generate documentation using training data for improved accuracy.
    
    When verbose is True, additional information about the training examples used and
    confidence scores will be included in the documentation.
    """
    try:
        logger.info(f"Starting smart analysis of file: {filename} with output format: {output_format}, verbose mode: {verbose}")

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
                model=DEFAULT_LLM_MODEL  # Use the configurable model instead of hard-coded "mistral"
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
                    with open(file_path, "r") as f:
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
                    connection_details = []
                    execution_order = []
                    
                    if flow_analysis and "components" in flow_analysis:
                        # Sort components by order
                        components = sorted(
                            flow_analysis["components"], 
                            key=lambda x: float(x["order"]) if x["order"] is not None else float('inf')
                        )
                        
                        # First, gather all connection details for easier reference
                        for component in components:
                            if component.get("connection_name"):
                                connection_details.append(f"{component['name']} uses the connection \"{component.get('connection_name')}\"")
                                
                                # Add queue name for RabbitMQ/AMQP components
                                if component.get("queue_name"):
                                    connection_details[-1] += f" and listens on queue \"{component.get('queue_name')}\""
                        
                        # First, identify parent components (like fork-joins)
                        parent_components = {}
                        for component in components:
                            if component.get("sub_components") and len(component.get("sub_components", [])) > 0:
                                parent_components[component["id"]] = {
                                    "name": component["name"],
                                    "type": component["type"],
                                    "children": component.get("sub_components", [])
                                }
                        
                        # Now build structured description
                        for component in components:
                            # Basic component info
                            component_desc = f"- **{component['name']}**"
                            
                            # Add type info
                            if component["type"] and component["type"] != "Unknown":
                                component_desc += f" (Type: {component['type']})"
                            
                            # Add connection info if available
                            if component.get("connection_name"):
                                component_desc += f"\n  * Uses connection: **{component.get('connection_name')}**"
                                
                                # Add queue name for RabbitMQ/AMQP components
                                if component.get("queue_name"):
                                    component_desc += f"\n  * Queue name: **{component.get('queue_name')}**"
                                
                                # Add endpoint info for HTTP components
                                if component.get("endpoint_url"):
                                    component_desc += f"\n  * Endpoint: {component.get('endpoint_url')}"
                                    if component.get("http_method"):
                                        component_desc += f" ({component.get('http_method')})"
                                
                                # Add database info for DB components
                                if component.get("database_name"):
                                    component_desc += f"\n  * Database: {component.get('database_name')}"
                                    if component.get("collection_name"):
                                        component_desc += f", Collection: {component.get('collection_name')}"
                            
                            # Add parent info
                            if component.get("parent_id"):
                                parent_name = next(
                                    (c["name"] for c in components if c["id"] == component["parent_id"]), 
                                    component["parent_id"]
                                )
                                component_desc += f"\n  * Part of: **{parent_name}**"
                                
                                # If parent is a fork-join, indicate which branch
                                if component["parent_id"] in parent_components:
                                    # Find the index of this component in the parent's children
                                    parent = parent_components[component["parent_id"]]
                                    if "fork" in parent["type"].lower() or "join" in parent["type"].lower() or "branch" in parent["type"].lower():
                                        component_desc += f" (Parallel Branch)"
                            
                            # Add execution order info
                            if "execution_order" in component:
                                execution_step = f"Step {component['execution_order'] + 1}: {component['name']}"
                                execution_order.append(execution_step)
                            
                            # Add description if available
                            if component["description"]:
                                component_desc += f"\n  * Description: {component['description']}"
                            
                            component_descriptions.append(component_desc)
                        
                    # Generate connection relationships
                    connection_flow = []
                    if flow_analysis and "connections" in flow_analysis:
                        for connection in flow_analysis["connections"]:
                            connection_flow.append(f"- **{connection['from_name']}** → **{connection['to_name']}**")
                    
                    # Generate a more complete flow structure
                    flow_structure = "### Components\n\n" + "\n\n".join(component_descriptions) if component_descriptions else "No component data available"
                    
                    if connection_details:
                        flow_structure += "\n\n### Connections\n\n" + "\n".join(connection_details)
                    
                    if connection_flow:
                        flow_structure += "\n\n### Data Flow\n\n" + "\n".join(connection_flow)
                    
                    if execution_order:
                        flow_structure += "\n\n### Execution Sequence\n\n" + "\n".join(execution_order)
                    
                    # Generate documentation with a specific prompt for this integration
                    # Use the custom query template or default if not available
                    query_template = get_query_template()
                    query = query_template.format(
                        integration_name=integration_name,
                        flow_structure=flow_structure
                    )
                    
                    # If verbose mode is enabled, get similar training examples
                    training_examples = []
                    similar_scores = []
                    if verbose and training_vectorstore:
                        try:
                            logger.info(f"Verbose mode enabled, retrieving similar training examples")
                            
                            # Create a JSON representation of the current integration
                            integration_json = json.dumps(flow_json)
                            
                            # Retrieve similar documents from training data
                            similar_docs = training_vectorstore.similarity_search_with_score(
                                integration_json,
                                k=3  # Get top 3 similar examples
                            )
                            
                            # Process the similar documents
                            for doc, score in similar_docs:
                                # The score from Chroma is a distance, lower is better
                                # Convert to similarity score (0-100%)
                                # Adjust formula to better handle Chroma's distance metric
                                adjusted_score = max(0, 1.0 - score)  # First invert so higher is better
                                similarity_score = int(adjusted_score * 100)  # Convert to percentage
                                
                                # Boost similarity scores to be more intuitive for users
                                # Map the range of typical scores to a more useful range
                                if similarity_score > 0:
                                    # Apply a logarithmic scaling to better differentiate similar examples
                                    # This makes small differences more visible in the UI
                                    # Boost low scores to ensure visibility
                                    similarity_score = max(40, min(95, int(similarity_score * 2.0)))
                                
                                # Get the expected documentation from metadata
                                expected_doc = doc.metadata.get("expected_documentation", "No documentation available")

                                # Add highlighting to the preview text regardless of similarity score
                                # This ensures we always see highlighted content for debugging
                                doc_preview = expected_doc[:200] + "..." if len(expected_doc) > 200 else expected_doc
                                
                                # Always apply highlighting for better visibility during debugging
                                doc_preview = f'<mark class="highlight-similar">{doc_preview}</mark>'
                                
                                # Get title/name if available
                                title = doc.metadata.get("title", "Unknown Example")
                                
                                # Log the details of this match for debugging
                                logger.info(f"Training match: {title}, Original score: {score}, Adjusted: {similarity_score}%")
                                logger.info(f"Preview content: {doc_preview[:50]}...")
                                
                                training_examples.append({
                                    "title": title,
                                    "preview": doc_preview,
                                    "similarity": similarity_score
                                })
                                similar_scores.append(similarity_score)
                            
                            logger.info(f"Found {len(training_examples)} similar training examples with scores: {similar_scores}")
                        except Exception as e:
                            logger.error(f"Error retrieving similar training examples: {str(e)}")
                            logger.error(traceback.format_exc())
                    
                    # Call the LLM for documentation generation
                    result = qa_chain.invoke(query)["result"]
                    
                    # Add to the documentation collection
                    doc_info = {
                        "name": integration_name,
                        "documentation": result,
                        "flow_structure": flow_structure
                    }
                    
                    # If verbose mode is on and we have training examples, add them
                    if verbose and training_examples:
                        doc_info["training_examples"] = training_examples
                        
                        # Calculate overall confidence based on similarity scores
                        if similar_scores:
                            # Calculate a weighted average confidence score
                            weighted_sum = 0
                            weights = 0
                            
                            # Give higher weight to higher similarity scores
                            for i, score in enumerate(similar_scores):
                                weight = 1.0 if i == 0 else 0.5  # First example has full weight
                                weighted_sum += score * weight
                                weights += weight
                            
                            # Calculate confidence as weighted average adjusted by number of examples
                            avg_similarity = weighted_sum / weights if weights > 0 else 0
                            confidence_boost = min(1.0, len(similar_scores) / 3.0)  # More examples = higher confidence
                            confidence = avg_similarity * confidence_boost
                            
                            # Ensure confidence is between 0-100 and is an integer
                            doc_info["confidence"] = max(1, min(100, int(confidence)))
                            logger.info(f"Calculated confidence score: {doc_info['confidence']}% from {len(similar_scores)} examples")
                        else:
                            doc_info["confidence"] = 0
                            logger.info("No similar examples found, confidence score is 0%")
                    
                    all_docs.append(doc_info)
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
                return {
                    "documentation": html_doc, 
                    "format": "html",
                    "all_docs": all_docs if verbose else None  # Include all_docs when in verbose mode
                }
            
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
                    "download_url": f"/download/{md_filename}",
                    "all_docs": all_docs if verbose else None  # Include all_docs when in verbose mode
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
                        "download_url": f"/download/{pdf_filename}",
                        "all_docs": all_docs if verbose else None  # Include all_docs when in verbose mode
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
                                doc_preview = data["documentation"][:100] +"..." if len(data["documentation"]) > 100 else data["documentation"]
                        
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

@app.post("/train-with-zip")
async def train_with_zip(
    filename: str = Form(...),
    integration_id: str = Form(...),
    documentation: str = Form(...)
):
    """
    Use a single integration from an uploaded ZIP file as training data.
    Creates a structured training example that can be used to improve documentation generation.
    """
    try:
        logger.info(f"Processing training with integration ID: {integration_id} from ZIP file: {filename}")
        
        # Extract
        upload_id = filename.split('_')[0]
        extract_dir = f"/data/extracted/{upload_id}"
        
        if not os.path.exists(extract_dir):
            logger.error(f"Extracted directory not found: {extract_dir}")
            return JSONResponse(
                status_code=404,
                content={"error": "Extracted archive not found. Please upload the file again."}
            )
        
        # Find the integration file
        integration_files = glob.glob(f"{extract_dir}/**/services/*_{integration_id}.json", recursive=True)
        
        if not integration_files:
            logger.error(f"Integration file not found for ID: {integration_id}")
            return JSONResponse(
                status_code=404,
                content={"error": "Integration file not found in the archive. Please upload again."}
            )
        
        integration_file = integration_files[0]
        logger.info(f"Found integration file: {integration_file}")
        
        # Load the integration JSON
        with open(integration_file, 'r') as f:
            integration_json = json.load(f)
        
        # Extract integration name
        integration_name = integration_json.get("name", f"Integration_{integration_id}")
        
        # IMPROVED: Analyze the integration flow to get comprehensive information
        # This matches what the smart documentation does, making the training example more complete
        flow_analysis = analyze_integration_flow(integration_id, extract_dir)
        
        # IMPROVED: Create an enriched integration JSON that includes flow analysis
        # This better matches what will be used during smart documentation generation
        enriched_integration_json = {
            "basic_data": integration_json,
            "flow_analysis": flow_analysis
        }
        
        # Create a training example with the enriched JSON
        training_example = {
            "integration_json": enriched_integration_json,
            "documentation": documentation
        }
        
        # Save to JSON file in langchain directory with proper naming
        os.makedirs("/data/langchain", exist_ok=True)
        training_file = f"/data/langchain/{integration_name}_{integration_id}.json"
        
        with open(training_file, "w") as f:
            json.dump(training_example, f, indent=2)
        
        logger.info(f"Saved training example to {training_file}")
        
        # Also save markdown version for easy reference
        markdown_file = f"/data/langchain/{integration_name}_{integration_id}.md"
        with open(markdown_file, "w") as f:
            f.write(f"# {integration_name} Documentation\n\n")
            f.write(documentation)
        
        logger.info(f"Saved training markdown to {markdown_file}")
        
        # Add to vector store for training
        try:
            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                base_url="http://ollama:11434",
                model="mistral"
            )
            
            # Create document with metadata
            # IMPROVED: Use the same enriched JSON representation for vector embedding
            # that will be used during smart documentation
            doc = Document(
                page_content=json.dumps(enriched_integration_json),
                metadata={
                    "expected_documentation": documentation,
                    "integration_name": integration_name,
                    "integration_id": integration_id,
                    "title": f"{integration_name}_{integration_id}"
                }
            )
            
            # Create or update vectorstore
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
                # Add document to existing vectorstore
                vectorstore.add_documents([doc])
            else:
                # Create new vectorstore
                logger.info("Creating new training vectorstore")
                vectorstore = Chroma.from_documents(
                    documents=[doc],
                    embedding=embeddings,
                    persist_directory=vectorstore_dir,
                    collection_name="training_examples"
                )
            
            # Persist changes
            vectorstore.persist()
            logger.info("Added training example to vectorstore")
            
        except Exception as e:
            logger.error(f"Error adding to vectorstore: {str(e)}")
            # Continue anyway, as we've saved the files
        
        return {
            "message": f"Successfully added training example for {integration_name}",
            "integration_name": integration_name,
            "integration_id": integration_id,
            "training_file": os.path.basename(training_file)
        }
        
    except Exception as e:
        logger.error(f"Error in train-with-zip: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing training data: {str(e)}"}
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
                                h2 {{ color: #3498db; margin-top: 30px; }}
                                h3 {{ color: #2980b9; }}
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
                                    logger.warning(f"Error extracting href: {str(e)}")
                            
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
                
                # IMPROVED: Use the same enriched training approach as in train-with-zip
                # Create structured training data
                for i, page in enumerate(training_data["pages"]):
                    # Create a meaningful ID for this training example
                    page_id = str(uuid.uuid4())
                    page_title = page["title"].replace(" ", "_")[:30]
                    
                    # Create enriched training data with the same structure as integration JSON
                    # This ensures consistency with our training data format
                    enriched_training_data = {
                        "basic_data": {
                            "title": page["title"],
                            "url": page["url"],
                            "source": "web_crawl",
                            "training_id": training_id,
                            "type": "documentation"
                        },
                        "content_analysis": {
                            "text_content": page["content"],
                            "word_count": len(page["content"].split()),
                            "crawled_at": training_data["crawled_at"]
                        }
                    }
                    
                    # Create a training example in the standard format
                    page_training_example = {
                        "integration_json": enriched_training_data,
                        "documentation": page["content"]
                    }
                    
                    # Save to JSON file in langchain directory with proper naming
                    os.makedirs("/data/langchain", exist_ok=True)
                    page_filename = f"WebDoc_{page_title}_{page_id}.json"
                    page_filepath = f"/data/langchain/{page_filename}"
                    
                    with open(page_filepath, "w") as f:
                        json.dump(page_training_example, f, indent=2)
                    
                    logger.info(f"Saved web training example to {page_filepath}")
                
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

def analyze_integration_flow(integration_id, extract_dir):
    """
    Analyze an integration flow to extract structure and component information.
    
    This function examines the integration file and related files in the extract directory
    to build a comprehensive picture of the integration flow, including components, connections,
    and their relationships.
    
    Args:
        integration_id: The UUID of the integration to analyze
        extract_dir: Path to the directory where the ZIP was extracted
        
    Returns:
        A dictionary containing flow analysis information, including components, connections, etc.
    """
    logger.info(f"Analyzing integration flow structure for integration ID: {integration_id}")
    
    try:
        # Step 1: Find the main integration file
        integration_files = glob.glob(f"{extract_dir}/**/services/*_{integration_id}.json", recursive=True)
        
        if not integration_files:
            logger.warning(f"Integration JSON file not found for ID: {integration_id}")
            return {"components": [], "connections": [], "properties": {}, "error": "Integration file not found"}
        
        integration_file = integration_files[0]
        logger.info(f"Found integration file: {integration_file}")
        
        # Step 2: Load the root integration JSON
        with open(integration_file, 'r') as f:
            integration_json = json.load(f)
        
        # Extract basic integration properties
        integration_name = integration_json.get("name", f"Integration_{integration_id}")
        integration_type = integration_json.get("type", "UNKNOWN")
        integration_description = integration_json.get("description", "")
        
        # Initialize flow analysis data structure
        flow_analysis = {
            "name": integration_name,
            "id": integration_id,
            "type": integration_type,
            "description": integration_description,
            "components": [],
            "connections": [],
            "properties": {},
            "connection_details": []
        }
        
        # Step 3: Find the project directory by searching upward from the integration file
        integration_dir = os.path.dirname(integration_file)
        project_root = None
        
        # Search upward to find the project root (should contain components, connections directories)
        current_dir = integration_dir
        while current_dir != extract_dir:
            if os.path.exists(os.path.join(current_dir, "components")) or os.path.exists(os.path.join(current_dir, "connections")):
                project_root = current_dir
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached the filesystem root
                break
            current_dir = parent_dir
        
        if not project_root:
            logger.warning(f"Could not find project root for integration {integration_id}")
            project_root = extract_dir  # Fallback to extract directory
        
        logger.info(f"Using project root: {project_root}")
        
        # Step 4: Find and process components directory
        components_dir = os.path.join(project_root, "components")
        components_files = []
        
        if os.path.exists(components_dir):
            # Collect all component files
            components_files = glob.glob(f"{components_dir}/**/*.json", recursive=True)
            logger.info(f"Found {len(components_files)} component files in {components_dir}")
        
        # Step 5: Find and process connections directory
        connections_dir = os.path.join(project_root, "connections")
        connections_files = []
        
        if os.path.exists(connections_dir):
            # Collect all connection files
            connections_files = glob.glob(f"{connections_dir}/**/*.json", recursive=True)
            logger.info(f"Found {len(connections_files)} connection files in {connections_dir}")
        
        # Map to store connection details by ID for easy lookup
        connection_map = {}
        
        # Process connection files to extract connection details
        for conn_file in connections_files:
            try:
                with open(conn_file, 'r') as f:
                    conn_data = json.load(f)
                    
                    # Extract connection ID and details
                    conn_id = conn_data.get("id")
                    if conn_id:
                        connection_map[conn_id] = {
                            "id": conn_id,
                            "name": conn_data.get("name", "Unknown Connection"),
                            "type": conn_data.get("type", "Unknown"),
                            "properties": conn_data.get("properties", {}),
                            "file_path": conn_file
                        }
                        
                        # Add to flow analysis connection details
                        flow_analysis["connection_details"].append(connection_map[conn_id])
            except Exception as e:
                logger.error(f"Error processing connection file {conn_file}: {str(e)}")
        
        # Step 6: Extract components from the integration JSON
        if "components" in integration_json:
            component_list = integration_json.get("components", [])
            
            # Process integration components
            for component in component_list:
                component_id = component.get("id", "")
                component_name = component.get("name", "Unnamed Component")
                component_type = component.get("type", "Unknown")
                component_description = component.get("description", "")
                component_order = component.get("order", None)
                component_parent_id = component.get("parentId", None)
                
                # Enhanced component data structure
                component_data = {
                    "id": component_id,
                    "name": component_name,
                    "type": component_type,
                    "description": component_description,
                    "order": component_order,
                    "parent_id": component_parent_id,
                    "properties": component.get("properties", {}),
                    "connection_id": component.get("connectionId", None),
                    "connection_name": None,
                    "sub_components": []
                }
                
                # If component has a connection, look it up for more details
                if component_data["connection_id"] and component_data["connection_id"] in connection_map:
                    conn = connection_map[component_data["connection_id"]]
                    component_data["connection_name"] = conn["name"]
                    component_data["connection_type"] = conn["type"]
                    component_data["connection_properties"] = conn["properties"]
                
                # Step 7: If this component has a detailed component file, load additional details
                component_file = None
                for cf in components_files:
                    if f"/{component_id}.json" in cf or f"\\{component_id}.json" in cf:
                        component_file = cf
                        break
                
                if component_file:
                    try:
                        with open(component_file, 'r') as f:
                            component_details = json.load(f)
                            
                            # Enhance component with additional details
                            if "properties" in component_details:
                                # Merge properties, giving precedence to detailed file
                                detailed_props = component_details["properties"]
                                if isinstance(detailed_props, dict):
                                    component_data["properties"].update(detailed_props)
                                
                            # Check for specific component types and extract relevant details
                            if component_type == "AMQP" or "RabbitMQ" in component_type or "Rabbit" in component_name:
                                # Extract RabbitMQ specific details
                                if "queue" in component_data["properties"]:
                                    component_data["queue_name"] = component_data["properties"]["queue"]
                                elif "queueName" in component_data["properties"]:
                                    component_data["queue_name"] = component_data["properties"]["queueName"]
                                
                            elif component_type == "HTTP" or component_type == "HTTPS" or "HTTP" in component_name:
                                # Extract HTTP client specific details
                                if "url" in component_data["properties"]:
                                    component_data["endpoint_url"] = component_data["properties"]["url"]
                                if "method" in component_data["properties"]:
                                    component_data["http_method"] = component_data["properties"]["method"]
                                
                            elif component_type == "MONGODB" or "Mongo" in component_name or "DB" in component_name:
                                # Extract MongoDB specific details
                                if "collection" in component_data["properties"]:
                                    component_data["collection_name"] = component_data["properties"]["collection"]
                                if "database" in component_data["properties"]:
                                    component_data["database_name"] = component_data["properties"]["database"]
                                if "operation" in component_data["properties"]:
                                    component_data["operation"] = component_data["properties"]["operation"]
                    except Exception as e:
                        logger.error(f"Error processing component details file {component_file}: {str(e)}")
                
                # Add processed component to the flow analysis
                flow_analysis["components"].append(component_data)
        
        # Step 8: Extract connections between components
        if "connections" in integration_json:
            connections = integration_json.get("connections", [])
            
            # Process each connection
            for connection in connections:
                from_id = connection.get("fromId", "")
                to_id = connection.get("toId", "")
                
                # Find component names for better readability
                from_name = next((c["name"] for c in flow_analysis["components"] if c["id"] == from_id), from_id)
                to_name = next((c["name"] for c in flow_analysis["components"] if c["id"] == to_id), to_id)
                
                # Structure the connection information
                connection_data = {
                    "from_id": from_id,
                    "to_id": to_id,
                    "from_name": from_name,
                    "to_name": to_name,
                    "properties": connection.get("properties", {})
                }
                
                # Add connection to the flow analysis
                flow_analysis["connections"].append(connection_data)
        
        # Step 9: Organize components into a tree structure for fork-join patterns
        component_map = {comp["id"]: comp for comp in flow_analysis["components"]}
        
        # Identify parent-child relationships
        for comp in flow_analysis["components"]:
            if comp["parent_id"] and comp["parent_id"] in component_map:
                parent = component_map[comp["parent_id"]]
                if "sub_components" not in parent:
                    parent["sub_components"] = []
                parent["sub_components"].append(comp["id"])
        
        # Step 10: Extract any global properties
        if "properties" in integration_json:
            flow_analysis["properties"] = integration_json["properties"]
        
        # Step 11: Determine execution sequence by following connections
        # Start with components that have no incoming connections (triggers/sources)
        incoming_connections = {conn["to_id"]: True for conn in flow_analysis["connections"]}
        source_components = [comp for comp in flow_analysis["components"] 
        if comp["id"] not in incoming_connections]
        
        # Set execution order for source components
        for i, comp in enumerate(source_components):
            comp["execution_order"] = i
        
        # Then follow connections to determine sequence
        sequence_order = len(source_components)
        processed = {comp["id"]: True for comp in source_components}
        
        # Keep processing until all components have been assigned an execution order
        while len(processed) < len(flow_analysis["components"]):
            progress_made = False
            
            for conn in flow_analysis["connections"]:
                from_id = conn["from_id"]
                to_id = conn["to_id"]
                
                # If source is processed but target is not
                if from_id in processed and to_id not in processed and to_id in component_map:
                    component_map[to_id]["execution_order"] = sequence_order
                    processed[to_id] = True
                    sequence_order += 1
                    progress_made = True
            
            # If no progress was made but we haven't processed all components,
            # there might be a cycle or disconnected components
            if not progress_made:
                # Assign arbitrary order to remaining components
                for comp in flow_analysis["components"]:
                    if comp["id"] not in processed:
                        comp["execution_order"] = sequence_order
                        processed[comp["id"]] = True
                        sequence_order += 1
        
        # Sort components by execution order for easier understanding
        flow_analysis["components"].sort(key=lambda x: x.get("execution_order", float('inf')))
        
        logger.info(f"Successfully analyzed integration flow with {len(flow_analysis['components'])} components and {len(flow_analysis['connections'])} connections")
        return flow_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing integration flow: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a minimal structure to avoid breaking code that expects this structure
        return {
            "components": [], 
            "connections": [], 
            "properties": {}, 
            "error": f"Error analyzing integration: {str(e)}"
        }

def flatten_json(json_obj, parent_key='', separator='_'):
    """
    Flattens a nested JSON object into a flat dictionary with compound keys.
    This makes it easier to process complex nested structures.
    
    Args:
        json_obj: The JSON object to flatten
        parent_key: The parent key prefix (used for recursion)
        separator: The separator to use between nested keys
        
    Returns:
        A flattened dictionary
    """
    items = {}
    
    # Handle the case where json_obj is None
    if json_obj is None:
        return {}
    
    # Handle the case where json_obj is a list
    if isinstance(json_obj, list):
        # If the list is empty, just return an empty dict
        if not json_obj:
            return {}
        
        # For non-empty lists, create entries for each item
        for i, item in enumerate(json_obj):
            new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
            
            # Recursively flatten the item if it's a dict or list
            if isinstance(item, (dict, list)):
                items.update(flatten_json(item, new_key, separator))
            else:
                items[new_key] = item
        return items
    
    # Handle the case where json_obj is a dict
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            # Recursively flatten the value if it's a dict or list
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, separator))
            else:
                items[new_key] = value
        return items
    
    # If json_obj is neither a dict nor list, just return it as a single item
    items[parent_key] = json_obj
    return items

@app.post("/cleanup-vectorstores")
async def cleanup_vectorstores():
    """
    Clean up old temporary vector stores to free up disk space.
    This endpoint removes temporary vector store directories created during document generation.
    """
    try:
        logger.info("Starting cleanup of temporary vector stores")
        
        # Get the base data directory
        data_dir = "/data"
        temp_pattern = "temp_vectorstore_*"
        
        # Find all temporary vector stores
        temp_dirs = glob.glob(f"{data_dir}/{temp_pattern}")
        
        # Sort by modification time (oldest first)
        temp_dirs.sort(key=lambda x: os.path.getmtime(x))
        
        if not temp_dirs:
            logger.info("No temporary vector stores found to clean up")
            return {"message": "No temporary vector stores found to clean up", "cleaned": 0}
        
        # Count of successfully removed directories
        removed_count = 0
        failed_dirs = []
        
        for temp_dir in temp_dirs:
            try:
                # Check if it's a directory and has the expected format
                if os.path.isdir(temp_dir) and "temp_vectorstore_" in temp_dir:
                    # Get the modification time for logging
                    mod_time = os.path.getmtime(temp_dir)
                    mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                    
                    logger.info(f"Removing temporary vector store: {temp_dir} (modified: {mod_time_str})")
                    
                    # Recursively remove the directory and all contents
                    shutil.rmtree(temp_dir)
                    removed_count += 1
            except Exception as e:
                logger.error(f"Error removing directory {temp_dir}: {str(e)}")
                failed_dirs.append({"dir": temp_dir, "error": str(e)})
        
        logger.info(f"Cleanup completed. Removed {removed_count} temporary vector stores")
        
        return {
            "message": f"Successfully cleaned up {removed_count} temporary vector stores",
            "cleaned": removed_count,
            "failed": failed_dirs
        }
        
    except Exception as e:
        logger.error(f"Error during vector store cleanup: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error during vector store cleanup: {str(e)}"}
        )

@app.get("/models")
async def get_available_models():
    """Get information about available LLM models in Ollama"""
    try:
        # Get currently configured model
        current_model = os.environ.get("LLM_MODEL", "mistral")
        
        # Check which models are actually available in Ollama
        available_models = []
        installed_models = []
        
        try:
            # Fetch installed models from Ollama
            response = requests.get("http://ollama:11434/api/tags", timeout=10)
            if response.status_code == 200:
                # Normalize model names by stripping ":latest"
                installed_models = [
                    model.get("name", "").split(":")[0] for model in response.json().get("models", [])
                ]
            else:
                logger.error(f"Failed to fetch installed models: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error fetching installed models: {str(e)}")
        
        # Combine supported models info with installation status
        for model_name, model_info in SUPPORTED_MODELS.items():
            is_installed = model_name in installed_models
            available_models.append({
                "name": model_name,
                "description": model_info["description"],
                "min_ram": model_info["min_ram"],
                "quality": model_info["quality"],
                "installed": is_installed,
                "current": model_name == current_model
            })
        
        return {
            "current_model": current_model,
            "available_models": available_models
        }
    except Exception as e:
        logger.error(f"Error getting model information: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting model information: {str(e)}"}
        )

@app.post("/models/set")
async def set_active_model(model_name: str = Form(...)):
    """Set the active LLM model to use for generation"""
    try:
        if model_name not in SUPPORTED_MODELS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported model: {model_name}. Available models are: {list(SUPPORTED_MODELS.keys())}"}
            )
        
        # Check if model is installed, if not try to install it
        if not ensure_model(model_name):
            return JSONResponse(
                status_code=503,
                content={"error": f"Failed to ensure {model_name} model is available. Please install it manually with 'docker exec -it ollama ollama pull {model_name}'"}
            )
        
        # Set environment variable
        os.environ["LLM_MODEL"] = model_name
        logger.info(f"Set active model to: {model_name}")
        
        # Additional model checks with increased timeout
        try:
            logger.info(f"Testing model {model_name} with increased timeout...")
            response = requests.post(
                "http://ollama:11434/api/generate",
                json={"model": model_name, "prompt": "Hello", "stream": False},
                timeout=60  # Increased timeout to 60 seconds for larger models
            )
            
            if response.status_code != 200:
                logger.warning(f"Model test returned status code {response.status_code}, but continuing anyway")
                # Don't fail here, just log the warning
            else:
                logger.info(f"Model {model_name} test successful")
        except requests.exceptions.Timeout:
            # Don't fail on timeout, just log it
            logger.warning(f"Timeout occurred while testing model {model_name}, but continuing anyway")
        except Exception as e:
            # Log other errors but don't fail
            logger.warning(f"Error testing model {model_name}: {str(e)}, but continuing anyway")
        
        # Return success even if the test failed - the model may just take longer to initialize
        return {"message": f"Successfully set active model to {model_name}", "model": model_name}
    except Exception as e:
        logger.error(f"Error setting active model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error setting active model: {str(e)}"}
        )

@app.get("/template")
async def get_current_template():
    """
    Retrieve the current query template.
    Checks for a custom template file first, falls back to the default template if not found.
    """
    try:
        template = get_query_template()
        return {"template": template}
    except Exception as e:
        logger.error(f"Error retrieving query template: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving query template: {str(e)}"}
        )

@app.post("/template")
async def update_template(new_template: str = Form(...)):
    """
    Update the query template.
    Saves the new template to the custom template file, ensuring it persists across container restarts.
    """
    try:
        template_dir = "/data/templates"
        custom_template_path = os.path.join(template_dir, "custom_query_template.txt")
        
        # Ensure the template directory exists
        os.makedirs(template_dir, exist_ok=True)
        
        # Save the new template
        with open(custom_template_path, "w") as f:
            f.write(new_template)
        
        logger.info("Custom query template updated successfully")
        return {"message": "Query template updated successfully"}
    except Exception as e:
        logger.error(f"Error updating query template: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error updating query template: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)