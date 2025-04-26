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
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import zipfile
import re
import uuid
import shutil
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionDoc")

def ensure_mistral_model():
    """Check if Mistral model is installed, and install it if not."""
    logger.info("Checking if Mistral model is available...")
    try:
        response = requests.post(
            "http://ollama:11434/api/generate",
            json={"model": "mistral", "prompt": "test", "stream": False},
            timeout=15  # Increased timeout from 5 to 15 seconds
        )
        if response.status_code == 200:
            logger.info("Mistral model is already installed")
            return True
        
        if response.status_code == 404:
            logger.info("Mistral model not found. Installing now...")
            pull_response = requests.post(
                "http://ollama:11434/api/pull",
                json={"name": "mistral"},
                timeout=600  # Longer timeout for model download
            )
            if pull_response.status_code == 200:
                logger.info("Successfully installed Mistral model")
                return True
            else:
                logger.error(f"Failed to install Mistral model: {pull_response.status_code}, {pull_response.text}")
                return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking/installing Mistral model: {str(e)}")
        return False

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
os.makedirs("/data/uploads", exist_ok=True)
os.makedirs("/data/extracted", exist_ok=True)

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
        logger.error(f"Error in analyze: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing file: {str(e)}"}
        )

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
            
            # Initialize LLM
            llm = OllamaLLM(
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
                    
                    # Create a vector store for retrieval
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        persist_directory=f"/data/temp_vectorstore_{integration_name}_{int(time.time())}",
                    )
                    
                    # Create a retrieval chain for this integration
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
                    logger.error(f"Error processing integration {integration_name}: {str(e)}")
                    # Add error message to documentation
                    all_docs.append({
                        "name": integration_name,
                        "documentation": f"Error generating documentation: {str(e)}",
                        "error": True
                    })
            
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
            
        else:
            # This is a regular single JSON file analysis
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
                    logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        return JSONResponse(
                            status_code=503,
                            content={"error": f"Failed to generate documentation after {max_retries} attempts: {str(e)}"}
                        )
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
    
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
            
            # Initialize LLM
            llm = OllamaLLM(
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
                    
                    # Create a vector store for retrieval
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        persist_directory=f"/data/temp_vectorstore_{integration_name}_{int(time.time())}",
                    )
                    
                    # Create a retrieval chain for this integration
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
                    logger.error(f"Error processing integration {integration_name}: {str(e)}")
                    # Add error message to documentation
                    all_docs.append({
                        "name": integration_name,
                        "documentation": f"Error generating documentation: {str(e)}",
                        "error": True
                    })
            
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
            
        else:
            # This is a regular single JSON file analysis
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
                    logger.error(f"LLM attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        return JSONResponse(
                            status_code=503,
                            content={"error": f"Failed to generate documentation after {max_retries} attempts: {str(e)}"}
                        )
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
    
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
        if os.path.exists(vectorstore_dir):
            # Rename existing vectorstore to backup
            backup_dir = f"{vectorstore_dir}_backup_{int(time.time())}"
            os.rename(vectorstore_dir, backup_dir)
            logger.info(f"Backed up existing vectorstore to {backup_dir}")
        
        # Create documents from all examples
        documents = []
        for example_file in example_files:
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
                logger.error(f"Error processing example file {example_file}: {str(e)}")
        
        if not documents:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid documents found in training examples"}
            )
        
        # Create new vectorstore from documents
        logger.info(f"Creating training vectorstore with {len(documents)} documents")
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vectorstore_dir,
            collection_name="training_examples"
        )
        
        # Remove rebuild flag if it exists
        if os.path.exists("/data/training/rebuild_required"):
            os.remove("/data/training/rebuild_required")
        
        return {"message": f"Successfully rebuilt training vectorstore with {len(documents)} examples"}
    except Exception as e:
        logger.error(f"Error rebuilding vectorstore: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error rebuilding vectorstore: {str(e)}"}
        )

@app.post("/train-with-zip")
async def train_with_zip(
    filename: str = Form(...),
    integration_id: str = Form(...),
    documentation: str = Form(...)
):
    """
    Train the model with a single integration from a ZIP file.
    The integration is identified by integration_id from the uploaded ZIP archive.
    """
    try:
        logger.info(f"Processing training data from ZIP file: {filename}")
        
        # Extract the upload ID from the filename
        upload_id = filename.split('_')[0]
        extract_dir = f"/data/extracted/{upload_id}"
        
        if not os.path.exists(extract_dir):
            return JSONResponse(
                status_code=404,
                content={"error": "Extracted archive not found. Please upload the file again."}
            )
        
        # Find the integration file
        integrations = find_integrations_in_archive(extract_dir)
        selected_integration = None
        
        for integration in integrations:
            if integration["id"] == integration_id:
                selected_integration = integration
                break
        
        if not selected_integration:
            return JSONResponse(
                status_code=404,
                content={"error": f"Integration with ID {integration_id} not found in the archive"}
            )
        
        # Read the JSON file
        file_path = selected_integration["file_path"]
        integration_name = selected_integration["name"]
        
        try:
            with open(file_path, 'r') as f:
                integration_json = json.load(f)
                
            # Analyze the integration flow to get comprehensive information
            flow_analysis = analyze_integration_flow(integration_id, extract_dir)
            
            # Convert the basic JSON to a more flat structure for processing
            flat_json = flatten_json(integration_json)
            
            # Combine the basic JSON and flow analysis into enriched JSON
            enriched_json = {
                "flow_analysis": flow_analysis,
                "basic_data": flat_json,
                "integration_id": integration_id,
                "integration_name": integration_name
            }
        except Exception as e:
            logger.error(f"Error reading and processing integration file {file_path}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error reading and processing integration file: {str(e)}"}
            )
        
        # Create a training example with the enriched JSON
        training_example = {
            "integration_json": enriched_json,
            "documentation": documentation
        }
        
        # Generate a unique ID for this training file
        training_id = str(uuid.uuid4())
        
        # Ensure the langchain directory exists
        os.makedirs("/data/langchain", exist_ok=True)
        
        training_file_path = f"/data/langchain/{integration_name}_{training_id}.json"
        
        # Save the training data
        with open(training_file_path, 'w') as f:
            json.dump(training_example, f, indent=2)
        
        # Also save a markdown version for reference
        markdown_path = f"/data/langchain/{integration_name}_{training_id}.md"
        with open(markdown_path, 'w') as f:
            f.write(f"# Training Example for {integration_name}\n\n")
            f.write("## Integration Structure\n\n")
            
            # Add component structure if available
            if flow_analysis and "components" in flow_analysis:
                f.write("### Components\n\n")
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
                    f.write(component_desc + "\n")
                f.write("\n")
            
            f.write("## Expected Documentation\n\n")
            f.write(documentation)
        
        logger.info(f"Training data saved to {training_file_path}")
        
        return {
            "message": f"Training data for {integration_name} saved successfully",
            "training_id": training_id
        }
        
    except Exception as e:
        logger.error(f"Error in train_with_zip: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing training data: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
