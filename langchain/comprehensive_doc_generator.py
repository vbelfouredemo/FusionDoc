"""
Comprehensive documentation generator for FusionDoc

This module handles the enhanced documentation generation process
using the improved flow analysis and component type understanding.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional

from langchain_community.llms.ollama import Ollama

logger = logging.getLogger("FusionDoc")

class ComprehensiveDocGenerator:
    """
    Enhanced documentation generator that leverages improved flow analysis
    and comprehensive component type understanding to generate detailed documentation.
    """
    
    def __init__(self, model_name: str = "mistral"):
        """
        Initialize the comprehensive documentation generator.
        
        Args:
            model_name: The LLM model name to use
        """
        self.model_name = model_name
        self.llm = Ollama(model=model_name, base_url="http://ollama:11434")
        
        # Initialize component registry and flow structure modules if available
        try:
            # Try to import the enhanced modules first
            try:
                from component_registry_enhanced import get_component_info, extract_key_properties, enhance_flow_analysis
                from flow_structure_enhanced import (
                    generate_flow_structure, 
                    generate_flow_patterns, 
                    generate_component_details_section,
                    generate_data_transformation_section,
                    generate_technical_specifications
                )
                from integration_patterns import enhance_flow_with_patterns
                
                self.get_component_info = get_component_info
                self.extract_key_properties = extract_key_properties
                self.enhance_flow_analysis = enhance_flow_analysis
                self.generate_flow_structure = generate_flow_structure
                self.generate_flow_patterns = generate_flow_patterns
                self.generate_component_details_section = generate_component_details_section
                self.generate_data_transformation_section = generate_data_transformation_section
                self.generate_technical_specifications = generate_technical_specifications
                self.enhance_flow_with_patterns = enhance_flow_with_patterns
                
                self.enhanced_modules_available = True
                self.comprehensive_modules_available = True
                self.integration_patterns_available = True
                logger.info("Comprehensive documentation modules loaded successfully")
                
            except ImportError:
                # Fall back to basic enhanced modules
                from component_registry import get_component_info, extract_key_properties
                from flow_structure import generate_flow_structure, generate_flow_patterns, generate_component_details_section
                
                self.get_component_info = get_component_info
                self.extract_key_properties = extract_key_properties
                self.generate_flow_structure = generate_flow_structure
                self.generate_flow_patterns = generate_flow_patterns
                self.generate_component_details_section = generate_component_details_section
                
                self.enhanced_modules_available = True
                self.comprehensive_modules_available = False
                self.integration_patterns_available = False
                logger.info("Enhanced documentation modules loaded successfully")
                
        except ImportError as e:
            self.enhanced_modules_available = False
            self.comprehensive_modules_available = False
            logger.warning(f"Enhanced documentation modules not available: {str(e)}")
    
    def get_query_template(self) -> str:
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
        
        # Fall back to a minimal template
        logger.info("Using minimal default query template")
        return """Generate detailed documentation for the '{integration_name}' integration.

The integration has the following structure and components:

{flow_structure}

Your task is to write comprehensive documentation that includes an introduction,
detailed flow description, and technical details for each component.
"""
    
    async def generate_documentation(
        self, 
        integration_json: Dict[str, Any], 
        integration_name: str,
        flow_analysis: Dict[str, Any],
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate documentation using the comprehensive flow analysis.
        
        Args:
            integration_json: The integration JSON data
            integration_name: The name of the integration
            flow_analysis: The flow analysis data
            output_format: The output format (markdown, html, etc.)
            
        Returns:
            Dictionary with documentation and metadata
        """
        logger.info(f"Generating comprehensive documentation for {integration_name}")
        
        # Apply enhanced flow analysis if available
        if hasattr(self, 'enhance_flow_analysis'):
            logger.info("Applying comprehensive flow analysis enhancements")
            flow_analysis = self.enhance_flow_analysis(flow_analysis)
        
        # Apply integration patterns analysis if available
        if hasattr(self, 'enhance_flow_with_patterns'):
            logger.info("Enhancing flow analysis with integration patterns")
            flow_analysis = self.enhance_flow_with_patterns(flow_analysis)
            
            # Add classification to the flow structure if available
            if "integration_classification" in flow_analysis:
                classifications = flow_analysis["integration_classification"]
                logger.info(f"Integration classified as: {', '.join(classifications)}")
        
        # Get the query template
        template = self.get_query_template()
        
        # Generate the flow structure information
        if self.enhanced_modules_available:
            # Use enhanced flow structure generation
            flow_structure = self.generate_flow_structure(flow_analysis)
            flow_patterns = self.generate_flow_patterns(flow_analysis)
            component_details = self.generate_component_details_section(flow_analysis)
            
            # Add comprehensive sections if available
            if self.comprehensive_modules_available:
                data_transformations = self.generate_data_transformation_section(flow_analysis)
                technical_specs = self.generate_technical_specifications(flow_analysis)
        else:
            # Fallback to basic flow structure
            flow_structure = f"# Integration: {flow_analysis.get('name')}\n"
            flow_structure += "## Components:\n"
            for comp in flow_analysis.get("components", []):
                flow_structure += f"- {comp.get('name')} ({comp.get('type')})\n"
            
            flow_patterns = "No specific flow patterns detected."
            component_details = "Please describe each component based on its type and role in the integration."
            
            if self.comprehensive_modules_available:
                data_transformations = "Please detail the data transformations in this integration."
                technical_specs = "Please include technical specifications for this integration."
        
        # Prepare the prompt with the flow structure
        prompt = template.replace("{integration_name}", integration_name)
        prompt = prompt.replace("{flow_structure}", flow_structure)
        
        # Add flow patterns if the template has the placeholder
        if "{flow_patterns}" in prompt:
            prompt = prompt.replace("{flow_patterns}", flow_patterns)
        
        # Add component details if the template has the placeholder
        if "{component_details}" in prompt:
            prompt = prompt.replace("{component_details}", component_details)
        
        # Add data transformations if the template has the placeholder and it's available
        if "{data_transformations}" in prompt and self.comprehensive_modules_available:
            prompt = prompt.replace("{data_transformations}", data_transformations)
        
        # Add technical specifications if the template has the placeholder and it's available
        if "{technical_specifications}" in prompt and self.comprehensive_modules_available:
            prompt = prompt.replace("{technical_specifications}", technical_specs)
        
        # Add integration classification if available
        if "integration_classification" in flow_analysis:
            classifications = flow_analysis["integration_classification"]
            classification_text = f"Integration Type: {', '.join(classifications)}"
            
            # Add to prompt
            if "{integration_classification}" in prompt:
                prompt = prompt.replace("{integration_classification}", classification_text)
            else:
                # Add after integration name
                prompt = prompt.replace(
                    f"'{integration_name}'", 
                    f"'{integration_name}' ({classification_text})"
                )
        
        # Generate the documentation using the LLM
        logger.info("Generating documentation with LLM")
        try:
            documentation = await self.llm.ainvoke(prompt)
            
            # Log success
            logger.info(f"Documentation generated successfully, length: {len(documentation)}")
            
            # Return the documentation and metadata
            return {
                "documentation": documentation,
                "format": output_format,
                "integration_name": integration_name,
                "used_enhanced_analysis": self.enhanced_modules_available,
                "used_comprehensive_analysis": self.comprehensive_modules_available
            }
        except Exception as e:
            # Log error and return error message
            logger.error(f"Error generating documentation: {str(e)}")
            
            return {
                "documentation": f"Error generating documentation: {str(e)}",
                "format": output_format,
                "integration_name": integration_name,
                "error": str(e)
            }
