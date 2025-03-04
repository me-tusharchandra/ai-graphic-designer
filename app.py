from fastapi import FastAPI, File, UploadFile, Form, Query
from pydantic import BaseModel
import io
from PIL import Image
import base64
import openai
import json
import svgwrite
import os
import time

app = FastAPI()

# Configure OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Available models
GPT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

class DesignResponse(BaseModel):
    svg_content: str
    design_explanation: str
    editable_elements: list
    model_used: str
    svg_filename: str

class EditRequest(BaseModel):
    svg_content: str
    edit_instruction: str
    model: str = "gpt-4o"

@app.post("/generate-design/", response_model=DesignResponse)
async def generate_design(
    logo: UploadFile = File(...),
    brand_name: str = Form(...),
    brand_description: str = Form(...),
    design_description: str = Form(...),
    model: str = Query("gpt-4o", enum=GPT_MODELS)
):
    # 1. Process the logo
    logo_image = await process_logo(logo)
    
    # 2. Analyze inputs with specified GPT model
    design_specs = await analyze_design_request(logo_image, brand_name, brand_description, design_description, model)
    
    # 3. Generate SVG based on specifications
    svg_content, editable_elements = await generate_svg_design(design_specs)
    
    # 4. Save SVG with model name in filename
    timestamp = int(time.time())
    filename = f"{brand_name.replace(' ', '_')}_{model}_{timestamp}.svg"
    svg_path = os.path.join("generated_svgs", filename)
    
    # Create directory if it doesn't exist
    os.makedirs("generated_svgs", exist_ok=True)
    
    # Save the SVG file
    with open(svg_path, "w") as f:
        f.write(svg_content)
    
    # 5. Prepare response
    return DesignResponse(
        svg_content=svg_content,
        design_explanation=design_specs["explanation"],
        editable_elements=editable_elements,
        model_used=model,
        svg_filename=filename
    )

async def process_logo(logo_file):
    """Process the uploaded logo file"""
    contents = await logo_file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Extract logo colors
    colors = extract_dominant_colors(image)
    
    # Convert to base64 for API transmission
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "base64": img_str,
        "dominant_colors": colors,
        "dimensions": image.size
    }

def extract_dominant_colors(image, num_colors=5):
    """Extract dominant colors from the logo"""
    # Resize image for faster processing
    img = image.copy()
    img.thumbnail((100, 100))
    
    # Convert to RGB if not already
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Use k-means clustering to find dominant colors
    pixels = list(img.getdata())
    width, height = img.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    
    # Flatten the list and sample pixels
    flat_pixels = [pixel for row in pixels for pixel in row]
    
    # Simple implementation - in production use sklearn or similar
    # This is a placeholder for the color extraction logic
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"][:num_colors]
    
    return colors

async def analyze_design_request(logo_data, brand_name, brand_description, design_description, model="gpt-4o"):
    """Use specified GPT model to analyze the design request and extract specifications"""
    
    # Prepare the prompt for GPT
    prompt = f"""
    I need to create a design based on the following information:
    
    Brand Name: {brand_name}
    Brand Description: {brand_description}
    Design Request: {design_description}
    
    The brand's logo has these dominant colors: {logo_data['dominant_colors']}
    
    Please analyze this information and provide detailed design specifications in JSON format:
    1. Color palette (primary, secondary, accent colors)
    2. Typography recommendations
    3. Layout structure
    4. Key visual elements to include
    5. Style guidance based on the brand description
    6. Specific SVG components needed
    7. Add an "explanation" field with a brief description of your design choices
    
    Format your response as a valid JSON object.
    """
    
    # Call GPT API with specified model
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional graphic designer specializing in brand-aligned design creation."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # Parse the response
    design_specs = json.loads(response.choices[0].message.content)
    
    # Add logo data and model info to the specs
    design_specs["logo_data"] = logo_data
    design_specs["model_used"] = model
    
    return design_specs

async def generate_svg_design(design_specs):
    """Generate an SVG design based on the specifications"""
    
    # Create SVG document
    width, height = 1200, 630  # Social media post size
    dwg = svgwrite.Drawing(size=(width, height))
    
    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), 
                    fill=design_specs["color_palette"]["primary"]))
    
    # Add model attribution as a watermark
    model_used = design_specs.get("model_used", "unknown-model")
    dwg.add(dwg.text(f"Generated with {model_used}", 
                    insert=(width-200, height-20), 
                    font_size=12, 
                    fill="#999999"))
    
    # Add editable elements with IDs and groups
    elements = []
    
    # Add logo placeholder
    logo_group = dwg.g(id="logo-group")
    logo_rect = dwg.rect(insert=(50, 50), size=(200, 200), 
                        fill=design_specs["color_palette"]["accent"])
    logo_group.add(logo_rect)
    dwg.add(logo_group)
    elements.append({"id": "logo-group", "type": "logo", "editable": True})
    
    # Add headline text
    headline_group = dwg.g(id="headline-group")
    headline_text = dwg.text(design_specs.get("headline", "Brand Headline"), 
                           insert=(50, 300), 
                           font_family=design_specs["typography"]["heading_font"],
                           font_size=48,
                           fill=design_specs["color_palette"]["text"])
    headline_group.add(headline_text)
    dwg.add(headline_group)
    elements.append({"id": "headline-group", "type": "text", "editable": True})
    
    # Add description text
    desc_group = dwg.g(id="description-group")
    desc_text = dwg.text(design_specs.get("description", "Brand description goes here."), 
                        insert=(50, 400), 
                        font_family=design_specs["typography"]["body_font"],
                        font_size=24,
                        fill=design_specs["color_palette"]["text"])
    desc_group.add(desc_text)
    dwg.add(desc_group)
    elements.append({"id": "description-group", "type": "text", "editable": True})
    
    # Add visual elements based on design specs
    for i, element in enumerate(design_specs.get("visual_elements", [])):
        element_group = dwg.g(id=f"visual-element-{i}")
        # This would be more complex in a real implementation
        # Just adding a placeholder shape here
        shape = dwg.rect(insert=(400 + i*100, 100), size=(80, 80), 
                        fill=design_specs["color_palette"]["secondary"])
        element_group.add(shape)
        dwg.add(element_group)
        elements.append({"id": f"visual-element-{i}", "type": "shape", "editable": True})
    
    # Add metadata to the SVG for editability
    metadata = dwg.g(id="metadata", display="none")
    metadata_text = dwg.text(json.dumps(design_specs), insert=(0, 0), font_size="0px")
    metadata.add(metadata_text)
    dwg.add(metadata)
    
    return dwg.tostring(), elements

@app.post("/edit-design/", response_model=DesignResponse)
async def edit_design(request: EditRequest):
    # 1. Parse the SVG to extract current structure and metadata
    current_design = parse_svg(request.svg_content)
    
    # 2. Process the edit instruction with specified GPT model
    edit_specs = await process_edit_instruction(request.edit_instruction, current_design, request.model)
    
    # 3. Apply edits to the SVG
    updated_svg, editable_elements = apply_edits(current_design, edit_specs)
    
    # 4. Save edited SVG with model name
    timestamp = int(time.time())
    filename = f"edited_{request.model}_{timestamp}.svg"
    svg_path = os.path.join("generated_svgs", filename)
    
    # Create directory if it doesn't exist
    os.makedirs("generated_svgs", exist_ok=True)
    
    # Save the SVG file
    with open(svg_path, "w") as f:
        f.write(updated_svg)
    
    return DesignResponse(
        svg_content=updated_svg,
        design_explanation=edit_specs["explanation"],
        editable_elements=editable_elements,
        model_used=request.model,
        svg_filename=filename
    )

def parse_svg(svg_content):
    """Parse SVG content to extract structure and metadata"""
    # This is a placeholder for actual SVG parsing logic
    # In a real implementation, you would use an XML parser
    
    # Mock implementation
    return {
        "elements": [
            {"id": "logo-group", "type": "logo"},
            {"id": "headline-group", "type": "text"},
            {"id": "description-group", "type": "text"}
        ],
        "metadata": {
            "color_palette": {
                "primary": "#FFFFFF",
                "secondary": "#000000",
                "accent": "#FF5733",
                "text": "#333333"
            }
        }
    }

async def process_edit_instruction(instruction, current_design, model="gpt-4o"):
    """Process natural language edit instruction using specified GPT model"""
    
    prompt = f"""
    I have an SVG design with the following elements:
    {json.dumps(current_design['elements'])}
    
    The user wants to make this edit: "{instruction}"
    
    Please provide specific edit instructions in JSON format:
    1. Which elements to modify (by ID)
    2. What properties to change
    3. New values for those properties
    4. Add an "explanation" field with a brief description of your edit choices
    
    Format your response as a valid JSON object.
    """
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional graphic designer who specializes in SVG editing."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    edit_specs = json.loads(response.choices[0].message.content)
    edit_specs["model_used"] = model
    return edit_specs

def apply_edits(current_design, edit_specs):
    """Apply edits to the SVG based on edit specifications"""
    # This is a placeholder for actual SVG editing logic
    # In a real implementation, you would modify the SVG DOM
    
    # Mock implementation
    updated_svg = f"<svg>Updated content would go here (edited with {edit_specs.get('model_used', 'unknown')})</svg>"
    editable_elements = current_design["elements"]
    
    return updated_svg, editable_elements

@app.get("/compare-models/")
async def compare_models(
    logo: UploadFile = File(...),
    brand_name: str = Form(...),
    brand_description: str = Form(...),
    design_description: str = Form(...)
):
    """Generate designs using all available models for comparison"""
    results = []
    
    for model in GPT_MODELS:
        # Process with each model
        response = await generate_design(
            logo=logo,
            brand_name=brand_name,
            brand_description=brand_description,
            design_description=design_description,
            model=model
        )
        results.append({
            "model": model,
            "svg_filename": response.svg_filename,
            "design_explanation": response.design_explanation
        })
    
    return {"comparison_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 