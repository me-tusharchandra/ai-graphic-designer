import os
import io
import json
import time
import openai
import base64
import svgwrite
import argparse
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

# Available models
GPT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

def process_logo(logo_path):
    """Process the logo image file"""
    image = Image.open(logo_path)
    
    # Extract logo colors (simplified version)
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
    """Extract dominant colors from the logo (simplified)"""
    # Placeholder for color extraction
    # In a real implementation, use k-means clustering or similar
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"][:num_colors]
    return colors

async def analyze_design_request(logo_data, brand_name, brand_description, design_description, model):
    """Use specified GPT model to analyze the design request"""
    
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
    
    # Call GPT API with specified model - updated for OpenAI v1.0+
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional graphic designer specializing in brand-aligned design creation."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # Parse the response - updated for OpenAI v1.0+
    design_specs = json.loads(response.choices[0].message.content)
    
    # Add logo data and model info to the specs
    design_specs["logo_data"] = logo_data
    design_specs["model_used"] = model
    
    return design_specs

def generate_svg_design(design_specs):
    """Generate an SVG design based on the specifications"""
    
    # Create SVG document
    width, height = 1200, 630  # Social media post size
    dwg = svgwrite.Drawing(size=(width, height))
    
    # Extract color palette with fallbacks
    color_palette = design_specs.get("color_palette", {})
    primary_color = color_palette.get("primary", "#FFFFFF")  # White fallback
    secondary_color = color_palette.get("secondary", "#000000")  # Black fallback
    
    # Make sure we're using a single color string, not a list
    dominant_colors = design_specs.get("logo_data", {}).get("dominant_colors", ["#FF5733"])
    accent_color = color_palette.get("accent", dominant_colors[0] if dominant_colors else "#FF5733")  # Use first dominant color as fallback
    text_color = color_palette.get("text", "#333333")  # Dark gray fallback
    
    # Extract typography with fallbacks
    typography = design_specs.get("typography", {})
    heading_font = typography.get("heading_font", "Arial")
    body_font = typography.get("body_font", "Arial")
    
    # Add background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=primary_color))
    
    # Add decorative elements
    # Add a subtle pattern or texture
    pattern_group = dwg.g(id="pattern-group")
    for i in range(20):
        x = i * 60
        pattern_group.add(dwg.line(start=(x, 0), end=(x, height), stroke=secondary_color, stroke_width=0.5, stroke_opacity=0.1))
    for i in range(12):
        y = i * 60
        pattern_group.add(dwg.line(start=(0, y), end=(width, y), stroke=secondary_color, stroke_width=0.5, stroke_opacity=0.1))
    dwg.add(pattern_group)
    
    # Add a large central circle as requested in the design description
    central_circle_group = dwg.g(id="central-circle")
    # Add highlight behind the circle
    highlight = dwg.circle(center=(width/2, height/2), r=220, 
                          fill=accent_color, fill_opacity=0.15)
    # Add the main circle
    main_circle = dwg.circle(center=(width/2, height/2), r=200, 
                            fill=secondary_color, fill_opacity=0.1)
    central_circle_group.add(highlight)
    central_circle_group.add(main_circle)
    dwg.add(central_circle_group)
    
    # Add decorative shapes
    shapes_group = dwg.g(id="decorative-shapes")
    # Add circles in the corners
    shapes_group.add(dwg.circle(center=(100, 100), r=80, fill=accent_color, fill_opacity=0.2))
    shapes_group.add(dwg.circle(center=(width-100, 100), r=60, fill=secondary_color, fill_opacity=0.15))
    shapes_group.add(dwg.circle(center=(100, height-100), r=70, fill=secondary_color, fill_opacity=0.1))
    shapes_group.add(dwg.circle(center=(width-100, height-100), r=50, fill=accent_color, fill_opacity=0.2))
    dwg.add(shapes_group)
    
    # Add logo with proper styling
    logo_group = dwg.g(id="logo-group")
    # Create a more sophisticated logo placeholder
    logo_rect = dwg.rect(insert=(width/2-100, 50), size=(200, 200), fill=accent_color, rx=10, ry=10)
    logo_text = dwg.text(design_specs.get("brand_name", "LOGO"), 
                        insert=(width/2, 150), 
                        font_family=heading_font,
                        font_size=36,
                        fill="#FFFFFF",
                        text_anchor="middle",
                        dominant_baseline="middle")
    logo_group.add(logo_rect)
    logo_group.add(logo_text)
    dwg.add(logo_group)
    
    # Add quote text with better styling
    quote_group = dwg.g(id="quote-group")
    
    # Add decorative quote marks
    opening_quote = dwg.text("\u201C",  # Unicode for left double quotation mark
                          insert=(width/2 - 150, height/2 - 50), 
                          font_family=heading_font,
                          font_size=120,
                          fill=accent_color,
                          fill_opacity=0.3,
                          text_anchor="middle")
    
    closing_quote = dwg.text("\u201D",  # Unicode for right double quotation mark
                          insert=(width/2 + 150, height/2 - 50), 
                          font_family=heading_font,
                          font_size=120,
                          fill=accent_color,
                          fill_opacity=0.3,
                          text_anchor="middle")
                          
    quote_group.add(opening_quote)
    quote_group.add(closing_quote)
    
    # Add the quote with a text shadow effect
    # First add a shadow/outline
    shadow_text = dwg.text("It Always Seems Impossible Until It's Done", 
                         insert=(width/2 + 2, height/2 + 2), 
                         font_family=heading_font,
                         font_size=36,
                         font_weight="bold",
                         fill=secondary_color,
                         fill_opacity=0.3,
                         text_anchor="middle")
    
    # Then add the main text
    quote_text = dwg.text("It Always Seems Impossible Until It's Done", 
                         insert=(width/2, height/2), 
                         font_family=heading_font,
                         font_size=36,
                         font_weight="bold",
                         fill=text_color,
                         text_anchor="middle")
    
    quote_group.add(shadow_text)
    quote_group.add(quote_text)
    dwg.add(quote_group)
    
    # Add a decorative line under the quote
    underline = dwg.line(start=(width/2 - 100, height/2 + 30), 
                        end=(width/2 + 100, height/2 + 30), 
                        stroke=accent_color, 
                        stroke_width=3)
    dwg.add(underline)
    
    # Add brand name with better styling
    brand_group = dwg.g(id="brand-group")
    # Add a subtle background for the brand name
    brand_bg = dwg.rect(insert=(width/2 - 150, height-130), 
                       size=(300, 60), 
                       fill=secondary_color, 
                       fill_opacity=0.1,
                       rx=5, ry=5)
    brand_group.add(brand_bg)
    
    brand_text = dwg.text(design_specs.get("brand_name", "Brand Name"), 
                         insert=(width/2, height-100), 
                         font_family=body_font,
                         font_size=24,
                         font_weight="bold",
                         fill=text_color,
                         text_anchor="middle")
    brand_group.add(brand_text)
    dwg.add(brand_group)
    
    # Add model attribution as a watermark
    model_used = design_specs.get("model_used", "unknown-model")
    dwg.add(dwg.text(f"Generated with {model_used}", 
                    insert=(width-200, height-20), 
                    font_size=12, 
                    fill="#999999"))
    
    # Add metadata to the SVG for editability
    metadata = dwg.g(id="metadata", display="none")
    metadata_text = dwg.text(json.dumps(design_specs), insert=(0, 0), font_size="0px")
    metadata.add(metadata_text)
    dwg.add(metadata)
    
    return dwg.tostring()

def compare_model_responses(responses):
    """Compare model responses and select the best one for interpretation"""
    # This is a simple heuristic - in reality, you might want more sophisticated comparison
    
    # Count the number of fields in each response as a simple quality metric
    scores = {}
    for model, specs in responses.items():
        # Count the number of fields and nested fields as a simple quality metric
        score = len(specs.keys())
        for key, value in specs.items():
            if isinstance(value, dict):
                score += len(value.keys())
            elif isinstance(value, list):
                score += len(value)
        scores[model] = score
    
    # Return the model with the highest score
    best_model = max(scores, key=scores.get)
    print(f"Selected {best_model} as the best model for interpretation")
    return best_model

async def main():
    parser = argparse.ArgumentParser(description='Generate SVGs based on brand guidelines')
    parser.add_argument('--logo', required=True, help='Path to logo image file')
    parser.add_argument('--brand_name', required=True, help='Brand name')
    parser.add_argument('--brand_description', required=True, help='Brand description')
    parser.add_argument('--design_description', required=True, help='Design description')
    parser.add_argument('--output_dir', default='generated_svgs', help='Output directory for SVGs')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process logo
    logo_data = process_logo(args.logo)
    
    # Generate designs with all models
    responses = {}
    svgs = {}
    
    for model in GPT_MODELS:
        print(f"Generating design with {model}...")
        design_specs = await analyze_design_request(
            logo_data, 
            args.brand_name, 
            args.brand_description, 
            args.design_description,
            model
        )
        responses[model] = design_specs
        svgs[model] = generate_svg_design(design_specs)
    
    # Determine the best model for interpretation
    best_model = compare_model_responses(responses)
    
    # Save all SVGs
    timestamp = int(time.time())
    safe_brand_name = args.brand_name.replace(' ', '_')
    
    for model, svg_content in svgs.items():
        filename = f"{safe_brand_name}_{model}_{timestamp}.svg"
        filepath = os.path.join(args.output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(svg_content)
        print(f"Saved {filepath}")
    
    # Save the best interpretation as a separate file
    best_specs = responses[best_model]
    best_interpretation_file = os.path.join(args.output_dir, f"{safe_brand_name}_best_interpretation_{timestamp}.json")
    
    with open(best_interpretation_file, "w") as f:
        json.dump(best_specs, f, indent=2)
    print(f"Saved best interpretation to {best_interpretation_file}")
    
    print("\nDesign generation complete!")
    print(f"Best model: {best_model}")
    print(f"Explanation: {best_specs.get('explanation', 'No explanation provided')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())