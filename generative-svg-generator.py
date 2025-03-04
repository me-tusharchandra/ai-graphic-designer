import os
import io
import json
import time
import openai
import base64
import argparse
import anthropic
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

# Configure Anthropic API
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

# Available models
GPT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
CLAUDE_MODELS = ["claude-3-7-sonnet-20250219"]
ALL_MODELS = GPT_MODELS + CLAUDE_MODELS

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

async def generate_svg_with_llm(logo_data, brand_name, brand_description, design_description, model):
    """Use specified LLM to generate SVG code directly"""
    
    # Prepare the prompt for the LLM
    prompt = f"""
    I need you to generate an SVG design based on the following information:
    
    Brand Name: {brand_name}
    Brand Description: {brand_description}
    Design Request: {design_description}
    
    The brand's logo has these dominant colors: {logo_data['dominant_colors']}
    
    Please generate complete, valid SVG code that:
    1. Has dimensions of 1200x630 pixels (social media post size)
    2. Incorporates the brand name and colors
    3. Follows the design request exactly
    4. Includes the quote "{design_description.split('"')[1]}" if a quote is mentioned
    5. Has a clean, modern design with appropriate typography
    6. Includes decorative elements that enhance the design
    7. Has a watermark indicating it was generated with {model}
    
    Return ONLY the complete SVG code without any explanation or markdown. The SVG should start with <svg> and end with </svg>.
    """
    
    # Call the appropriate API based on the model
    if model in GPT_MODELS:
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional SVG designer. You create clean, valid SVG code based on design requirements."},
                {"role": "user", "content": prompt}
            ]
        )
        svg_code = response.choices[0].message.content.strip()
    
    elif model in CLAUDE_MODELS:
        # Call Anthropic API
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            system="You are a professional SVG designer. You create clean, valid SVG code based on design requirements.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        svg_code = response.content[0].text
    
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    # Clean up the response to ensure it's valid SVG
    if svg_code.startswith("```") and svg_code.endswith("```"):
        svg_code = svg_code[3:-3].strip()
    
    if svg_code.startswith("```svg") or svg_code.startswith("```xml"):
        svg_code = svg_code.split("\n", 1)[1]
        svg_code = svg_code.rsplit("```", 1)[0].strip()
    
    # Ensure the SVG code starts with <svg and ends with </svg>
    if not svg_code.startswith("<svg"):
        svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630">\n{svg_code}\n</svg>'
    
    # Add the actual logo image to the SVG using the base64 data
    logo_image_element = f'''
    <image x="500" y="50" width="200" height="200" 
           xlink:href="data:image/png;base64,{logo_data['base64']}"
           preserveAspectRatio="xMidYMid meet" />
    '''
    
    # Make sure the SVG has the xlink namespace
    if 'xmlns:xlink="http://www.w3.org/1999/xlink"' not in svg_code:
        svg_code = svg_code.replace('<svg ', '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ')
    
    # Insert the logo image before the closing </svg> tag
    if "</svg>" in svg_code:
        svg_code = svg_code.replace("</svg>", f"{logo_image_element}\n</svg>")
    else:
        svg_code += f"\n{logo_image_element}\n</svg>"
    
    # Add metadata to the SVG
    metadata = f'<metadata id="metadata">\n<data id="design-info">\n<![CDATA[\n{json.dumps({"brand_name": brand_name, "brand_description": brand_description, "design_description": design_description, "model_used": model})}\n]]>\n</data>\n</metadata>'
    
    # Insert metadata before the closing </svg> tag
    if "</svg>" in svg_code:
        svg_code = svg_code.replace("</svg>", f"{metadata}\n</svg>")
    else:
        svg_code += f"\n{metadata}\n</svg>"
    
    return svg_code

async def main():
    parser = argparse.ArgumentParser(description='Generate SVGs using LLM')
    parser.add_argument('--logo', required=True, help='Path to logo image file')
    parser.add_argument('--brand_name', required=True, help='Brand name')
    parser.add_argument('--brand_description', required=True, help='Brand description')
    parser.add_argument('--design_description', required=True, help='Design description')
    parser.add_argument('--output_dir', default='generated_svgs', help='Output directory for SVGs')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, help='Models to use for generation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process logo
    logo_data = process_logo(args.logo)
    
    # Generate SVGs with specified models
    timestamp = int(time.time())
    safe_brand_name = args.brand_name.replace(' ', '_')
    
    for model in args.models:
        if model not in ALL_MODELS:
            print(f"Warning: {model} is not in the list of supported models. Skipping.")
            continue
            
        print(f"Generating SVG with {model}...")
        try:
            svg_code = await generate_svg_with_llm(
                logo_data, 
                args.brand_name, 
                args.brand_description, 
                args.design_description,
                model
            )
            
            # Save the SVG
            filename = f"{safe_brand_name}_{model}_{timestamp}.svg"
            filepath = os.path.join(args.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(svg_code)
            print(f"Saved {filepath}")
            
        except Exception as e:
            print(f"Error generating SVG with {model}: {e}")
    
    print("\nSVG generation complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 