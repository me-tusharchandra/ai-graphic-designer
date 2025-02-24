import os
import json
import dotenv
import textwrap
import svgwrite
from openai import OpenAI
from PyPDF2 import PdfReader

dotenv.load_dotenv()

def extract_brand_guidelines(pdf_path):
    """Extract text from brand guidelines PDF"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def analyze_brand_with_ai(text):
    """Analyze brand guidelines and extract structured rules"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    analysis_prompt = textwrap.dedent("""
    Analyze the brand guidelines and create structured rules for asset generation.
    Focus on these aspects:
    
    1. Color System:
    - Primary palette (HEX/RGB)
    - Secondary palette
    - Usage ratios
    - Color combinations to avoid
    
    2. Typography:
    - Primary typeface (headings)
    - Secondary typeface (body)
    - Tertiary typeface (accent)
    - Size hierarchy
    - Line spacing rules
    
    3. Imagery Rules (if mentioned):
    - Graphic style description (geometric/organic/abstract)
    - Shape vocabulary
    - Line thickness
    - Pattern styles
    - Iconography rules
    - Composition principles
    
    4. Brand Graphics (if described):
    - Recurring graphic elements
    - Spatial relationships
    - Scaling rules
    - Opacity guidelines
    - Texture usage
    
    5. Composition:
    - Grid systems
    - Margins/padding rules
    - Alignment principles
    - White space usage
    
    Format as JSON with these keys:
    {
        "color_system": { ... },
        "typography": { ... },
        "imagery_rules": { ... },
        "brand_graphics": { ... },
        "composition_rules": { ... }
    }
    """)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": f"{analysis_prompt}\n\nBRAND GUIDELINES:\n{text}"
        }],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def generate_svg_design(brand_specs, output_path):
    """Generate SVG design based on brand specifications"""
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=("800px", "600px"))
    
    # Background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=brand_specs.get("secondary_color", "#FFFFFF")))
    
    # Header with primary color
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "80px"), fill=brand_specs.get("primary_color", "#000000")))
    
    # Sample logo text
    dwg.add(dwg.text("Brand Logo", insert=(20, 50), 
                    font_family=brand_specs.get("primary_font", "Arial"),
                    fill=brand_specs.get("secondary_color", "#000000"),
                    font_size="40px"))
    
    # Content area
    content = dwg.add(dwg.g(transform="translate(50, 100)"))
    
    # Sample social media post
    content.add(dwg.text("Social Media Post", insert=(0, 30),
                        font_family=brand_specs.get("secondary_font", "Helvetica"),
                        fill=brand_specs.get("primary_color", "#000000"),
                        font_size="24px"))
    
    # Save SVG
    dwg.save()

# Example usage
if __name__ == "__main__":
    # Step 1: Extract text from PDF
    brand_text = extract_brand_guidelines("ripen-brandbook.pdf")
    
    # Step 2: Analyze with AI
    brand_rules = analyze_brand_with_ai(brand_text)
    
    print("Structured Brand Rules:")
    print(json.dumps(brand_rules, indent=2))
    
    # Step 3: Generate SVG
    generate_svg_design({
        "primary_color": "#2A5CAA",
        "secondary_color": "#FFFFFF",
        "primary_font": "Helvetica",
        "secondary_font": "Arial"
    }, "social_media_post.svg")