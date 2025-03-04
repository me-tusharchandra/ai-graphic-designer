# SVG Design Generator API - [DEMO 🎬](https://www.youtube.com/watch?v=6A6fipaYcuo)

A FastAPI application that generates and edits SVG designs using OpenAI's GPT models.

## Overview

This API allows users to:
- Generate custom SVG designs based on brand information and a logo
- Edit existing SVG designs using natural language instructions
- Compare designs generated by different GPT models

## Features

- **Design Generation**: Upload a logo and provide brand details to generate a custom SVG design
- **Design Editing**: Modify existing SVG designs with natural language instructions
- **Model Comparison**: Generate designs using multiple GPT models to compare results
- **Editable Elements**: All generated designs include editable elements with proper metadata

## Requirements

- Python 3.8+
- FastAPI
- Pillow
- svgwrite
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/svg-design-generator.git
   cd svg-design-generator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Start the server:

```
uvicorn app:app --reload
```

The API will be available at http://localhost:8000

### API Endpoints

#### Generate a Design

```
POST /generate-design/
```

Parameters:
- `logo`: Image file (required)
- `brand_name`: String (required)
- `brand_description`: String (required)
- `design_description`: String (required)
- `model`: String (optional, default: "gpt-4o")

#### Edit a Design

```
POST /edit-design/
```

Request body:
```json
{
  "svg_content": "...",
  "edit_instruction": "Make the background blue and increase the font size",
  "model": "gpt-4o"
}
```

Parameters:
- `logo`: Image file (required)
- `brand_name`: String (required)
- `brand_description`: String (required)
- `design_description`: String (required)
- `model`: String (optional, default: "gpt-4o")

#### Compare Models

```
GET /compare-models/
```

Parameters:
- `logo`: Image file (required)
- `brand_name`: String (required)
- `brand_description`: String (required)
- `design_description`: String (required)

## Example

```python
import requests

url = "http://localhost:8000/generate-design/"
files = {"logo": open("logo.png", "rb")}
data = {
    "brand_name": "Acme Corp",
    "brand_description": "A technology company focused on innovation",
    "design_description": "Create a modern social media banner for our new product launch",
    "model": "gpt-4o"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Project Structure

- `app.py`: Main application file with API endpoints and business logic
- `generated_svgs/`: Directory where generated SVG files are stored

## Supported Models

- gpt-4o
- gpt-4o-mini
- gpt-4-turbo

## License

[MIT License](LICENSE)
