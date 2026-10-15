# Freight Cost Assistant

An AI-powered application designed to help users analyze freight costs, understand shipment data, and obtain insights through a conversational interface.

## Features

- **Freight Cost Analysis:** Analyze shipment and freight-related data.
- **AI-Powered Assistant:** Ask questions about freight costs and receive responses using a configured language model.
- **Data-Driven Insights:** Help identify cost patterns and understand shipment information.
- **Interactive Interface:** Explore the application's functionality through a Streamlit interface.

_Note: Available features depend on the current implementation and configuration._

## Technology Stack

- Python
- Streamlit
- Groq API for language-model integration
- Pandas and other Python libraries used by the project

## Prerequisites

Before running the application, ensure you have:

- Python 3.10 or a compatible version supported by the project's dependencies.
- pip, the Python package installer.
- A Groq API key if you want to use the AI assistant functionality.

## Installation

### 1. Clone the Repository

```bash
git clone
cd freight-cost-assistant
```

Replace the repository URL and directory name with the actual values for your project.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If the project does not yet contain a `requirements.txt` file, create one containing the dependencies actually used by the application before proceeding.

## Configuration

The AI assistant requires a valid Groq API key.

### 1. Set the API Key

Create a `.env` file in the project root if the application uses `python-dotenv`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, set the environment variable directly.

**Windows Command Prompt:**

```cmd
set GROQ_API_KEY=your_groq_api_key_here
```

**PowerShell:**

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

**Linux/macOS:**

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Use a valid API key and keep it private. Never commit your `.env` file or personal API credentials to the repository.

### 2. Configure the Language Model

Ensure that the model configured in the application is supported by Groq and accessible to the API key being used.

The evaluator can configure their own API key and update the model configuration if necessary.

## Running the Application

From the project root, activate the virtual environment and execute the Streamlit entry-point file.

For example, if the main file is `app.py`:

```bash
streamlit run app.py
```

If your application uses a different entry-point filename, replace `app.py` with the correct path.

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in a browser to interact with the application.

## Usage

1. Start the application using Streamlit.
2. Explore the available freight-cost analysis features.
3. Provide shipment data or inputs supported by the current application.
4. Review the available cost analysis and insights.
5. Use the AI assistant, if configured, to ask questions related to the supported freight and shipment data.

The exact workflow depends on the features implemented in the current version.

## Troubleshooting

### Missing API Key

If the AI assistant reports that the API key is missing, configure `GROQ_API_KEY` in the environment or in the appropriate secrets configuration.

### Model Not Found

If Groq returns a model-related error, check the configured model name and verify that the model is currently supported and accessible.

### Missing Python Packages

If imports fail, activate the project's virtual environment and run:

```bash
pip install -r requirements.txt
```

### Application Does Not Start

Verify that Python is installed, all required dependencies are available, and the correct Streamlit entry-point file is being used.

## Security

- API keys must be provided by each user or deployment environment.
- Do not upload personal credentials or secret configuration files.
- Keep dependencies and model configurations up to date.

## Limitations

- AI-generated responses depend on the configured language model and API availability.
- The AI assistant requires a valid API key.
- Analysis capabilities depend on the data formats and functionality implemented in the current version.

## Future Improvements

- Enhanced freight-cost forecasting and comparison.
- Additional shipment-data analysis.
- Improved reporting and visualization.
- Expanded validation and automated testing.

## Author

Developed as a Freight Cost Assistant project for demonstrating freight-cost analysis and AI-assisted insights.
