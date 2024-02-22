# ESG Policy Analyzer

This repository contains a Python-based tool that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to assist in the analysis of Environmental, Social, and Governance (ESG) policies.

**Key Features**

*   Identifies potential gaps and discrepancies between company policies and relevant ESG regulations.
*   Utilizes OpenAI's LLM for advanced text understanding and analysis generation.
*   Employs the Langchain library for flexible integration of LLMs and retrieval techniques.
*   Pinecone vector database for efficient storage and retrieval of regulations.
*   Streamlit integration for a user-friendly web interface.

**Prerequisites**

*   Python 3.7 or later
*   OpenAI API key 
*   Pinecone API key and environment 
*   A structured source of ESG regulations (e.g., JSON file, database)
*   Required Python libraries:
    *   `openai`
    *   `langchain`
    *   `streamlit`
    *   `spacy`
    *   `pypdf2`
    *   `pinecone-client`

**Installation**

1.  Clone this repository:
    ```bash
    git clone https://github.com/mominalix/ai-powered-esg-compliance.git
    ```

2.  Install dependencies:
    ```bash
    cd esg-policy-analyzer
    pip install -r requirements.txt 
    ```

**Running the Tool**

1.  Set your API keys in the `ESG assessment.py` file or as environment variables.
2.  Prepare your regulations data in the supported format.
3.  Start the Streamlit application:
    ```bash
    streamlit run ESGassessment.py
    ```

**Usage**

1.  Upload your company policy documents (PDF format).
2.  The tool will extract the policy text, analyze it against relevant ESG regulations, and highlight potential gaps or areas for improvement.

**Disclaimer**

This tool is intended for preliminary analysis and should not be considered a substitute for professional legal advice.

**Future Development**

*   Fine-tuning the LLM on ESG-specific data.
*   More sophisticated gap analysis logic using advanced NLP techniques.
*   Integration with external regulatory databases.
