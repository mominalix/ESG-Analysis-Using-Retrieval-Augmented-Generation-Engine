# Sample ESG Data

This directory contains sample ESG documents and data for testing and demonstration purposes.

## Directory Structure

```
sample_data/
├── frameworks/          # ESG framework documents
│   ├── csrd/           # CSRD (Corporate Sustainability Reporting Directive)
│   ├── gri/            # GRI Standards
│   ├── sasb/           # SASB Standards
│   ├── tcfd/           # TCFD Recommendations
│   └── eu_taxonomy/    # EU Taxonomy Regulation
├── policies/           # Sample company policies
├── reports/            # Sample ESG reports
├── regulations/        # Regulatory documents
└── integration_examples/ # Integration examples

```

## Sample Documents Included

### ESG Frameworks
- **CSRD**: European Sustainability Reporting Standards (ESRS)
- **GRI**: Global Reporting Initiative Universal Standards
- **SASB**: Sustainability Accounting Standards Board Materiality Map
- **TCFD**: Task Force on Climate-related Financial Disclosures Framework
- **EU Taxonomy**: Environmental Objectives and Technical Screening Criteria

### Company Policies
- Environmental Management Policy
- Diversity and Inclusion Policy  
- Code of Conduct and Ethics
- Supply Chain Sustainability Policy
- Data Privacy and Security Policy

### Reports
- Annual Sustainability Report Template
- Carbon Footprint Assessment Report
- Social Impact Measurement Report
- Governance and Risk Management Report

## Usage

1. **Upload Documents**: Use the API endpoints to upload these sample documents
2. **Test Queries**: Ask questions about ESG compliance, gaps, and best practices
3. **Framework Comparison**: Compare requirements across different frameworks

## Example Queries

### CSRD Compliance
- "What are the mandatory disclosure requirements under CSRD for carbon emissions?"
- "How should companies report on biodiversity impacts according to ESRS E4?"
- "What are the double materiality assessment requirements?"

### Cross-Framework Analysis
- "What are the differences between GRI and SASB materiality approaches?"
- "How do TCFD climate risk disclosures align with CSRD requirements?"
- "Which frameworks require Scope 3 emissions reporting?"

### Policy Gap Analysis
- "Does our environmental policy cover all CSRD environmental disclosure requirements?"
- "What governance elements are missing compared to GRI 2-9 requirements?"
- "How does our diversity policy align with SASB human capital disclosure standards?"

## Integration Examples

### Upload via API
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data/frameworks/csrd/ESRS_E1_Climate_Change.pdf" \
  -F "esg_framework=CSRD" \
  -F "document_type=standard"
```

### Batch Upload
```bash
curl -X POST "http://localhost:8000/api/v1/documents/batch-upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_data/frameworks/csrd/ESRS_E1_Climate_Change.pdf" \
  -F "files=@sample_data/frameworks/gri/GRI_301_Materials.pdf" \
  -F "esg_framework=Multi-Framework"
```

### Query Example
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key climate-related disclosure requirements?",
    "esg_framework": "CSRD",
    "search_strategy": "hybrid",
    "k": 5
  }'
```

## Data Sources

All sample documents are derived from publicly available sources:
- Official ESG framework publications
- Regulatory guidance documents
- Industry best practice examples
- Template documents from standard-setting organizations

## Disclaimer

These are sample documents for demonstration purposes only. For actual compliance work, always refer to the latest official publications from the respective standard-setting organizations and regulatory bodies.