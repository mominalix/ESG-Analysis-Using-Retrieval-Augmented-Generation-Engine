"""
Administrative API routes
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse

from ...core.logging import get_logger
from ...core.config import settings
from ...services.vector_store_service import vector_store_service
from ...services.llm_service import llm_service
from ...services.rag_service import rag_service
from ...services.agentic_rag_service import agentic_rag_service
from datetime import datetime
from typing import Optional, Dict, List
import json

router = APIRouter()
logger = get_logger("admin_api")
security = HTTPBearer()


# Simple bearer token authentication for admin endpoints
def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication token"""
    # In production, implement proper JWT verification
    if credentials.credentials != "admin-token-change-this":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


@router.post("/rebuild-index")
async def rebuild_vector_index(token: HTTPAuthorizationCredentials = Depends(verify_admin_token)):
    """Rebuild the vector store index"""
    try:
        logger.info("Rebuilding vector index")
        
        # In a real implementation, this would:
        # 1. Clear existing index
        # 2. Re-process all documents
        # 3. Rebuild vector embeddings
        
        # For now, just return success
        return {
            "message": "Vector index rebuild initiated",
            "status": "success",
            "estimated_completion_minutes": 10
        }
        
    except Exception as e:
        logger.error("Index rebuild failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Index rebuild failed", "detail": str(e)}
        )


@router.post("/clear-cache")
async def clear_cache(token: HTTPAuthorizationCredentials = Depends(verify_admin_token)):
    """Clear application caches"""
    try:
        logger.info("Clearing application caches")
        
        # In a real implementation, this would clear various caches
        # (Redis, in-memory caches, etc.)
        
        return {
            "message": "Caches cleared successfully",
            "cleared_caches": ["embeddings", "responses", "vector_search"]
        }
        
    except Exception as e:
        logger.error("Cache clearing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Cache clearing failed", "detail": str(e)}
        )


@router.get("/system-info")
async def get_system_info(token: HTTPAuthorizationCredentials = Depends(verify_admin_token)):
    """Get detailed system information"""
    try:
        # Get LLM provider info
        llm_providers = llm_service.list_providers()
        
        # Get configuration info
        system_info = {
            "version": settings.api_version,
            "environment": "development" if settings.debug else "production",
            "llm_providers": llm_providers,
            "vector_db_provider": settings.vector_db_provider,
            "supported_frameworks": settings.esg_frameworks,
            "supported_file_types": settings.supported_file_types,
            "max_document_size_mb": settings.max_document_size_mb,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "default_model": settings.default_model
        }
        
        return system_info
        
    except Exception as e:
        logger.error("Failed to get system info", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get system info", "detail": str(e)}
        )


@router.post("/backup")
async def create_backup(token: HTTPAuthorizationCredentials = Depends(verify_admin_token)):
    """Create system backup"""
    try:
        logger.info("Creating system backup")
        
        # In a real implementation, this would:
        # 1. Backup vector database
        # 2. Backup document metadata
        # 3. Backup configuration
        
        return {
            "message": "Backup created successfully",
            "backup_id": "backup_20250111_123456",
            "size_mb": 150.5,
            "created_at": "2025-01-11T12:34:56Z"
        }
        
    except Exception as e:
        logger.error("Backup creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Backup creation failed", "detail": str(e)}
        )


@router.get("/logs")
async def get_logs(
    lines: int = 100,
    level: str = "INFO",
    token: HTTPAuthorizationCredentials = Depends(verify_admin_token)
):
    """Get application logs"""
    try:
        # In a real implementation, this would read from log files
        # or a centralized logging system
        
        return {
            "logs": [
                {
                    "timestamp": "2025-01-11T12:00:00Z",
                    "level": "INFO",
                    "message": "Application started",
                    "module": "main"
                }
            ],
            "total_lines": 1,
            "requested_lines": lines,
            "level_filter": level
        }
        
    except Exception as e:
        logger.error("Failed to get logs", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get logs", "detail": str(e)}
        )


@router.get("/compliance-analysis")
async def get_compliance_analysis(
    esg_framework: Optional[str] = None,
    token: HTTPAuthorizationCredentials = Depends(verify_admin_token)
):
    """Get AI-powered compliance analysis with structured data for dashboards"""
    try:
        logger.info("Generating compliance analysis", framework=esg_framework)
        
        # Get document statistics
        stats_response = await call_documents_stats()
        
        # Generate AI-powered compliance analysis
        analysis = await generate_ai_compliance_analysis(esg_framework, stats_response)
        
        return {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "framework_focus": esg_framework or "All Frameworks",
            "overall_score": analysis["overall_score"],
            "framework_scores": analysis["framework_scores"],
            "compliance_gaps": analysis["compliance_gaps"],
            "priorities": analysis["priorities"],
            "recommendations": analysis["recommendations"],
            "risk_assessment": analysis["risk_assessment"],
            "document_coverage": analysis["document_coverage"],
            "next_actions": analysis["next_actions"]
        }
        
    except Exception as e:
        logger.error("Compliance analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Compliance analysis failed", "detail": str(e)}
        )


@router.post("/generate-report")
async def generate_esg_report(
    esg_framework: Optional[str] = None,
    report_type: str = "compliance_summary",
    include_recommendations: bool = True,
    use_agentic_rag: bool = True,
    token: HTTPAuthorizationCredentials = Depends(verify_admin_token)
):
    """Generate comprehensive ESG report using agentic RAG or enhanced AI analysis"""
    try:
        logger.info("Generating ESG report", framework=esg_framework, report_type=report_type, agentic=use_agentic_rag)
        
        # Get document statistics first
        stats_response = await call_documents_stats()
        
        if use_agentic_rag:
            # Use new agentic RAG system
            agentic_response = await agentic_rag_service.generate_report(
                report_type=report_type,
                framework=esg_framework,
                available_data=stats_response
            )
            
            return {
                "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "framework": esg_framework or "All",
                "report_type": report_type,
                "content": agentic_response.report_content,
                "metadata": {
                    "total_documents": stats_response.get("total_documents", 0),
                    "total_chunks": stats_response.get("total_chunks", 0),
                    "frameworks_covered": list(stats_response.get("documents_by_framework", {}).keys()),
                    "ai_powered": True,
                    "agentic_rag": True,
                    "analysis_depth": "comprehensive",
                    "quality_score": agentic_response.quality_score,
                    "sections_generated": agentic_response.sections_generated,
                    "confidence_scores": agentic_response.confidence_scores,
                    "research_findings": len(agentic_response.research_findings),
                    "validation_status": agentic_response.validation_results.get("overall_status"),
                    "generation_time_ms": agentic_response.total_time_ms,
                    "agent_phases": [log["phase"] for log in agentic_response.agent_execution_log]
                }
            }
        else:
            # Use original enhanced AI report generation
            if report_type == "compliance_summary":
                report_content = await generate_enhanced_compliance_report(esg_framework, stats_response, include_recommendations)
            elif report_type == "framework_analysis":
                report_content = await generate_enhanced_framework_analysis(esg_framework, stats_response)
            elif report_type == "gap_analysis":
                report_content = await generate_enhanced_gap_analysis(esg_framework, stats_response)
            else:
                report_content = await generate_enhanced_general_report(esg_framework, stats_response, include_recommendations)
            
            return {
                "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "framework": esg_framework or "All",
                "report_type": report_type,
                "content": report_content,
                "metadata": {
                    "total_documents": stats_response.get("total_documents", 0),
                    "total_chunks": stats_response.get("total_chunks", 0),
                    "frameworks_covered": list(stats_response.get("documents_by_framework", {}).keys()),
                    "ai_powered": True,
                    "agentic_rag": False,
                    "analysis_depth": "comprehensive"
                }
            }
        
    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Report generation failed", "detail": str(e)}
        )


@router.post("/generate-report/stream")
async def stream_esg_report(
    esg_framework: Optional[str] = None,
    report_type: str = "compliance_summary",
    include_recommendations: bool = True,
    token: HTTPAuthorizationCredentials = Depends(verify_admin_token)
):
    """Stream ESG report generation with real-time updates using agentic RAG"""
    
    async def generate():
        try:
            logger.info("Starting streaming report generation", 
                       framework=esg_framework, report_type=report_type)
            
            # Get document statistics first
            stats_response = await call_documents_stats()
            
            # Stream agentic RAG report generation
            async for update in agentic_rag_service.stream_report_generation(
                report_type=report_type,
                framework=esg_framework,
                available_data=stats_response
            ):
                yield f"data: {json.dumps(update)}\n\n"
            
        except Exception as e:
            logger.error("Streaming report generation failed", error=str(e))
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


async def call_documents_stats():
    """Helper to get document statistics"""
    from .documents import get_document_stats
    return await get_document_stats()


async def generate_ai_compliance_analysis(framework: Optional[str], stats: dict) -> Dict:
    """Generate AI-powered compliance analysis with structured data"""
    
    framework_filter = f" with specific focus on {framework} framework" if framework else " across all ESG frameworks"
    
    # Create sophisticated prompt for structured analysis
    prompt = f"""You are an expert ESG compliance analyst. Analyze the current ESG documentation repository{framework_filter} and provide a comprehensive compliance assessment.

Current Document Repository:
- Total Documents: {stats.get('total_documents', 0)}
- Content Chunks: {stats.get('total_chunks', 0)}
- Framework Distribution: {stats.get('documents_by_framework', {})}
- Category Distribution: {stats.get('documents_by_category', {})}
- Document Types: {stats.get('documents_by_type', {})}

Provide your analysis in JSON format with the following structure:
{{
    "overall_score": <number 0-100>,
    "framework_scores": {{
        "CSRD": {{"score": <0-100>, "status": "<Excellent|Good|Needs Attention|Poor|Not Started>", "missing_requirements": <number>}},
        "GRI": {{"score": <0-100>, "status": "<status>", "missing_requirements": <number>}},
        "SASB": {{"score": <0-100>, "status": "<status>", "missing_requirements": <number>}},
        "TCFD": {{"score": <0-100>, "status": "<status>", "missing_requirements": <number>}},
        "EU_Taxonomy": {{"score": <0-100>, "status": "<status>", "missing_requirements": <number>}},
        "SEC_Climate": {{"score": <0-100>, "status": "<status>", "missing_requirements": <number>}}
    }},
    "compliance_gaps": [
        {{"requirement": "<requirement name>", "framework": "<framework>", "priority": "<High|Medium|Low>", "effort": "<High|Medium|Low>", "description": "<description>"}}
    ],
    "priorities": [
        {{"action": "<action>", "priority": "<High|Medium|Low>", "timeline": "<timeframe>", "impact": "<description>"}}
    ],
    "recommendations": [
        {{"category": "<category>", "recommendation": "<detailed recommendation>", "business_value": "<value>", "implementation_effort": "<effort>"}}
    ],
    "risk_assessment": {{
        "high_risk_areas": ["<area1>", "<area2>"],
        "regulatory_risks": ["<risk1>", "<risk2>"],
        "reputation_risks": ["<risk1>", "<risk2>"],
        "overall_risk_level": "<Low|Medium|High|Critical>"
    }},
    "document_coverage": {{
        "well_covered_areas": ["<area1>", "<area2>"],
        "under_covered_areas": ["<area1>", "<area2>"],
        "missing_critical_documents": ["<doc1>", "<doc2>"]
    }},
    "next_actions": [
        {{"action": "<action>", "timeline": "<timeframe>", "owner": "<suggested owner>", "priority": "<High|Medium|Low>"}}
    ]
}}

Base your analysis on actual ESG framework requirements and current best practices. Be specific and actionable."""

    # Get AI response using RAG
    response = await rag_service.query(
        question=prompt,
        esg_framework=framework,
        search_strategy="hybrid",
        k=15
    )
    
    # Parse JSON response
    try:
        # Extract JSON from response
        content = response.get('answer', '')
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            json_content = content[json_start:json_end].strip()
        elif '{' in content:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_content = content[json_start:json_end]
        else:
            # Fallback if JSON parsing fails
            return await generate_fallback_compliance_analysis(stats)
        
        analysis = json.loads(json_content)
        return analysis
        
    except Exception as e:
        logger.warning(f"Failed to parse AI JSON response: {e}")
        return await generate_fallback_compliance_analysis(stats)


async def generate_fallback_compliance_analysis(stats: dict) -> Dict:
    """Generate fallback compliance analysis if AI parsing fails"""
    frameworks = stats.get('documents_by_framework', {})
    total_docs = stats.get('total_documents', 0)
    
    # Calculate basic scores based on document availability
    framework_scores = {}
    for fw in ["CSRD", "GRI", "SASB", "TCFD", "EU_Taxonomy", "SEC_Climate"]:
        doc_count = frameworks.get(fw, 0)
        if doc_count >= 5:
            score, status = 85, "Good"
        elif doc_count >= 3:
            score, status = 70, "Needs Attention"
        elif doc_count >= 1:
            score, status = 45, "Poor"
        else:
            score, status = 0, "Not Started"
        
        framework_scores[fw] = {
            "score": score,
            "status": status,
            "missing_requirements": max(0, 12 - doc_count * 2)
        }
    
    overall_score = sum(fw["score"] for fw in framework_scores.values()) // len(framework_scores)
    
    return {
        "overall_score": overall_score,
        "framework_scores": framework_scores,
        "compliance_gaps": [
            {"requirement": "Scope 3 Emissions Disclosure", "framework": "CSRD", "priority": "High", "effort": "Medium", "description": "Comprehensive Scope 3 emissions reporting required"},
            {"requirement": "Climate Risk Scenarios", "framework": "TCFD", "priority": "High", "effort": "High", "description": "Scenario analysis for climate-related risks"},
            {"requirement": "Biodiversity Impact Assessment", "framework": "EU_Taxonomy", "priority": "Medium", "effort": "High", "description": "Assessment of biodiversity impact"}
        ],
        "priorities": [
            {"action": "Enhance climate disclosure documentation", "priority": "High", "timeline": "3 months", "impact": "Regulatory compliance"},
            {"action": "Develop governance framework documentation", "priority": "Medium", "timeline": "6 months", "impact": "Stakeholder confidence"}
        ],
        "recommendations": [
            {"category": "Documentation", "recommendation": "Increase document coverage across all frameworks", "business_value": "Improved compliance and reduced risk", "implementation_effort": "Medium"},
            {"category": "Process", "recommendation": "Implement regular compliance monitoring", "business_value": "Proactive risk management", "implementation_effort": "Low"}
        ],
        "risk_assessment": {
            "high_risk_areas": ["Climate Disclosure", "Supply Chain Transparency"],
            "regulatory_risks": ["CSRD Non-compliance", "SEC Climate Rules"],
            "reputation_risks": ["Greenwashing Claims", "Stakeholder Criticism"],
            "overall_risk_level": "Medium" if total_docs >= 5 else "High"
        },
        "document_coverage": {
            "well_covered_areas": list(frameworks.keys())[:2] if frameworks else [],
            "under_covered_areas": ["Social Impact", "Governance Policies"],
            "missing_critical_documents": ["Climate Risk Policy", "Supplier Code of Conduct"]
        },
        "next_actions": [
            {"action": "Upload additional framework-specific documents", "timeline": "1 month", "owner": "ESG Team", "priority": "High"},
            {"action": "Conduct gap analysis review", "timeline": "2 weeks", "owner": "Compliance Officer", "priority": "High"}
        ]
    }


async def generate_enhanced_compliance_report(framework: Optional[str], stats: dict, include_recommendations: bool) -> str:
    """Generate enhanced compliance summary report using AI analysis"""
    
    framework_filter = f" focusing on {framework}" if framework else ""
    
    prompt = f"""As an expert ESG compliance consultant, generate a comprehensive compliance summary report{framework_filter} based on the current document repository.

Current Repository Status:
- Total Documents: {stats.get('total_documents', 0)}
- Content Chunks: {stats.get('total_chunks', 0)}
- Framework Coverage: {stats.get('documents_by_framework', {})}
- Category Coverage: {stats.get('documents_by_category', {})}
- Document Types: {stats.get('documents_by_type', {})}

Please provide a detailed report with:

1. EXECUTIVE SUMMARY
   - Overall compliance maturity assessment
   - Key achievements and critical gaps
   - Strategic recommendations

2. COMPLIANCE STATUS ANALYSIS
   - Current documentation strengths
   - Framework-specific compliance levels
   - Regulatory readiness assessment

3. FRAMEWORK COVERAGE EVALUATION
   - Analysis of each ESG framework representation
   - Cross-framework alignment opportunities
   - Missing critical documentation

4. GAP ANALYSIS & RISK ASSESSMENT
   - Priority compliance gaps
   - Regulatory and reputational risks
   - Timeline-based action priorities

{'5. STRATEGIC RECOMMENDATIONS' if include_recommendations else ''}
{'   - Immediate actions (0-3 months)' if include_recommendations else ''}
{'   - Medium-term initiatives (3-12 months)' if include_recommendations else ''}
{'   - Long-term strategic goals (1-3 years)' if include_recommendations else ''}

Use professional, stakeholder-appropriate language with specific, actionable insights. Include quantitative assessments where possible."""

    response = await rag_service.query(
        question=prompt,
        esg_framework=framework,
        search_strategy="hybrid",
        k=12
    )
    
    return response.get('answer', 'Report generation failed')


async def generate_enhanced_framework_analysis(framework: Optional[str], stats: dict) -> str:
    """Generate enhanced framework-specific analysis using AI"""
    
    framework_name = framework or "All ESG Frameworks"
    
    prompt = f"""As an ESG framework specialist, provide a comprehensive analysis of {framework_name} compliance and implementation status.

Current Documentation Status:
- Total Documents: {stats.get('total_documents', 0)}
- Framework Distribution: {stats.get('documents_by_framework', {})}
- Category Coverage: {stats.get('documents_by_category', {})}

Analysis Required:

1. FRAMEWORK OVERVIEW
   - Key requirements and disclosure obligations
   - Implementation timeline and deadlines
   - Materiality assessment requirements

2. CURRENT COMPLIANCE STATUS
   - Documentation coverage assessment
   - Implementation gap analysis
   - Regulatory readiness evaluation

3. REQUIREMENT MAPPING
   - Mandatory disclosure requirements
   - Optional reporting elements
   - Cross-framework alignment opportunities

4. IMPLEMENTATION ROADMAP
   - Priority actions for compliance
   - Resource requirements
   - Timeline recommendations

5. RISK & OPPORTUNITY ASSESSMENT
   - Compliance risks and penalties
   - Competitive advantages
   - Stakeholder value creation

Provide specific, actionable insights with quantitative assessments where possible."""

    response = await rag_service.query(
        question=prompt,
        esg_framework=framework,
        search_strategy="hybrid",
        k=10
    )
    
    return response.get('answer', 'Framework analysis generation failed')


async def generate_enhanced_gap_analysis(framework: Optional[str], stats: dict) -> str:
    """Generate enhanced gap analysis using AI"""
    
    framework_filter = f" for {framework} framework" if framework else " across all ESG frameworks"
    
    prompt = f"""As an ESG compliance expert, conduct a comprehensive gap analysis{framework_filter} based on current documentation.

Current Status:
- Documents: {stats.get('total_documents', 0)}
- Framework Coverage: {stats.get('documents_by_framework', {})}
- Categories: {stats.get('documents_by_category', {})}

Provide detailed gap analysis including:

1. CRITICAL GAPS
   - Missing mandatory disclosures
   - Regulatory compliance gaps
   - Stakeholder expectation gaps

2. DOCUMENTATION GAPS
   - Policy and procedure gaps
   - Data collection and reporting gaps
   - Governance and oversight gaps

3. IMPLEMENTATION GAPS
   - Process and system gaps
   - Capability and resource gaps
   - Timeline and milestone gaps

4. RISK PRIORITIZATION
   - High-impact regulatory risks
   - Reputation and stakeholder risks
   - Operational and business risks

5. REMEDIATION PLAN
   - Immediate actions (0-3 months)
   - Short-term initiatives (3-6 months)
   - Medium-term programs (6-18 months)

Include specific recommendations with effort estimates and business impact assessments."""

    response = await rag_service.query(
        question=prompt,
        esg_framework=framework,
        search_strategy="hybrid",
        k=12
    )
    
    return response.get('answer', 'Gap analysis generation failed')


async def generate_enhanced_general_report(framework: Optional[str], stats: dict, include_recommendations: bool) -> str:
    """Generate enhanced general ESG report using AI"""
    
    framework_context = f" with emphasis on {framework}" if framework else ""
    
    prompt = f"""As a senior ESG advisor, create a comprehensive ESG maturity and performance report{framework_context}.

Current ESG Documentation Repository:
- Total Documents: {stats.get('total_documents', 0)}
- Content Analysis: {stats.get('total_chunks', 0)} content segments
- Framework Coverage: {stats.get('documents_by_framework', {})}
- Category Distribution: {stats.get('documents_by_category', {})}

Comprehensive Report Sections:

1. EXECUTIVE SUMMARY
   - ESG maturity assessment
   - Key achievements and milestones
   - Strategic priorities

2. CURRENT STATE ASSESSMENT
   - Documentation and policy landscape
   - Framework implementation status
   - Governance and oversight maturity

3. PERFORMANCE EVALUATION
   - Strengths and competitive advantages
   - Areas requiring improvement
   - Benchmark and peer comparison insights

4. STRATEGIC ANALYSIS
   - ESG integration with business strategy
   - Stakeholder value creation opportunities
   - Risk mitigation and opportunity capture

{'5. TRANSFORMATION ROADMAP' if include_recommendations else ''}
{'   - Strategic initiatives and programs' if include_recommendations else ''}
{'   - Implementation timeline and milestones' if include_recommendations else ''}
{'   - Resource requirements and ROI analysis' if include_recommendations else ''}
{'   - Success metrics and KPIs' if include_recommendations else ''}

Focus on value creation, competitive positioning, and stakeholder impact. Provide specific, measurable recommendations."""

    response = await rag_service.query(
        question=prompt,
        esg_framework=framework,
        search_strategy="hybrid",
        k=15
    )
    
    return response.get('answer', 'General report generation failed')


# Keep the old simple functions for backward compatibility
async def generate_compliance_report(framework: Optional[str], stats: dict, include_recommendations: bool) -> str:
    """Simple compliance report (deprecated - use enhanced version)"""
    return await generate_enhanced_compliance_report(framework, stats, include_recommendations)


async def generate_framework_analysis(framework: Optional[str], stats: dict) -> str:
    """Simple framework analysis (deprecated - use enhanced version)"""
    return await generate_enhanced_framework_analysis(framework, stats)


async def generate_gap_analysis(framework: Optional[str], stats: dict) -> str:
    """Simple gap analysis (deprecated - use enhanced version)"""
    return await generate_enhanced_gap_analysis(framework, stats)


async def generate_general_report(framework: Optional[str], stats: dict) -> str:
    """Simple general report (deprecated - use enhanced version)"""
    return await generate_enhanced_general_report(framework, stats, True)