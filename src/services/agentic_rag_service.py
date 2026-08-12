"""
Agentic RAG Service implementing intelligent, autonomous report generation
with planning, research, validation, and synthesis capabilities.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from .llm_service import llm_service
from .rag_service import rag_service
from ..core.logging import LoggingMixin
from ..core.exceptions import ESGAnalysisException


class AgentType(Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"
    QUALITY_ASSURANCE = "qa"


@dataclass
class AgentTask:
    """Represents a task for an agent"""

    task_id: str
    agent_type: AgentType
    description: str
    context: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReportPlan:
    """Structured plan for report generation"""

    report_type: str
    target_framework: Optional[str]
    structure: List[Dict[str, Any]]
    research_questions: List[str]
    validation_checks: List[str]
    quality_criteria: List[str]
    estimated_sections: int
    data_requirements: Dict[str, Any]


@dataclass
class ResearchFinding:
    """Represents a research finding with confidence and sources"""

    topic: str
    finding: str
    confidence_score: float
    sources: List[Document]
    supporting_evidence: List[str]
    gaps_identified: List[str]
    follow_up_questions: List[str]


@dataclass
class AgenticRAGResponse:
    """Response from agentic RAG system with detailed metadata"""

    report_content: str
    plan: ReportPlan
    research_findings: List[ResearchFinding]
    validation_results: Dict[str, Any]
    quality_score: float
    agent_execution_log: List[Dict[str, Any]]
    total_time_ms: int
    sections_generated: int
    confidence_scores: Dict[str, float]


class PlannerAgent(LoggingMixin):
    """Agent responsible for analyzing requirements and creating report plans"""

    def __init__(self):
        self.planning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert ESG report planning agent. Your role is to analyze the available data and requirements to create a comprehensive, structured plan for report generation.

Given the context about available documents, frameworks, and report requirements, create a detailed plan that:
1. Defines the optimal report structure based on available data
2. Identifies key research questions that need investigation
3. Determines validation checks needed for accuracy
4. Sets quality criteria for the final report

You must respond with a JSON object containing:
- "structure": Array of sections with titles, objectives, and required information
- "research_questions": Array of specific questions that need investigation
- "validation_checks": Array of fact-checking and validation requirements
- "quality_criteria": Array of quality standards the report must meet
- "data_requirements": Object describing what data is needed for each section
- "estimated_complexity": String indicating complexity level (low/medium/high)
- "reasoning": Your reasoning for this plan

Be adaptive - the plan should match the available data and be realistic about what can be accomplished.""",
                ),
                ("human", "{requirements}"),
            ]
        )

    async def create_plan(
        self, report_type: str, framework: Optional[str], available_data: Dict[str, Any]
    ) -> ReportPlan:
        """Create a comprehensive plan for report generation"""

        self.logger.info("Creating report plan", report_type=report_type, framework=framework)

        requirements = f"""
        Report Type: {report_type}
        Target Framework: {framework or "All frameworks"}
        Available Data:
        - Total Documents: {available_data.get("total_documents", 0)}
        - Frameworks: {available_data.get("documents_by_framework", {})}
        - Categories: {available_data.get("documents_by_category", {})}
        - Document Types: {available_data.get("documents_by_type", {})}
        
        Create a plan that maximizes the value of available data while ensuring comprehensive coverage.
        """

        try:
            response = await llm_service.generate(
                self.planning_prompt.format(requirements=requirements)
            )

            # Parse the JSON response
            plan_data = json.loads(response.content)

            return ReportPlan(
                report_type=report_type,
                target_framework=framework,
                structure=plan_data.get("structure", []),
                research_questions=plan_data.get("research_questions", []),
                validation_checks=plan_data.get("validation_checks", []),
                quality_criteria=plan_data.get("quality_criteria", []),
                estimated_sections=len(plan_data.get("structure", [])),
                data_requirements=plan_data.get("data_requirements", {}),
            )

        except Exception as e:
            self.logger.error("Plan creation failed", error=str(e))
            # Fallback to basic plan
            return self._create_fallback_plan(report_type, framework, available_data)

    def _create_fallback_plan(
        self, report_type: str, framework: Optional[str], available_data: Dict[str, Any]
    ) -> ReportPlan:
        """Create a basic fallback plan if AI planning fails"""

        basic_structure = [
            {"title": "Executive Summary", "objective": "Provide high-level overview"},
            {"title": "Current State Analysis", "objective": "Analyze existing documentation"},
            {"title": "Gap Analysis", "objective": "Identify missing elements"},
            {"title": "Recommendations", "objective": "Provide actionable next steps"},
        ]

        return ReportPlan(
            report_type=report_type,
            target_framework=framework,
            structure=basic_structure,
            research_questions=["What is the current compliance status?"],
            validation_checks=["Check source accuracy"],
            quality_criteria=["Clear structure", "Actionable recommendations"],
            estimated_sections=4,
            data_requirements={"documents": "all", "frameworks": framework or "all"},
        )


class ResearchAgent(LoggingMixin):
    """Agent responsible for iterative research and information gathering"""

    def __init__(self):
        self.research_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert ESG research agent. Your role is to conduct thorough, iterative research on specific topics using available documents.

For each research question, you should:
1. Analyze the question and determine what information is needed
2. Formulate specific search queries
3. Evaluate the quality and completeness of retrieved information  
4. Identify gaps that require additional investigation
5. Generate follow-up questions for deeper investigation

Respond with JSON containing:
- "findings": Array of key findings with evidence
- "confidence_score": Float 0-1 indicating confidence in findings
- "evidence_quality": Assessment of source quality
- "gaps_identified": Array of information gaps found
- "follow_up_questions": Array of questions for further research
- "recommendations": Specific recommendations based on findings""",
                ),
                ("human", "{research_question}\n\nContext: {context}"),
            ]
        )

    async def research_topic(
        self, question: str, framework: Optional[str], max_iterations: int = 3
    ) -> ResearchFinding:
        """Conduct iterative research on a specific topic"""

        self.logger.info("Starting research", question=question, framework=framework)

        all_findings = []
        all_sources = []
        iteration_count = 0
        current_question = question
        analysis = {}  # Initialize analysis to avoid variable scope issues

        while iteration_count < max_iterations:
            iteration_count += 1
            self.logger.info(f"Research iteration {iteration_count}", question=current_question)

            # Search for relevant information
            rag_response = await rag_service.query(
                question=current_question, esg_framework=framework, search_strategy="hybrid", k=8
            )

            all_sources.extend(rag_response.source_documents)

            # Analyze findings and plan next iteration
            context = {
                "iteration": iteration_count,
                "previous_findings": all_findings,
                "current_sources": len(rag_response.source_documents),
                "answer": rag_response.answer,
            }

            analysis_response = await llm_service.generate(
                self.research_prompt.format(
                    research_question=current_question, context=json.dumps(context)
                )
            )

            try:
                analysis = json.loads(analysis_response.content)
                all_findings.extend(analysis.get("findings", []))

                follow_ups = analysis.get("follow_up_questions", [])
                if follow_ups and iteration_count < max_iterations:
                    # Use the first follow-up question for next iteration
                    current_question = follow_ups[0]
                else:
                    break

            except json.JSONDecodeError:
                self.logger.warning("Failed to parse research analysis", iteration=iteration_count)
                analysis = {"gaps_identified": [], "follow_up_questions": []}  # Fallback values
                break

        # Synthesize final findings
        try:
            gaps = analysis.get("gaps_identified", []) if analysis else []
            follow_ups = analysis.get("follow_up_questions", []) if analysis else []
        except Exception as e:
            self.logger.error(f"Error accessing analysis: {e}, analysis type: {type(analysis)}")
            gaps = []
            follow_ups = []

        return ResearchFinding(
            topic=question,
            finding=rag_response.answer,
            confidence_score=rag_response.confidence_score,
            sources=all_sources,
            supporting_evidence=all_findings,
            gaps_identified=gaps,
            follow_up_questions=follow_ups,
        )


class ValidationAgent(LoggingMixin):
    """Agent responsible for fact-checking and validation"""

    def __init__(self):
        self.validation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert ESG validation agent. Your role is to fact-check findings, ensure accuracy, and identify inconsistencies.

For each finding to validate, you should:
1. Cross-reference information across multiple sources
2. Check for consistency across the supplied repository sources
3. Identify any contradictions or gaps
4. Assess the reliability of sources
5. Flag areas that need additional verification

Respond with JSON containing:
- "validation_status": "validated" | "needs_review" | "contradictory" | "insufficient_data"
- "confidence_level": Float 0-1 indicating validation confidence
- "cross_references": Array of supporting/contradicting sources
- "inconsistencies": Array of identified inconsistencies
- "reliability_assessment": Assessment of source reliability
- "recommendations": Actions needed for full validation""",
                ),
                ("human", "Validate this finding:\n{finding}\n\nSources:\n{sources}"),
            ]
        )

    async def validate_finding(self, finding: ResearchFinding) -> Dict[str, Any]:
        """Validate a research finding through cross-referencing"""

        self.logger.info("Validating finding", topic=finding.topic)

        # Prepare source information for validation
        sources_text = "\n".join(
            [
                f"Source {i + 1}: {doc.page_content[:500]}..."
                for i, doc in enumerate(finding.sources[:5])
            ]
        )

        validation_response = await llm_service.generate(
            self.validation_prompt.format(finding=finding.finding, sources=sources_text)
        )

        try:
            validation_result = json.loads(validation_response.content)
            return validation_result
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse validation response")
            return {
                "validation_status": "needs_review",
                "confidence_level": 0.5,
                "cross_references": [],
                "inconsistencies": [],
                "reliability_assessment": "uncertain",
                "recommendations": ["Manual review required"],
            }


class SynthesisAgent(LoggingMixin):
    """Agent responsible for synthesizing findings into structured reports"""

    def __init__(self):
        self.synthesis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert ESG report synthesis agent. Your role is to combine research findings into well-structured, professional report sections.

You should:
1. Organize findings logically within each section structure
2. Ensure professional, stakeholder-appropriate language
3. Include specific citations and evidence
4. Maintain consistency in tone and terminology
5. Provide actionable insights and recommendations

Use only the supplied findings and sources. Preserve their source citations,
label uncertainty, and never turn missing repository evidence into a claim of
non-compliance.""",
                ),
                (
                    "human",
                    """
Section: {section_title}
Objective: {section_objective}
Research Findings: {findings}
Validation Results: {validation_results}
Quality Criteria: {quality_criteria}

Generate professional report content for this section.
""",
                ),
            ]
        )

    async def synthesize_section(
        self,
        section: Dict[str, Any],
        findings: List[ResearchFinding],
        validation_results: List[Dict[str, Any]],
    ) -> str:
        """Synthesize research findings into a report section"""

        self.logger.info("Synthesizing section", title=section.get("title"))

        # Prepare findings summary
        findings_text = "\n".join(
            [
                f"Topic: {f.topic}\nFinding: {f.finding}\nConfidence: {f.confidence_score}"
                for f in findings
            ]
        )

        validation_text = "\n".join(
            [
                f"Status: {v.get('validation_status')}, Confidence: {v.get('confidence_level')}"
                for v in validation_results
            ]
        )

        synthesis_response = await llm_service.generate(
            self.synthesis_prompt.format(
                section_title=section.get("title", "Section"),
                section_objective=section.get("objective", "Provide analysis"),
                findings=findings_text,
                validation_results=validation_text,
                quality_criteria="Professional, actionable, evidence-based",
            )
        )

        return synthesis_response.content


class AgenticRAGOrchestrator(LoggingMixin):
    """Main orchestrator for the agentic RAG system"""

    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.validator = ValidationAgent()
        self.synthesizer = SynthesisAgent()

    @traceable(name="agentic_rag_generate_report")
    async def generate_report(
        self,
        report_type: str,
        framework: Optional[str] = None,
        available_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AgenticRAGResponse:
        """Generate a comprehensive report using agentic RAG approach"""

        start_time = datetime.now()
        execution_log = []

        self.logger.info(
            "Starting agentic report generation", report_type=report_type, framework=framework
        )

        # Default data if not provided
        if available_data is None:
            available_data = {"total_documents": 0, "documents_by_framework": {}}

        try:
            # Phase 1: Planning
            execution_log.append({"phase": "planning", "timestamp": datetime.now().isoformat()})
            plan = await self.planner.create_plan(report_type, framework, available_data)
            self.logger.info("Plan created", sections=plan.estimated_sections)

            # Phase 2: Research
            execution_log.append({"phase": "research", "timestamp": datetime.now().isoformat()})
            research_findings = []

            for question in plan.research_questions:
                finding = await self.researcher.research_topic(question, framework)
                research_findings.append(finding)
                self.logger.info(
                    "Research completed", topic=question, confidence=finding.confidence_score
                )

            # Phase 3: Validation
            execution_log.append({"phase": "validation", "timestamp": datetime.now().isoformat()})
            validation_results = []

            for finding in research_findings:
                validation = await self.validator.validate_finding(finding)
                validation_results.append(validation)
                self.logger.info("Validation completed", status=validation.get("validation_status"))

            # Phase 4: Synthesis
            execution_log.append({"phase": "synthesis", "timestamp": datetime.now().isoformat()})
            report_sections = []

            for section in plan.structure:
                section_content = await self.synthesizer.synthesize_section(
                    section, research_findings, validation_results
                )
                report_sections.append(f"## {section.get('title', 'Section')}\n\n{section_content}")

            # Combine all sections
            report_content = "\n\n".join(report_sections)

            # Calculate metrics
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            quality_score = self._calculate_quality_score(validation_results, research_findings)

            # Generate confidence scores by section
            confidence_scores = {
                section.get("title", f"Section {i + 1}"): research_findings[i].confidence_score
                if i < len(research_findings)
                else 0.5
                for i, section in enumerate(plan.structure)
            }

            execution_log.append({"phase": "completed", "timestamp": datetime.now().isoformat()})

            return AgenticRAGResponse(
                report_content=report_content,
                plan=plan,
                research_findings=research_findings,
                validation_results={
                    "validations": validation_results,
                    "overall_status": "validated"
                    if all(v.get("validation_status") == "validated" for v in validation_results)
                    else "needs_review",
                },
                quality_score=quality_score,
                agent_execution_log=execution_log,
                total_time_ms=int(total_time),
                sections_generated=len(report_sections),
                confidence_scores=confidence_scores,
            )

        except Exception as e:
            self.logger.error("Agentic report generation failed", error=str(e))
            raise ESGAnalysisException("Agentic RAG failed", error_code="AGENTIC_RAG_FAILED") from e

    def _calculate_quality_score(
        self, validation_results: List[Dict[str, Any]], research_findings: List[ResearchFinding]
    ) -> float:
        """Calculate overall quality score for the report"""

        if not validation_results or not research_findings:
            return 0.5

        # Average validation confidence
        validation_confidence = sum(
            v.get("confidence_level", 0.5) for v in validation_results
        ) / len(validation_results)

        # Average research confidence
        research_confidence = sum(f.confidence_score for f in research_findings) / len(
            research_findings
        )

        # Weighted average (validation is more important)
        quality_score = (validation_confidence * 0.6) + (research_confidence * 0.4)

        return min(max(quality_score, 0.0), 1.0)

    async def stream_report_generation(
        self,
        report_type: str,
        framework: Optional[str] = None,
        available_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream the report generation process with real-time updates"""

        yield {"status": "starting", "phase": "initialization"}

        # Default data if not provided
        if available_data is None:
            available_data = {"total_documents": 0, "documents_by_framework": {}}

        try:
            # Phase 1: Planning
            yield {"status": "progress", "phase": "planning", "message": "Creating report plan..."}
            plan = await self.planner.create_plan(report_type, framework, available_data)
            yield {
                "status": "progress",
                "phase": "planning",
                "message": f"Plan created with {plan.estimated_sections} sections",
            }

            # Phase 2: Research
            research_findings = []
            for i, question in enumerate(plan.research_questions):
                yield {
                    "status": "progress",
                    "phase": "research",
                    "message": f"Researching question {i + 1}/{len(plan.research_questions)}: {question[:50]}...",
                }
                finding = await self.researcher.research_topic(question, framework)
                research_findings.append(finding)
                yield {
                    "status": "progress",
                    "phase": "research",
                    "message": f"Research completed with confidence: {finding.confidence_score:.2f}",
                }

            # Phase 3: Validation
            validation_results = []
            for i, finding in enumerate(research_findings):
                yield {
                    "status": "progress",
                    "phase": "validation",
                    "message": f"Validating finding {i + 1}/{len(research_findings)}...",
                }
                validation = await self.validator.validate_finding(finding)
                validation_results.append(validation)
                yield {
                    "status": "progress",
                    "phase": "validation",
                    "message": f"Validation: {validation.get('validation_status')}",
                }

            # Phase 4: Synthesis
            report_sections = []
            for i, section in enumerate(plan.structure):
                yield {
                    "status": "progress",
                    "phase": "synthesis",
                    "message": f"Generating section {i + 1}/{len(plan.structure)}: {section.get('title')}",
                }
                section_content = await self.synthesizer.synthesize_section(
                    section, research_findings, validation_results
                )
                report_sections.append(f"## {section.get('title', 'Section')}\n\n{section_content}")
                yield {
                    "status": "progress",
                    "phase": "synthesis",
                    "message": f"Section '{section.get('title')}' completed",
                }

            # Final report
            report_content = "\n\n".join(report_sections)
            quality_score = self._calculate_quality_score(validation_results, research_findings)

            yield {
                "status": "completed",
                "phase": "finished",
                "report_content": report_content,
                "quality_score": quality_score,
                "sections_generated": len(report_sections),
                "message": "Report generation completed successfully",
            }

        except Exception as e:
            yield {"status": "error", "phase": "error", "message": f"Generation failed: {str(e)}"}


# Global agentic RAG service instance
agentic_rag_service = AgenticRAGOrchestrator()
