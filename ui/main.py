"""Streamlit frontend for the ESG Analysis API."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.api_client import APIError, api_client
from ui.config import ui_settings
from ui.reporting import markdown_to_pdf


st.set_page_config(
    page_title="ESG Evidence Studio",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1400px; padding-top: 1.5rem;}
      [data-testid="stMetric"] {background: rgba(46, 125, 50, .06); border: 1px solid rgba(46, 125, 50, .16); padding: .9rem; border-radius: .75rem;}
      .eyebrow {letter-spacing: .08em; text-transform: uppercase; color: #2e7d32; font-weight: 700; font-size: .8rem;}
      .muted {color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(
    endpoint: str,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    show_error: bool = True,
) -> Any | None:
    try:
        return api_client.request(endpoint, method=method, data=data, files=files)
    except APIError as exc:
        if show_error:
            st.error(exc.user_message)
        return None


@st.cache_data(ttl=60, show_spinner=False)
def platform_info() -> dict[str, Any]:
    return api_request("/", show_error=False) or {}


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict[str, Any]:
    return api_request("/health", show_error=False) or {}


@st.cache_data(ttl=10, show_spinner=False)
def document_stats() -> dict[str, Any]:
    return api_request("/api/v1/documents/stats", show_error=False) or {}


def framework_catalog() -> list[dict[str, str]]:
    return platform_info().get("frameworks", [])


def framework_ids() -> list[str]:
    return [framework["id"] for framework in framework_catalog()]


def clear_data_caches() -> None:
    health.clear()
    document_stats.clear()
    platform_info.clear()


def iter_sse(response):
    event_name = "message"
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            event_name = "message"
            continue
        if line.startswith("event: "):
            event_name = line[7:]
        elif line.startswith("data: "):
            try:
                yield event_name, json.loads(line[6:])
            except json.JSONDecodeError:
                continue


def render_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">ESG Evidence Studio</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="muted">{subtitle}</div>', unsafe_allow_html=True)


def render_source(source: dict[str, Any], index: int) -> None:
    metadata = source.get("metadata", {})
    score = source.get("retrieval_score")
    label = f"{index}. {metadata.get('filename', 'Unknown source')}"
    if score is not None:
        label += f" · relevance {score:.2f}"
    with st.expander(label):
        st.caption(
            " · ".join(
                value
                for value in [
                    metadata.get("esg_framework"),
                    metadata.get("esg_category"),
                    metadata.get("document_type"),
                ]
                if value
            )
            or "No additional metadata"
        )
        st.write(source.get("content", ""))


def home_page() -> None:
    render_header(
        "Evidence-first ESG analysis",
        "Search your own source documents, inspect citations, and generate traceable reports.",
    )
    status = health()
    stats = document_stats()
    services = status.get("services", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", stats.get("total_documents", 0))
    col2.metric("Indexed chunks", stats.get("total_chunks", 0))
    col3.metric("Frameworks represented", len(stats.get("documents_by_framework", {})))
    col4.metric("API status", status.get("status", "offline").title())

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Repository coverage")
        coverage = stats.get("documents_by_framework", {})
        if coverage:
            frame = pd.DataFrame(
                [{"Framework": key, "Documents": value} for key, value in coverage.items()]
            )
            figure = px.bar(
                frame,
                x="Framework",
                y="Documents",
                color="Documents",
                color_continuous_scale="Greens",
            )
            figure.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Upload documents to build repository coverage.")
    with right:
        st.subheader("Service readiness")
        if not services:
            st.error(f"API unavailable at {ui_settings.esg_api_url}")
        for service, service_status in services.items():
            icon = "✅" if service_status == "healthy" else "⚠️"
            st.write(f"{icon} {service.replace('_', ' ').title()}: {service_status}")
        if services.get("llm_service") == "not_configured":
            st.caption(
                "Document upload and lexical search work without an LLM. Configure a provider key to enable answers and reports."
            )

    st.subheader("Configured frameworks")
    catalog = framework_catalog()
    if catalog:
        columns = st.columns(min(3, len(catalog)))
        for index, framework in enumerate(catalog):
            with columns[index % len(columns)]:
                st.markdown(f"**{framework['label']}**")
                st.caption(f"{framework['region']} · {framework['type']}")
    else:
        st.warning("The framework catalog could not be loaded from the API.")


def query_page(selected_framework: str) -> None:
    render_header(
        "Ask the repository",
        "Answers are generated from retrieved chunks and include inspectable sources.",
    )
    with st.form("query_form"):
        question = st.text_area(
            "Question",
            placeholder="What evidence in the uploaded documents supports the climate-risk disclosure?",
            height=120,
        )
        col1, col2, col3 = st.columns(3)
        framework_options = ["All"] + framework_ids()
        default_index = (
            framework_options.index(selected_framework)
            if selected_framework in framework_options
            else 0
        )
        framework = col1.selectbox("Framework", framework_options, index=default_index)
        strategy = col2.selectbox("Retrieval", ["hybrid", "similarity"])
        result_count = col3.slider("Source chunks", 1, 20, 5)
        streaming = st.checkbox("Stream the answer", value=True)
        submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)

    if not submitted:
        return
    if len(question.strip()) < 2:
        st.warning("Enter a question first.")
        return

    payload = {
        "question": question.strip(),
        "search_strategy": strategy,
        "k": result_count,
        "esg_framework": None if framework == "All" else framework,
    }
    if streaming:
        answer_box = st.empty()
        answer = ""
        try:
            response = api_client.stream("/api/v1/query/stream", payload)
            for event, event_payload in iter_sse(response):
                if event == "chunk":
                    answer += event_payload.get("chunk", "")
                    answer_box.markdown(answer)
                elif event == "error":
                    raise APIError(event_payload.get("error", "Streaming query failed"))
            if answer:
                st.download_button(
                    "Download answer as PDF",
                    markdown_to_pdf(answer, "ESG Analysis"),
                    "esg-analysis.pdf",
                    "application/pdf",
                )
        except APIError as exc:
            st.error(exc.user_message)
        return

    response = api_request("/api/v1/query", method="POST", data=payload)
    if not response:
        return
    st.markdown(response["answer"])
    metrics = st.columns(3)
    metrics[0].metric("Retrieval confidence", f"{response['confidence_score']:.0%}")
    metrics[1].metric("Sources", len(response.get("source_documents", [])))
    metrics[2].metric("Total time", f"{response['total_time_ms'] / 1000:.1f}s")
    st.download_button(
        "Download answer as PDF",
        markdown_to_pdf(response["answer"], "ESG Analysis"),
        "esg-analysis.pdf",
        "application/pdf",
    )
    st.subheader("Sources")
    for index, source in enumerate(response.get("source_documents", []), start=1):
        render_source(source, index)


def documents_page(selected_framework: str) -> None:
    render_header(
        "Document workspace",
        "Upload, search, inspect, and remove documents from the configured store.",
    )
    upload_tab, library_tab, search_tab = st.tabs(["Upload", "Library", "Search"])

    with upload_tab:
        files = st.file_uploader(
            "Source documents",
            type=[
                extension.lstrip(".")
                for extension in platform_info().get("supported_extensions", [])
            ],
            accept_multiple_files=True,
        )
        col1, col2, col3 = st.columns(3)
        options = [""] + framework_ids()
        default = options.index(selected_framework) if selected_framework in options else 0
        framework = col1.selectbox("Framework metadata", options, index=default)
        document_type = col2.selectbox(
            "Document type", [""] + platform_info().get("document_types", [])
        )
        company_id = col3.text_input("Company ID (optional)")
        if st.button("Upload and index", type="primary", disabled=not files):
            progress = st.progress(0)
            successful = 0
            for index, uploaded_file in enumerate(files or [], start=1):
                form = {
                    "esg_framework": framework or None,
                    "document_type": document_type or None,
                    "company_id": company_id.strip() or None,
                }
                response = api_request(
                    "/api/v1/upload",
                    method="POST",
                    data={key: value for key, value in form.items() if value},
                    files={
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    },
                )
                successful += bool(response)
                progress.progress(index / len(files))
            if successful:
                clear_data_caches()
                st.success(f"Indexed {successful} of {len(files)} documents.")

    with library_tab:
        query = urlencode(
            {
                "limit": 100,
                **({"esg_framework": selected_framework} if selected_framework != "All" else {}),
            }
        )
        response = api_request(f"/api/v1/documents/list?{query}")
        documents = response.get("documents", []) if response else []
        if not documents:
            st.info("No indexed documents match this view.")
        for item in documents:
            metadata = item["metadata"]
            with st.expander(
                f"{metadata['filename']} · {metadata.get('esg_framework') or 'Unclassified'}"
            ):
                st.write(item.get("content", ""))
                details = api_request(
                    f"/api/v1/documents/document/{metadata['document_hash']}",
                    show_error=False,
                )
                st.caption(
                    f"{details.get('total_chunks', 0) if details else '?'} chunks · "
                    f"{metadata['file_size_mb']:.2f} MB · SHA-256 {metadata['document_hash'][:12]}…"
                )
                if st.button("Delete document", key=f"delete_{metadata['document_hash']}"):
                    deleted = api_request(
                        f"/api/v1/documents/{metadata['document_hash']}",
                        method="DELETE",
                    )
                    if deleted:
                        clear_data_caches()
                        st.rerun()

    with search_tab:
        search_query = st.text_input("Search document chunks")
        if search_query:
            results = api_request(
                "/api/v1/documents/search",
                method="POST",
                data={
                    "query": search_query,
                    "k": 20,
                    "esg_framework": None if selected_framework == "All" else selected_framework,
                },
            )
            for index, source in enumerate((results or {}).get("documents", []), start=1):
                render_source(source, index)


def reports_page(selected_framework: str) -> None:
    render_header(
        "Evidence-grounded reports",
        "Report generation is disabled when source documents, an LLM, or admin authentication are unavailable.",
    )
    current_health = health()
    current_stats = document_stats()
    blockers = []
    if not current_stats.get("total_documents"):
        blockers.append("Upload at least one source document")
    if current_health.get("services", {}).get("llm_service") != "healthy":
        blockers.append("Configure OPENAI_API_KEY or ANTHROPIC_API_KEY")
    if not ui_settings.admin_token:
        blockers.append("Configure ADMIN_TOKEN for both the API and UI")
    if blockers:
        for blocker in blockers:
            st.warning(blocker)

    with st.form("report_form"):
        col1, col2 = st.columns(2)
        report_type = col1.selectbox(
            "Report type",
            ["compliance_summary", "framework_analysis", "gap_analysis", "general"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        options = ["All"] + framework_ids()
        default = options.index(selected_framework) if selected_framework in options else 0
        framework = col2.selectbox("Framework", options, index=default)
        include_recommendations = st.checkbox("Include evidence-linked recommendations", value=True)
        use_agentic = st.checkbox("Use multi-stage agentic workflow", value=True)
        submitted = st.form_submit_button(
            "Generate report",
            type="primary",
            use_container_width=True,
            disabled=bool(blockers),
        )

    if not submitted:
        return
    payload = {
        "report_type": report_type,
        "esg_framework": None if framework == "All" else framework,
        "include_recommendations": include_recommendations,
        "use_agentic_rag": use_agentic,
    }
    with st.spinner("Retrieving evidence and generating the report…"):
        response = api_request("/api/v1/admin/generate-report", method="POST", data=payload)
    if not response:
        return
    st.success("Report generated from the current repository evidence.")
    st.markdown(response["content"])
    st.json(response.get("metadata", {}), expanded=False)
    st.download_button(
        "Download PDF",
        markdown_to_pdf(response["content"], report_type.replace("_", " ").title()),
        f"{response['report_id']}.pdf",
        "application/pdf",
    )


def system_page() -> None:
    render_header("System status", "Live service health and server-owned configuration.")
    current_health = health()
    info = platform_info()
    if not current_health:
        st.error(f"API unavailable at {ui_settings.esg_api_url}")
        return
    st.metric("Overall status", current_health.get("status", "unknown").title())
    frame = pd.DataFrame(
        [
            {"Service": service.replace("_", " ").title(), "Status": status}
            for service, status in current_health.get("services", {}).items()
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.subheader("Public runtime configuration")
    st.json(
        {
            "api_url": ui_settings.esg_api_url,
            "api_version": info.get("version"),
            "frameworks": info.get("supported_frameworks", []),
            "categories": info.get("supported_categories", []),
            "document_types": info.get("document_types", []),
            "supported_extensions": info.get("supported_extensions", []),
            "admin_auth_configured_in_ui": bool(ui_settings.admin_token),
        }
    )
    if st.button("Refresh status"):
        clear_data_caches()
        st.rerun()


def main() -> None:
    catalog_ids = framework_ids()
    with st.sidebar:
        st.markdown("## 🌱 ESG Studio")
        page = st.radio(
            "Workspace",
            ["Overview", "Ask", "Documents", "Reports", "System"],
            label_visibility="collapsed",
        )
        st.divider()
        framework = st.selectbox("Framework focus", ["All"] + catalog_ids)
        current_health = health()
        status = current_health.get("status", "offline")
        st.caption(f"API {status} · {ui_settings.esg_api_url}")

    pages = {
        "Overview": home_page,
        "Ask": lambda: query_page(framework),
        "Documents": lambda: documents_page(framework),
        "Reports": lambda: reports_page(framework),
        "System": system_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
