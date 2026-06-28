"""
mcp_server.py
--------------
Wraps the literature search tool as an MCP (Model Context Protocol) server.

Why this matters for the project:
Right now, search_papers() is just a Python function only THIS project
can use. By exposing it as an MCP server, ANY agent system (built by
anyone, in any framework) could plug into this same literature-search
tool over a standard protocol — that's the "clever reuse of an existing
toolset" the competition rubric explicitly rewards, and it's also just
good practice: tools should be reusable, not locked inside one app.

Run with:  python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP
from literature_search import search_papers, format_papers_for_prompt, fetch_paper_details

mcp = FastMCP("literature-search")


@mcp.tool()
def search_literature(query: str, limit: int = 3) -> str:
    """
    Search scientific literature (via Semantic Scholar) for papers
    relevant to the given query. Returns formatted paper summaries
    including a paperId (needed for get_paper_details), title, year,
    citation count, and abstract excerpt.

    Use this first, broadly, to find candidate papers. If one paper
    becomes contentious in a debate/rebuttal, use get_paper_details
    with its paperId to dig deeper before committing to a claim.
    """
    papers = search_papers(query, limit=limit)
    return format_papers_for_prompt(papers)


@mcp.tool()
def get_paper_details(paper_id: str) -> str:
    """
    Fetches deeper metadata for ONE specific paper, given its paperId
    (obtained from a prior search_literature call). Returns venue,
    full author list, citation count, and a TL;DR summary if available.

    Use this when a specific paper's claim is being directly disputed
    and a quick title+abstract isn't enough to settle the point.
    """
    details = fetch_paper_details(paper_id)
    if not details:
        return f"Could not retrieve details for paper ID: {paper_id}"

    return (
        f"Title: {details['title']}\n"
        f"Venue: {details['venue']} ({details['year']})\n"
        f"Authors: {', '.join(details['authors'])}\n"
        f"Citations: {details['citationCount']}\n"
        f"TL;DR: {details['tldr'] or 'Not available'}\n"
        f"Abstract: {details['abstract']}"
    )


if __name__ == "__main__":
    mcp.run()