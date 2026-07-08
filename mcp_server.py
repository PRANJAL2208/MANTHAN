"""
mcp_server.py
--------------
Exposes literature search and citation grounding verification as reusable
Model Context Protocol (MCP) tools — making them callable by any
MCP-compatible agent framework, not just this application.

Tools exposed:
  - search_literature       : Search OpenAlex & PubMed for scientific papers
  - get_paper_details       : Fetch deep metadata for a specific paper by ID
  - verify_claim_grounding  : Check whether a claim is supported by paper abstracts

Run with:  python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP
from literature_search import search_papers, format_papers_for_prompt, fetch_paper_details
from guardrails import check_grounding as _check_grounding

mcp = FastMCP("literature-search")


@mcp.tool()
def search_literature(query: str, limit: int = 3) -> str:
    """
    Search scientific literature (via OpenAlex & PubMed) for papers
    relevant to the given query. Returns formatted paper summaries
    including a paperId (needed for get_paper_details), title, year,
    citation count, and abstract excerpt.

    Use this first, broadly, to find candidate papers. If one paper
    becomes contentious in a debate or rebuttal, use get_paper_details
    with its paperId to dig deeper before committing to a claim.

    Args:
        query: The search query string. Must be non-empty.
        limit: Number of papers to return. Must be between 1 and 10. Defaults to 3.

    Returns:
        Formatted string of paper summaries ready to use in an LLM prompt.

    Raises:
        ValueError: If query is empty or limit is out of range.
        RuntimeError: If the literature search fails unexpectedly.
    """
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty.")
    if not (1 <= limit <= 10):
        raise ValueError(f"limit must be between 1 and 10, got {limit}.")

    try:
        papers = search_papers(query, limit=limit)
    except Exception as e:
        raise RuntimeError(f"Literature search failed: {e}") from e

    if not papers:
        return f"No papers found for query: '{query}'. Try broadening your search terms."

    return format_papers_for_prompt(papers)


@mcp.tool()
def get_paper_details(paper_id: str) -> str:
    """
    Fetches deep metadata for ONE specific paper, given its paperId
    (obtained from a prior search_literature call). Returns venue,
    full author list, citation count, and a TL;DR summary if available.

    Use this when a specific paper's claim is being directly disputed
    and a quick title + abstract is not enough to settle the point.

    Args:
        paper_id: The paperId string returned by search_literature. Must be non-empty.

    Returns:
        Formatted string with full paper metadata.

    Raises:
        ValueError: If paper_id is empty or no paper is found for the given ID.
        RuntimeError: If the fetch operation fails unexpectedly.
    """
    paper_id = paper_id.strip()
    if not paper_id:
        raise ValueError("paper_id cannot be empty.")

    try:
        details = fetch_paper_details(paper_id)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch paper details: {e}") from e

    if not details:
        raise ValueError(
            f"No paper found for ID: '{paper_id}'. "
            "Ensure the paperId came from a recent search_literature call."
        )

    return (
        f"Title: {details['title']}\n"
        f"Venue: {details['venue']} ({details['year']})\n"
        f"Authors: {', '.join(details['authors'])}\n"
        f"Citations: {details['citationCount']}\n"
        f"TL;DR: {details['tldr'] or 'Not available'}\n"
        f"Abstract: {details['abstract']}"
    )


@mcp.tool()
def verify_claim_grounding(claim: str, paper_abstracts: list[str]) -> str:
    """
    Verifies whether a scientific claim is meaningfully grounded in the
    provided paper abstracts. Uses citation-index overlap analysis to
    check that any [n] citation markers in the claim correspond to
    actual content in the cited abstract — not just a matching title.

    This is the same hallucination guardrail used internally by MANTHAN's
    debate agents, now exposed as a reusable tool for any agent pipeline.

    Usage pattern:
        1. Call search_literature to get papers.
        2. Have an agent write a claim citing papers as [1], [2], etc.
        3. Call verify_claim_grounding with the claim and the abstracts
           of those papers IN ORDER (abstract[0] = paper [1], etc.).

    Args:
        claim: The scientific claim text, which should contain [n] citation markers.
        paper_abstracts: Ordered list of paper abstracts. Index 0 corresponds to [1].
                         Must contain at least one abstract.

    Returns:
        A grounding verdict string: grounded status, which papers supported
        the claim, and any warnings about unsupported citations.

    Raises:
        ValueError: If claim is empty or paper_abstracts is empty.
    """
    claim = claim.strip()
    if not claim:
        raise ValueError("claim cannot be empty.")
    if not paper_abstracts:
        raise ValueError("paper_abstracts must contain at least one abstract.")

    # Build the paper dicts the guardrail expects
    papers = [
        {"abstract": abstract.strip(), "title": f"Paper {i + 1}"}
        for i, abstract in enumerate(paper_abstracts)
        if abstract and abstract.strip()
    ]

    if not papers:
        raise ValueError("All provided paper_abstracts were empty strings.")

    result = _check_grounding(claim, papers)

    if result["grounded"]:
        supported = ", ".join(result["matched_papers"]) if result["matched_papers"] else "none listed"
        return (
            f"✅ GROUNDED\n"
            f"The claim is supported by the provided abstracts.\n"
            f"Supporting papers: {supported}"
        )
    else:
        return (
            f"⚠️ NOT GROUNDED\n"
            f"Warning: {result['warning']}\n"
            f"Partially matched: {result.get('matched_papers', [])}\n"
            f"Action: Revise the claim to only assert what the abstracts actually contain."
        )


if __name__ == "__main__":
    mcp.run()