import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

# ── Env vars (set in Render dashboard) ──
os.environ["GOOGLE_API_KEY"]  = os.environ.get("GOOGLE_API_KEY", "")
os.environ["TAVILY_API_KEY"]  = os.environ.get("TAVILY_API_KEY", "")

# ── LLM ──
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tavily_search = TavilySearchResults(max_results=3)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ── Tools ──

@tool
def search_grants(keywords: str) -> str:
    """
    Searches Grants.gov for open and forecasted grants.
    Use this to find a list of potential grant opportunities.
    """
    url = "https://api.grants.gov/v1/api/search2"
    payload = {
        "keyword": keywords,
        "oppStatuses": "posted",
        "rows": 10
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        if response.status_code == 200:
            opportunities = response.json().get("data", {}).get("oppHits") or []
            output = ""
            for opp in opportunities:
                output += f"ID: {opp.get('id')} | Number: {opp.get('number')} | Title: {opp.get('title')}\n"
            return output if output else "No grants found for these keywords."
        return f"Error: Received status code {response.status_code}"
    except Exception as e:
        return f"Error searching grants: {str(e)}"


@tool
def get_grant_details(opportunity_id: str) -> str:
    """
    Fetches the full description and synopsis for a specific grant ID.
    If the official description is missing, it falls back to a web search.
    """
    clean_id = str(opportunity_id).replace("ID:", "").strip()
    url = "https://api.grants.gov/v1/api/fetchOpportunity"
    try:
        response = requests.post(url, json={"opportunityId": clean_id}, headers=HEADERS)
        if response.status_code == 200:
            data   = response.json().get("data", {})
            title  = data.get("opportunityTitle", "Unknown Title")
            number = data.get("fundingOpportunityNumber", "Unknown Number")

            synopsis = data.get("synopsis")
            if synopsis:
                desc = synopsis.get("synopsisExplanation") or synopsis.get("synopsisDesc")
                if desc:
                    return f"Official Synopsis: {desc}"

            forecast = data.get("forecast")
            if forecast:
                desc = forecast.get("forecastExplanation") or forecast.get("forecastDesc")
                if desc:
                    return f"Official Forecast: {desc}"

            # Tavily fallback
            web_results = tavily_search.invoke({"query": f"Grants.gov {number} {title} grant summary description eligibility"})
            return f"No official API description available. Web Search Fallback for {number}:\n{web_results}"
        return f"Error: Could not fetch details for ID {clean_id}."
    except Exception as e:
        return f"Error fetching details: {str(e)}"


@tool
def score_grant_match(grant_description: str, user_profile: str, project_needs: str) -> str:
    """
    Calculates a FunderWonder Match Score (0-100) between a grant and a user's specific needs.
    Use this AFTER fetching the grant details to advise the user on whether they should apply.
    """
    scoring_prompt = f"""
    You are an expert grant evaluator. Evaluate this grant out of 100 based on the user's profile and needs.

    User Profile: {user_profile}
    Project Needs: {project_needs}
    Grant Description: {grant_description}

    Calculate the score using this rubric:
    - Relevance (40 pts): Does the research topic align with their project needs?
    - Eligibility (30 pts): Can this specific user (based on their profile) apply?
    - Funding Type (30 pts): Does it cover what they need?

    Format your response as:
    **FunderWonder Match Score: [Score]/100**
    * **Relevance:** [Brief reason]
    * **Eligibility:** [Brief reason]
    * **Funding Type:** [Brief reason]
    """
    return llm.invoke(scoring_prompt).content


@tool
def generate_and_save_proposal(opportunity_id: str, user_profile: str, project_description: str) -> str:
    """
    Generates a full grant proposal draft for the given opportunity.
    Google Docs export is coming soon — for now returns the full proposal text.
    """
    grant_details = get_grant_details.invoke(opportunity_id)

    proposal_prompt = f"""
    You are a grant proposal template maker for students trying to find funding. Structure the draft like this:
    1. A paragraph for the executive summary - Give an example of the concise overview of the user's project description.
       Write project goals, amount of funding needed, expected impact. REMEMBER, do not make up details about the project that you are unaware of, ask the user for them.
    2. Statement of need - provide resources that prove the user's claim. DO NOT make up websites, articles, evidence. Give links to everything you're referencing.
    3. At least 3 possible goals for the project + project objectives
    4. Project scope: each task with a dependency, deliverable, and acceptance criteria (get MOST of this info from the user)
    5. A box for a budget explanation

    REMEMBER: if you have to fill any gaps of project goals, explanation, details, make it clear that this is a hypothetical proposal and it will get more accurate the more data they give.
    DO NOT use characters to format the output into boxes. Use numbered categories with space between them.

    USER PROFILE:
    {user_profile}

    PROJECT DESCRIPTION:
    {project_description}

    GRANT DETAILS:
    {grant_details}

    Write a full professional grant proposal.
    """
    proposal = llm.invoke(proposal_prompt).content
    return f"Here is your grant proposal draft (Google Docs export coming soon):\n\n{proposal}"


# ── Agent ──
agent_prompt = """You are FunderWonder, an expert research assistant specializing in finding and securing grant funding.

You have access to the following tools. NEVER hallucinate tool names.
1. search_grants(keywords): Searches for grants and returns summaries with their numeric IDs.
2. get_grant_details(opportunity_id): Fetches the full description of a specific grant.
3. score_grant_match(grant_description, user_profile, project_needs): Calculates a match score.
4. generate_and_save_proposal(opportunity_id, user_profile, project_description): Drafts a full proposal.

Follow this strict workflow based on the user's request:
- Phase 1: Discovery. If the user wants to find grants, ALWAYS use search_grants first. Present the findings clearly with their numeric IDs.
- Phase 2: Evaluation. If the user asks if a specific grant is a good fit, ensure you have BOTH their personal/professional background AND their project needs. If missing either, ASK before proceeding. Then run get_grant_details and immediately run score_grant_match.
- Phase 3: Proposal Generation. If the user asks to write a proposal, use generate_and_save_proposal. Since you already gathered their profile and project details in Phase 2, pass them directly into this tool.
"""

tools = [search_grants, get_grant_details, score_grant_match, generate_and_save_proposal]

prompt   = hub.pull("hwchase17/react")
agent    = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# ── FastAPI ──
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    history: List[Dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Build a single input string from the last user message
        last_user_msg = ""
        for msg in reversed(req.history):
            role = msg.get("role", "")
            if role in ("human", "user"):
                last_user_msg = msg.get("content", "")
                break

        # Build chat history string for context
        history_str = ""
        for msg in req.history[:-1]:
            role    = msg.get("role", "human")
            content = msg.get("content", "")
            history_str += f"{role.upper()}: {content}\n"

        full_input = f"{history_str}HUMAN: {last_user_msg}" if history_str else last_user_msg

        response   = executor.invoke({"input": full_input})
        final_text = response.get("output", "Sorry, I could not generate a response.")
        return {"response": final_text}

    except Exception as e:
        return {"error": str(e)}


# ── Serve index.html at root ──
app.mount("/", StaticFiles(directory=".", html=True), name="static")