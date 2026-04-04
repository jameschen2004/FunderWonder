import os
import requests
import json
from fastapi import FastAPI
from fastapi import Request, responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub


os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY", "")
GCP_JSON = os.environ.get("GCP_CREDENTIALS")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tavily_search = TavilySearchResults(max_results=3)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive.file']

@tool("search_grants")
def search_grants(keywords: str) -> str:
    """Search for open and forecasted grants on Grants.gov using keywords."""
    url = "https://api.grants.gov/v1/api/search2"
    payload = {"keyword": keywords, "oppStatuses": "posted", "rows": 10}
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

@tool("get_grant_details")
def get_grant_details(opportunity_id: str) -> str:
    """Fetch the full description and eligibility details for a specific grant ID."""
    clean_id = str(opportunity_id).replace("ID:", "").strip()
    url = "https://api.grants.gov/v1/api/fetchOpportunity"
    try:
        response = requests.post(url, json={"opportunityId": clean_id}, headers=HEADERS)
        if response.status_code == 200:
            data = response.json().get("data", {})
            synopsis = data.get("synopsis")
            if synopsis:
                desc = synopsis.get("synopsisExplanation") or synopsis.get("synopsisDesc")
                if desc: return f"Official Synopsis: {desc}"
            
            web_results = tavily_search.invoke({"query": f"Grants.gov {clean_id} description eligibility"})
            return f"Web Search Fallback: {web_results}"
        return f"Error: Could not fetch details for ID {clean_id}."
    except Exception as e:
        return f"Error fetching details: {str(e)}"

@tool("score_grant_match")
def score_grant_match(grant_description: str, user_profile: str, project_needs: str) -> str:
    """Calculate a match score (0-100) by comparing a user profile and project needs to a grant description."""
    scoring_prompt = (
        f"Grant: {grant_description}\n"
        f"User Profile: {user_profile}\n"
        f"Project: {project_needs}\n"
        "Evaluate the match and provide a score out of 100 with a brief reasoning."
    )
    return llm.invoke(scoring_prompt).content

@tool("generate_and_save_proposal")
def generate_and_save_proposal(opportunity_id: str, user_profile: str, project_description: str) -> str:
    """Draft a full professional grant proposal for a specific grant and user profile."""
    proposal_prompt = (
        f"Write a professional grant proposal for Grant ID {opportunity_id}.\n"
        f"Applicant Profile: {user_profile}\n"
        f"Project Description: {project_description}\n"
        "Generate a comprehensive proposal."
    )
    proposal = llm.invoke(proposal_prompt).content
    # The PROPOSAL tags are critical for the frontend to detect the export trigger
    return f"PROPOSAL_START\n{proposal}\nPROPOSAL_END"

# Create a strong system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are FunderWonder, an expert research assistant specializing in finding and securing grant funding.

You have access to the following tools:
1. search_grants(keywords): Use this to find grant opportunities.
2. get_grant_details(opportunity_id): Use this to get the full text of a specific grant.
3. score_grant_match(grant_description, user_profile, project_needs): Use this to see if a grant fits the user.
4. generate_and_save_proposal(opportunity_id, user_profile, project_description): Use this to draft the final proposal.

Workflow:
- Phase 1 (Discovery): Use `search_grants`. Always list results using the format: ID: [id] | Number: [number] | Title: [title] at the end of your message.
- Phase 2 (Evaluation): If the user asks about a fit, get the details via `get_grant_details`, then use `score_grant_match`.
- Phase 3 (Proposal): Use `generate_and_save_proposal`. To successfully save to Google Docs, your final response MUST contain the output of the tool, including the PROPOSAL_START and PROPOSAL_END tags.
     
Whenever you find grants, you MUST append them to the very bottom of your response in this EXACT format for the UI:
ID: [id] | Number: [number] | Title: [title]"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [
    search_grants, 
    get_grant_details, 
    score_grant_match, 
    generate_and_save_proposal
]

agent_runnable = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    history: List[Dict]

@app.get("/auth/google")
async def auth_google():
    if not GCP_JSON:
        return {"error": "GCP_CREDENTIALS not set on server"}
    client_config = json.loads(GCP_JSON)
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri="https://funderwonder-43cc.onrender.com/callback"
    )
    
    # This generates the URL, the state, and the PKCE code_verifier
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true', prompt='consent')
    
    response = responses.RedirectResponse(authorization_url)
    
    # Save the state and code_verifier in secure cookies so the callback route can access them
    response.set_cookie(key="oauth_state", value=state, httponly=True, secure=True)
    if hasattr(flow, 'code_verifier'):
        response.set_cookie(key="code_verifier", value=flow.code_verifier, httponly=True, secure=True)
        
    return response

@app.get("/callback")
async def callback(request: Request):
    client_config = json.loads(GCP_JSON)
    
    # Retrieve the state we saved in the cookie
    saved_state = request.cookies.get("oauth_state")
    
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        state=saved_state,
        redirect_uri="https://funderwonder-43cc.onrender.com/callback"
    )
    
    # Retrieve the code_verifier from the cookie and inject it into the new flow
    code_verifier = request.cookies.get("code_verifier")
    if code_verifier:
        flow.code_verifier = code_verifier
        
    # Now fetch the token
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials
    
    return responses.HTMLResponse(content=f"""
        <html>
            <script>
                localStorage.setItem('gdocs_token', `{creds.to_json()}`); 
                window.location.href = '/';
            </script>
        </html>
    """)

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        history_objs = []
        # Process history (all but the last message)
        for m in req.history[:-1]:
            if m["role"] in ["user", "human"]:
                history_objs.append(HumanMessage(content=m["content"]))
            else:
                history_objs.append(AIMessage(content=m["content"]))

        # Get the actual current input from the last message in history
        current_input = req.history[-1]["content"]

        # Call the agent_executor (NOT 'executor' or 'agent.invoke')
        result = agent_executor.invoke({
            "input": current_input,
            "chat_history": history_objs
        })
        
        return {"response": result.get("output", "I couldn't process that.")}
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"error": str(e)}

class ExportRequest(BaseModel):
    token_json: str
    proposal_text: str

@app.post("/export-to-docs")
def export_to_docs(req: ExportRequest):
    try:
        # Load the user's saved Google credentials
        creds_data = json.loads(req.token_json)
        creds = Credentials.from_authorized_user_info(creds_data, SCOPES)

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(GoogleRequest())

        docs_service = build('docs', 'v1', credentials=creds)

        # Create a blank document
        doc = docs_service.documents().create(body={'title': 'FunderWonder Grant Proposal'}).execute()
        document_id = doc.get('documentId')

        # Insert the AI-generated proposal text into the document
        requests = [{'insertText': {'location': {'index': 1}, 'text': req.proposal_text}}]
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()

        # Return the clickable URL
        return {"url": f"https://docs.google.com/document/d/{document_id}/edit"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
