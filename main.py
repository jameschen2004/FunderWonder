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
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
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

@tool
def search_grants(keywords: str) -> str:
    """Searches Grants.gov for open and forecasted grants."""
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

@tool
def get_grant_details(opportunity_id: str) -> str:
    """Fetches full description/synopsis for a specific grant ID."""
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

@tool
def score_grant_match(input_query: str) -> str:
    """Calculates a FunderWonder Match Score (0-100). Provide the user profile, needs, and grant description in the input."""
    scoring_prompt = f"Evaluate this grant match based on this information: {input_query}"
    return llm.invoke(scoring_prompt).content

@tool
def generate_and_save_proposal(input_query: str) -> str:
    """Generates a full grant proposal draft. Provide the user profile, project description, and grant details in the input."""
    proposal_prompt = f"Write a professional grant proposal based on this information: {input_query}"
    proposal = llm.invoke(proposal_prompt).content
    return f"PROPOSAL_START\n{proposal}\nPROPOSAL_END"

tools = [search_grants, get_grant_details, score_grant_match, generate_and_save_proposal]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

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
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    
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
        
    # Now fetch the token (Google will be happy because we provided the verifier!)
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials
    
    return responses.HTMLResponse(content=f"""
        <html>
            <script>
                localStorage.setItem('gdocs_token', JSON.stringify({creds.to_json()}));
                window.location.href = '/';
            </script>
        </html>
    """)

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        last_user_msg = next((m["content"] for m in reversed(req.history) if m["role"] in ["user", "human"]), "")
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in req.history[:-1]])
        full_input = f"{history_str}\nHUMAN: {last_user_msg}" if history_str else last_user_msg

        response = executor.invoke({"input": full_input})
        return {"response": response.get("output", "I couldn't process that.")}
    except Exception as e:
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
