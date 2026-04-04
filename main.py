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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent
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
def score_grant_match(input_query: str) -> str:
    """Calculate a match score (0-100) by comparing a user profile to a grant description."""
    scoring_prompt = f"Evaluate this grant match based on this information: {input_query}"
    return llm.invoke(scoring_prompt).content

@tool("generate_and_save_proposal")
def generate_and_save_proposal(input_query: str) -> str:
    """Draft a full professional grant proposal for a specific grant and user profile."""
    proposal_prompt = f"Write a professional grant proposal based on this information: {input_query}"
    proposal = llm.invoke(proposal_prompt).content
    return f"PROPOSAL_START\n{proposal}\nPROPOSAL_END"

# Fix: Re-initialize the tools list with the named tools
tools = [search_grants, get_grant_details, score_grant_match, generate_and_save_proposal]

# Create a comprehensive system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are FunderWonder, a premier AI-driven grant strategist. Your mission is to assist researchers and students in navigating the complex landscape of funding opportunities.

OPERATIONAL PROTOCOLS:
1. SEARCH: Use `search_grants` to identify opportunities. Always verify keywords with the user's research description.
2. ANALYZE: For any grant of interest, use `get_grant_details` to retrieve eligibility and synopsis data.
3. SCORE: Use `score_grant_match` to provide a quantitative alignment analysis (0-100) between the user's profile and the grant requirements. Provide a brief justification for the score.
4. DRAFT: Only use `generate_and_save_proposal` when a user confirms they want a full draft for a specific grant ID.

FORMATTING:
When listing search results, you MUST append them to the end of your response in this EXACT format for the UI:
ID: [id] | Number: [number] | Title: [title]"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Re-initialize the agent and executor
agent = create_tool_calling_agent(llm, tools, prompt)
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
        # Convert simple list of dicts into proper LangChain message objects
        lc_history = []
        # We take all messages except the very last one to use as context
        for m in req.history[:-1]:
            if m["role"] in ["user", "human"]:
                lc_history.append(HumanMessage(content=m["content"]))
            else:
                lc_history.append(AIMessage(content=m["content"]))
        
        # The last message in the history is the current user input
        last_user_msg = req.history[-1]["content"]

        # Pass the structured history and the new input to the executor
        response = executor.invoke({
            "input": last_user_msg,
            "chat_history": lc_history
        })
        
        return {"response": response.get("output", "I couldn't process that.")}
    except Exception as e:
        print(f"Chat Error: {str(e)}") # Log for Render debugging
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
