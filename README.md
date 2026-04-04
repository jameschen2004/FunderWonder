# FunderWonder

An AI research assistant that helps students discover grant funding and automates the proposal drafting process.

## Overview
FunderWonder addresses the challenges graduate students face when navigating the competitive grant funding landscape. Many researchers struggle to identify grants they qualify for or find it difficult to begin writing complex proposals. This agent acts as a bridge by building a personalized profile based on academic background and research interests. It surfaces relevant opportunities and provides a starting point for applications to help level the playing field for student researchers.

## Features
* Uses academic field and project descriptions to find relevant grants.
* Leverages real time searches via the Grants.gov API to identify open and forecasted opportunities.
* Match Scoring: Evaluates project fit and eligibility using large language models to rank potential funding sources.
* Drafts professional templates tailored to specific grant requirements and applicant profiles.
* Automatically exports generated proposals to a user's Google Drive account.

## Tech Stack
* **Language:** Python 3.x
* **Framework:** FastAPI
* **AI Integration:** LangChain and Google Gemini 2.0 Flash
* **Tools:** Tavily Search API and Grants.gov API
* **APIs:** Google Docs API and Google Drive API
* **Hosting:** Render
