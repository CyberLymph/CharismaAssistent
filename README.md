# CharismaAssistent
Automated Analysis of Charismatic Leadership Tactics in Famous Texts Using Large Language Models
The project consists of:
- **FastAPI Backend** (analysis & interface to LLMs)
- **Django UI** (web interface for input and presentation)


---

##  Projektstruktur
<pre>
CharismaAssistent-main/
│
├─ charismaassistent_ui/
│  │
│  ├─ manage.py
│  │
│  ├─ config/
│  │  ├─ __init__.py
│  │  ├─ asgi.py
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  └─ wsgi.py
│  │
│  └─ chatui/
│     │
│     ├─ migrations/
│     │
│     ├─ templates/
│     │  ├─ base.html
│     │  ├─ wizard_step1.html
│     │  └─ wizard_step2.html
│     │
│     ├─ styles/
│     │  ├─ base.css
│     │  └─ styles.css
│     │
│     ├─ js/
│     │  ├─ wizard_step1.js
│     │  └─ wizard_step2.js
│     │
│     ├─ fav.ico
│     │
│     ├─ __init__.py
│     ├─ admin.py
│     ├─ apps.py
│     ├─ models.py
│     ├─ tests.py
│     ├─ urls.py
│     └─ views.py
│
├─ fastapi_app/
│  │
│  ├─ prompts/
│  │  ├─ system_default.txt
│  │  ├─ user_default.txt
│  │  └─ user_hybrid_default.txt
│  │
│  ├─ api.py
│  ├─ gemini.py
│  ├─ openaigpt.py
│  └─ schemas.py
│
├─ .gitignore
├─ README.md
└─ requirements.txt
</pre>


## If the Docker setup fails: Development start (manual)
If you cannot run the project via Docker/Docker Compose (e.g., due to local Docker issues or missing permissions), you can start the backend and frontend manually in development mode.


1) Clone the repository and create a virtual environment (root directory)

git clone <REPO_URL><br>
cd CharismaAssistent-main


#### macOS / Linux
python3 -m venv .venv<br>
source .venv/bin/activate<br>
cd CharismaAssistent-main

2) Install dependencies (in root directory) Mac/Linux

pip install --upgrade pip<br>
pip install -r requirements.txt


---
#### Windows (PowerShell)

python -m venv .venv // Do this inside root
.venv\Scripts\Activate.ps1<br>
cd CharismaAssistent-main

2) Install dependencies (in root directory) Windows
python -m pip install --upgrade pip<br>
pip install -r requirements.txt



3)Backend (FastAPI) setup
## Create the .env file
Create a file named .env inside CharismaAssistantMain/fastapi_app

Example .env (adjust values to your keys):
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Google Gemini
GEMINI_API_KEY=your_gemini_key_here<br>
GEMINI_MODEL=gemini-2.5-flash

# Prompt settings
SYSTEM_PROMPT_DEFAULT_PATH=prompts/system_default.txt<br>
USER_PROMPT_DEFAULT_PATH=prompts/user_default.txt<br>
HYBRID_USER_PROMPT_DEFAULT_PATH=prompts/user_hybrid_default.txt


4) Start the FastAPI backend

From the project root:
cd fastapi_app<br>
uvicorn api:app --reload --host 127.0.0.1 --port 8000

Backend should now be reachable at:<br>
http://127.0.0.1:8000


## Frontend (Django) setup
5) Start the Django UI<br>
Open a second terminal, activate the same virtual environment from the project root again, then:
cd CharismaAssistent-main #if you are not in root  <br>
cd charismaassistent_ui<br>
python manage.py migrate<br>
python manage.py runserver 127.0.0.1:8001

Frontend should now be reachable at:<br>
http://127.0.0.1:8001

Notes / common issues<br>
-Make sure ports 8000 (FastAPI) and 8001 (Django) are free.<br>
-If the Django UI cannot reach the backend, ensure the backend URL is not set to localhost:8000 inside a container context.<br>
-In manual development mode,the backend URL should be: http://127.0.0.1:8000.


