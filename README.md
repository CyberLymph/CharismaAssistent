# CharismaAssistent
Automated Analysis of Charismatic Leadership Tactics in Famous Texts Using Large Language Models


##  Starting the server
**3. Starting FastAPI Backend**
   
#### Windows (PowerShell) / macOS & Linux

source .venv/bin/activate --> macOS & Linux

.venv\Scripts\Activate.ps1 --> Windows(PowerShell)

cd fastapi_app

uvicorn api:app --host 127.0.0.1 --port 8000 --reload

uvicorn api:app --reload --host 127.0.0.1 --port 8000

---
Swagger UI → http://127.0.0.1:8000/docs

ReDoc → http://127.0.0.1:8000/redoc

---