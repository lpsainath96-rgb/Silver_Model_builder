# 🚀 Deployment Guide: Data Modeler Agent

This guide provides end-to-end steps to launch and deploy the Data Modeler Agent Streamlit application.

## 1. Streamlit in Snowflake (SiS) — [RECOMMENDED]

Snowflake can host Streamlit applications natively. This is the most secure and easiest way to deploy this app.

### Advantages:
- **No Servers**: No need for Azure DevOps, Docker, or external containers.
- **Integrated Security**: Uses Snowflake's native Role Based Access Control (RBAC).
- **Direct Data Access**: No need to manage credentials in `.env` files.

### Steps to Deploy:
1. **Prepare Stage**: Create a stage to hold your code files:
   ```sql
   CREATE STAGE IF NOT EXISTS STREAMLIT_STAGE;
   ```
2. **Upload Files**: Upload the following files from `src/` to the stage (you can use Snowsight or `PUT` command):
   - `ui/streamlit_app.py`
   - `extract/profile_real_data.py`
   - `ai/bronze_silver_mapper.py`
   - `ai/iterative_mapper.py`
   - ... (all other source files in `src/`)
3. **Create Streamlit Object**:
   ```sql
   CREATE STREAMLIT DATA_MODELER_AGENT
   ROOT_LOCATION = '@STREAMLIT_STAGE'
   MAIN_FILE = 'ui/streamlit_app.py'
   QUERY_WAREHOUSE = 'YOUR_WAREHOUSE';
   ```

> [!NOTE]
> **Packages in SiS**: Snowflake's native Streamlit environment uses the Anaconda channel. Ensure all libraries in `requirements.txt` are supported. If a library like `ydata-profiling` is not available, you may need to upload it as a `.zip` to your stage or use a compatible alternative.

---

## 2. Local Launch (Native Streamlit)

### Prerequisites
- Python 3.9 or higher.
- A Snowflake account with Cortex AI permissions.

### Steps
1. **Clone/Download the repository**.
2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**:
   - Create/Update `src/.env` with your Snowflake credentials:
     ```env
     SNOWFLAKE_ACCOUNT="your_account"
     SNOWFLAKE_USER="your_user"
     SNOWFLAKE_PASSWORD="your_password"
     SNOWFLAKE_DATABASE="your_db"
     SNOWFLAKE_SCHEMA="your_schema"
     SNOWFLAKE_WAREHOUSE="your_wh"
     SNOWFLAKE_ROLE="your_role"
     ```
5. **Run the App**:
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

---

## 2. Azure DevOps Integration

Since your code is in Azure DevOps, you can use the following options:

### Option A: Azure App Service (Recommended)
You can deploy your Streamlit app directly to Azure App Service as a Web App.
1. **Azure Pipeline**: Create a `azure-pipelines.yml` to build and deploy your code.
2. **Startup Command**: Set the startup command in Azure App Service to:
   ```bash
   python -m streamlit run src/ui/streamlit_app.py --server.port 8080 --server.address 0.0.0.0
   ```

### Option B: Docker Container
1. **Dockerfile**: Create a Dockerfile in the root:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "src/ui/streamlit_app.py"]
   ```
2. **Azure Container Registry**: Push your image and deploy to Azure Container Instances or App Service for Containers.

---

## 3. Streamlit Community Cloud

**Can I connect same repo to Streamlit?**
Yes, but Streamlit Community Cloud currently supports **GitHub** directly.

### Workaround for Azure DevOps:
1. **Mirror to GitHub**: Create a private GitHub repository and mirror your Azure DevOps repo to it.
2. **Connect GitHub to Streamlit**: Log into [share.streamlit.io](https://share.streamlit.io) and select your GitHub repo.
3. **Secrets Management**: Instead of a `.env` file, use the "Secrets" section in Streamlit Cloud to add your Snowflake credentials.

---

## 🔍 Verification
After launching, navigate to the URL provided (default: `http://localhost:8501`).
- Test connectivity using the sidebar button.
- Verify that table profiling and AI mapping flows work as expected.
