# APAN5210

This project is an implementation of a stack based on Docker (docker-compose) using Ollama and Streamlit.

## Features

- **Streamlit**: Web app framework for Machine Learning and Data Science
- **Ollama**: AI model hosting and management
- **Ollama WebUI**: Web interface for managing AI models

## Agents Overview

This stack consists of the following main agents/services:

- **Streamlit App**  
  Provides a user-friendly web interface for interacting with AI models hosted on Ollama. Users can submit prompts and view responses directly in the browser.

- **Ollama**  
  Hosts and manages large language models (LLMs). Exposes an API for model inference and management, which is consumed by the Streamlit app and Ollama WebUI.

- **Ollama WebUI**  
  A web-based management interface for Ollama, allowing users to manage models, view logs, and monitor usage.

### How They Work Together

1. The **Streamlit App** communicates with the **Ollama API** to send user prompts and receive model responses.
2. The **Ollama WebUI** provides administrative capabilities for managing models and monitoring Ollama.
3. All services are containerized and communicate over a shared Docker network.

## Folder Structure

```
apan5210/
├── docker-compose.yml            # Docker Compose configuration
├── app-streamlit/                # Source code for the Streamlit web application
│   ├── Home.py                   # Main entry point for the Streamlit app
│   ├── agents/                   # Folder for agents
│   ├── controllers/              # Folder for contollers
│   ├── db/                       # Folder for chroma sql
│   ├── files/                    # Folder for all the files
│   ├── locales/                  # Folder for locales(languages)
│   ├── pages/                    # Folder for additional pages
│   │   ├── 0_Data Analysis.py    # Page for performing data analysis
│   │   ├── 1_Analyzer Agent.py   # Page showcasing the analyzer agent functionality
│   │   ├── 2_Corder Agent.py     # Page for corder agent functionality
│   │   ├── 3_Translater Agent.py # Page for translator agent functionality
│   │   ├── 4_Insight Agent.py    # Page for insight agent functionality
│   │   ├── 5_Reporter Agent.py   # Page for reporter agent functionality
│   │   ├── 6_Reviewer Agent.py   # Page for reviewer agent functionality
│   │   ├── 7_Application.py      # Page for application-related functionality
│   │   └── 8_Setting.py          # Page for application settings
│   │
│   └── utils/                    # folder for contants variable file or else.
├── docker/                       # Fodler for Dockerfile
├── requirements/                 # Folder for requirement files
├── .env                          # env file
└── README.md                     # Project documentation
```
## Project Structure

- `app-streamlit/`: Contains the Streamlit application code.
    - `Home.py`: Main entry point for the Streamlit app.
    - `agents/`: Folder for agents.
    - `controllers/`: Folder for controllers.
    - `db/`: Folder for Chroma SQL database.
    - `files/`: Folder for all the files.
    - `locales/`: Folder for localization (languages).
    - `pages/`: Folder for application pages.
    - `utils/`: Folder for constants and utility/helper functions.
- `docker/`: Folder for Dockerfile.
- `requirements/`: Folder for requirement files.
- `docker-compose.yml`: Defines and configures all services and their networking.
- `.env`: Environment variables file.
- `README.md`: This documentation file.

## Building & Running

```sh
# Clone the repository
git clone git@github.com:hyper07/apan5210.git

# Move to the project directory
cd apan5210/

# Build and run the containers
docker-compose up -d

# Stop and remove the containers
docker-compose down
```

### Streamlit App

- **Web Interface**: [http://localhost:19501](http://localhost:19501)

### Ollama

- **API Endpoint**: [http://localhost:39870](http://localhost:39870)

### Ollama WebUI

- **Web Interface**: [http://localhost:39090](http://localhost:39090)

### Ollama API

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

## Additional Information

- **Docker Network**: All services are connected via a custom Docker network `apan5210-net`.
- **Volumes**: Persistent data storage is managed using Docker volumes.

For more detailed information on each service, please refer to the respective documentation.