# APAN All-in-One Docker Stack

This project is an implementation of a stack based on Docker (docker-compose) using MongoDB, PostgreSQL, Express JS (Node.js), Flask, Jupyter, and more.

## Features

- **MongoDB**: NoSQL database
- **PostgreSQL**: Relational database
- **Express JS (Node.js)**: Web framework for Node.js
- **Flask**: Micro web framework for Python
- **Jupyter**: Interactive computing environment
- **Adminer**: Database management tool
- **Streamlit**: Web app framework for Machine Learning and Data Science
- **Ollama**: AI model hosting and management
- **Ollama WebUI**: Web interface for managing AI models

## Building & Running

```sh
# Clone the repository
git clone git@github.com:hyper07/apan5210.git

# Move to the project directory
cd apan-project/

# Build and run the containers
docker-compose up -d

# Stop and remove the containers
docker-compose down
```

### MongoDB

- **Web Interface (Mongo Express)**: [http://localhost:8082](http://localhost:8082)
- **Connection String**:
  ```python
  from pymongo import MongoClient
  client = MongoClient('mongodb://admin:PassW0rd@apan-mongo:27017/')
  ```

### PostgreSQL

- **Web Interface (PgAdmin)**: [http://localhost:5080](http://localhost:5080)
- **Connection Details**:
  ```
  Hostname: apan-postgres
  Port: 5432
  Database: db
  Username: admin
  Password: PassW0rd
  ```

### Jupyter

- **Web Interface**: [http://localhost:8879](http://localhost:8879)

### Adminer

- **Web Interface**: [http://localhost:9091](http://localhost:9091)

### Flask App

- **Web Interface**: [http://localhost:5211](http://localhost:5211)

### Streamlit App

- **Web Interface**: [http://localhost:19502](http://localhost:19502)

### Ollama

- **API Endpoint**: [http://localhost:39870](http://localhost:39870)

### Ollama WebUI

- **Web Interface**: [http://localhost:39081](http://localhost:39081)

## Additional Information

- **Docker Network**: All services are connected via a custom Docker network `apan5210-net`.
- **Volumes**: Persistent data storage is managed using Docker volumes.

For more detailed information on each service, please refer to the respective documentation.