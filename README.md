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

## Resolving "Permission denied (publickey)" Error

If you encounter the "Permission denied (publickey)" error when cloning the repository, follow these steps:

1. **Check for existing SSH keys**:
    ```sh
    ls -al ~/.ssh
    ```

2. **Generate a new SSH key** (if you don't have one):
    ```sh
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```

3. **Add your SSH key to the ssh-agent**:
    ```sh
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
    ```

4. **Add the SSH key to your GitHub account**:
    - Copy the SSH key to your clipboard:
        ```sh
        cat ~/.ssh/id_rsa.pub
        ```
    - Go to [GitHub SSH settings](https://github.com/settings/keys) and click "New SSH key".
    - Paste your SSH key and save.

5. **Test your SSH connection**:
    ```sh
    ssh -T git@github.com
    ```

## Accessing Services

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