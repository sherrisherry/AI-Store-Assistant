# AI Store Assistant

An open-source conversational store assistant that combines **RAG, predictive analytics, structured menu data, and MongoDB-backed retrieval** to provide natural-language access to store and customer information.

The project demonstrates an agentic architecture in which a conversational LLM can retrieve store-specific knowledge, query structured data, access a predictive sales model, and retrieve KOC (Key Opinion Consumer) information through a shared Python tool library.

## Features

* **Conversational AI** interface built with Panel
* **Retrieval-Augmented Generation (RAG)** using Chroma for store-specific knowledge
* **Gemini** Resilient model access with automatic model/API-key failover on connection timeouts
* **Predictive sales model** for estimating dish sales
* **MongoDB-backed KOC database** for retrieving KOC information
* **Structured menu dataset** included in the repository
* **Python-based tool library** that exposes data retrieval and predictive capabilities to the agent
* **Dockerized application** for reproducible deployment

## Architecture

The assistant combines conversational AI with retrieval, structured data, and predictive analytics.

```text
                         ┌─────────────────────┐
                         │     Panel UI        │
                         │  Conversational UI  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Gemini-powered    │
                         │       Agent         │
                         └──────────┬──────────┘
                                    │
                          Python Tool Library
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
              ▼                     ▼                      ▼
       ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
       │ Chroma      │      │   MongoDB    │      │ Predictive   │
       │ Vector DB   │      │  KOC & Menu  │      │ Sales Model  │
       └──────┬──────┘      └──────────────┘      └──────────────┘
              │
              ▼
       Brand Stories &
       Store Descriptions
```
### Model Failover and Connection Recovery

The agent includes a lightweight failover mechanism for model connectivity failures. When a model API connection times out, the application automatically rotates to an alternate model/API key and retries the request, allowing the conversational workflow to continue without requiring manual intervention.

This provides basic resilience against transient model-provider connectivity failures while keeping model access configuration separate from the agent workflow.

### Knowledge Retrieval

Brand stories and store descriptions are embedded and stored in **Chroma**. The agent retrieves relevant store-specific context when answering questions that require background or descriptive information.

### Predictive Analytics

A dish-sales predictive model is exposed through the same tool library. The agent can invoke the model when a user request requires a sales prediction.

### Structured Data

The repository includes a menu dataset that is accessed by the predictive through the Python tool library.

### KOC Retrieval

KOC information is stored in a **MongoDB** database hosted separately from the repository. The database contains data collected from publicly available online sources.

The repository does **not** distribute the scraped KOC dataset. Instead, the application provides a configuration placeholder for the MongoDB connection address.

## Agent and Tool Architecture

The agent uses a Python module containing the application's available tools.

The workflow imports this module and exposes its functions to the agent as callable capabilities.

Conceptually:

```text
User request
     │
     ▼
Gemini Agent
     │
     ▼
Tool selection
     │
     ├──────────► Chroma retrieval
     │
     ├──────────► Menu data retrieval
     │
     ├──────────► Dish sales prediction
     │
     └──────────► MongoDB KOC retrieval
                     │
                     ▼
                Tool results
                     │
                     ▼
                Agent response
```

This design keeps data access and analytical capabilities in reusable Python functions while allowing the conversational agent to decide which capabilities are needed for a given request.

## Technology Stack

| Component            | Technology       |
| -------------------- | ---------------- |
| LLM                  | Gemini           |
| Conversational UI    | Panel            |
| RAG / Vector Search  | Chroma           |
| Predictive Analytics | Python ML model  |
| KOC Database         | MongoDB          |
| Tool Integration     | Python functions |
| Deployment           | Docker           |

## Getting Started

### Prerequisites

Install or obtain access to:

* Docker
* A Gemini API key
* A MongoDB instance containing the required KOC data
* Python, if running the project outside Docker

### Configuration

Create a local environment configuration based on the provided template.

Configure:

* Gemini API credentials
* MongoDB connection information
* Other application-specific settings

The MongoDB connection is intentionally configured as an external dependency because the scraped KOC dataset is not distributed with this repository.

> Never commit API keys, database credentials, or other secrets to the repository.

### Run with Docker

Build the application image:

```bash
docker build -t ai-store-assistant .
```

Refer to customer_manual.docx for instructions of running the application.

## Example Use Cases

The assistant can support natural-language questions such as:

* "Tell me the story behind this store."
* "What dishes are available?"
* "Which dishes are expected to sell well?"
* "What KOCs are relevant to this store?"
* "What dishes should we promote based on expected sales?"
* "Tell me about this store and its menu."

The available capabilities depend on the configured data and tools.

## Project Structure

```text
.
├── vdb/                  # Chroma-based retrieval components
├── models/                # Predictive model components
├── data/                 # Included menu and pricing
├── bot.py                  # The agent
├── CBextension.py			# The tools
├── customer_manual.docx	# User manual
├── Dockerfile
├── requirements.txt
└── README.md
```

> The directory names above are illustrative. Update them to match the actual repository structure.

## Design Principles

### Ground LLM responses in application data

The assistant uses RAG and structured data tools to provide access to store-specific information instead of relying solely on the LLM's pretrained knowledge.

### Combine generative and deterministic capabilities

The LLM handles natural-language interaction and task selection, while data retrieval and predictive analytics are performed through dedicated Python functions.

This allows the system to use deterministic data and model outputs when those capabilities are more appropriate than generating an answer directly.

### Modular tool integration

Application capabilities are implemented as reusable Python functions in a shared tool module. The agent workflow imports the module and exposes the relevant functions as callable tools.

This makes it possible to add or modify capabilities without embedding their implementation directly into the conversational workflow.

### Separate application code from external data

The application code and required example datasets are maintained in the repository, while the scraped KOC database remains an external dependency.

## Roadmap

Potential future improvements include:

## Roadmap

Potential future improvements include:

* [ ] Standardized tool interfaces using MCP
* [ ] Agent execution tracing and structured observability
* [ ] Automated evaluation benchmarks for retrieval, tool selection, and answer quality
* [ ] Retrieval evaluation and reranking
* [ ] Additional predictive analytics tools
* [ ] Improved tool execution security and sandboxing
* [ ] Expanded store and menu datasets

## Data and Privacy

The KOC database contains information collected from publicly available online sources and is hosted separately from this repository.

The scraped database itself is **not distributed with the project**. Users deploying the application are responsible for ensuring that any external data they connect to the application is collected, stored, and used in accordance with applicable laws, platform terms, and data-use requirements.

## License

This project is licensed under the **Apache License 2.0**.

See [`LICENSE`](http://www.apache.org/licenses/LICENSE-2.0) for the full license text.

## Disclaimer

This repository is an open-source reference implementation intended for experimentation, research, and demonstration.

The included datasets, predictive models, and configurations may be simplified and should be evaluated and secured appropriately before being used in a production environment.
