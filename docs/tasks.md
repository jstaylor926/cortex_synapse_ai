Phase 0: Foundations & Environment Setup

* Infrastructure Setup:
    * Configure and launch the Docker environment using infra/docker-compose.yml
      for PostgreSQL (with pgvector) and Ollama.
    * Verify that all services are running and accessible.
* Backend Setup (FastAPI):
    * Initialize the FastAPI application structure in the api/ directory.
    * Establish the initial database connection and session management in
      api/db.py.
    * Set up Alembic for database migrations in api/migrations/.
* Frontend Setup (React):
    * Initialize the React + TypeScript project using Vite in the web/
      directory.
    * Set up basic application structure with a root App.tsx component.
* Initial Verification:
    * Create a seed script or manually insert sample data into the database.
    * Confirm that the FastAPI backend can connect to the database and that the
      React frontend can fetch data from a sample API endpoint.
    * Pull the required LLM models using Ollama (e.g., ollama pull llama3,
      ollama pull nomic-embed-text).

Phase 1: Core Persistence & Notebook CRUD

* Backend (API & Database):
    * Define SQLAlchemy models in api/models.py for Projects, Notebooks, and
      Snippets.
    * Create Pydantic schemas for data validation and serialization.
    * Implement RESTful CRUD endpoints in api/main.py for managing projects,
      notebooks, and snippets.
    * Generate the initial database schema using an Alembic migration.
* Frontend (UI):
    * Integrate the Monaco Editor into the React application for markdown and
      code cell editing.
    * Build UI components for creating, viewing, editing, and deleting
      notebooks and snippets.
    * Implement state management for handling application data.

Phase 2: Code Execution & Background Jobs

* Backend:
    * Implement a system for executing Python code cells within a sandboxed
      environment (e.g., per-notebook virtual environments).
    * Integrate a background task queue (e.g., Celery, RQ, or Arq) for handling
      long-running processes like document indexing or model actions.
    * Create an API endpoint to trigger and manage code execution jobs.
* Frontend:
    * Add a "Run" button to code cells in the notebook.
    * Display the output (stdout, stderr, results) from code execution jobs in
      the UI.

Phase 3: Embeddings & Hybrid Search

* Backend:
    * Develop a document processing pipeline to import and chunk text from PDF
      and DOCX files.
    * Integrate an embedding model via Ollama (e.g., nomic-embed-text) to
      generate vector embeddings for text chunks.
    * Implement a hybrid search service that combines keyword-based search
      (e.g., BM25/Postgres FTS) with vector similarity search from pgvector.
    * Create an API endpoint for performing hybrid searches across all indexed
      content.
* Frontend:
    * Build a global search bar or a dedicated search panel in the UI.
    * Display search results, including source information and snippets of
      matching text.

Phase 4: RAG and Local LLM Integration

* Backend:
    * Implement the Model Swap Harness (llm/provider.py) to allow switching
      between different local LLMs (Llama 3, GPT-OSS).
    * Build the RAG (Retrieval-Augmented Generation) pipeline that:
        1. Takes a user query.
        2. Uses the hybrid search service to retrieve relevant context.
        3. Injects the context into a prompt for the selected LLM.
        4. Returns the generated answer.
    * Develop logic to include citations (source document/chunk) with the RAG
      response.
* Frontend:
    * Create a Q&A panel or chat interface for interacting with the RAG system.
    * Display the LLM-generated answers and clearly link to the source
      citations.

Phase 5: AI Assistant Actions

* Backend:
    * Implement backend logic for high-level "Actions" that use the LLM
      provider:
        * RefactorSnippet: Takes a code snippet and returns a refactored
          version.
        * GenerateTests: Generates unit test cases for a given function or
          snippet.
        * SummarizeDocument: Creates a concise summary of a document or note.
        * GenerateFlashcards: Extracts key concepts from text and formats them
          as Q&A pairs.
    * Expose these actions via new API endpoints.
* Frontend:
    * Add UI elements (buttons, context menus) to trigger these AI actions on
      snippets, notes, or documents.
    * Create a view for displaying and using the generated flashcards.

Phase 6: Quality of Life & Export

* Backend:
    * Implement functionality to export user data (notes, snippets) to markdown
      or JSON formats.
    * Develop a system for creating and managing "Packs" (shareable, versioned
      bundles of content).
    * Build the logic for the spaced repetition algorithm (e.g., SM-2) to
      schedule flashcard reviews.
* Frontend:
    * Create a "Study Dashboard" to display scheduled flashcards and review
      progress.
    * Build a settings page for application configuration (e.g., model
      selection, data management).
    * Implement a simple UI for exporting data and managing packs.
