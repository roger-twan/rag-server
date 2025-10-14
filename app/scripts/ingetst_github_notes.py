import asyncio
import base64

from github import Auth, Github

from app.core.config import settings
from app.services.ingestion import ingest_documents_batch

fetch_path_list = ["Portfolio", "Technical", "Skills.md"]
ignore_file_list = ["_index.md"]


async def fetch_and_ingest_github_notes_files():
    auth = Auth.Token(settings.GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo("roger-twan/notes")

    documents = []
    paths_to_process = fetch_path_list.copy()

    while paths_to_process:
        path = paths_to_process.pop(0)
        file_content = repo.get_contents(path)

        if type(file_content) is list:
            paths_to_process.extend(x.path for x in file_content)
        elif (
            file_content.name.endswith(".md")
            and file_content.name not in ignore_file_list
        ):
            content = base64.b64decode(file_content.content).decode("utf-8")

            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "source": "github",
                        "repo": "roger-twan/notes",
                        "path": file_content.path,
                        "filename": file_content.name,
                        "url": file_content.html_url,
                        "sha": file_content.sha,
                    },
                }
            )

    if documents:
        print(f"\nIngesting {len(documents)} documents into Qdrant...")
        result = await ingest_documents_batch(documents)
        print(f"Ingestion complete: {result}")
    else:
        print("No markdown files found to ingest.")


if __name__ == "__main__":
    print("Starting GitHub Notes ingestion...")
    asyncio.run(fetch_and_ingest_github_notes_files())
    print("Done!")
