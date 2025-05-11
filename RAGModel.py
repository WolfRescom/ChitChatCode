import transformers
import os
from github import Github
from datetime import datetime
import base64
from typing import Dict, List, Optional
import time
from langchain.docstore.document import Document
import codeChunking as codeChunking
from dotenv import find_dotenv, load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import cProfile
import pstats
import shutil
import tempfile
import subprocess
import traceback
import stat
import github_graphql as github_graphql
import torch
import pickle
from sqlalchemy import MetaData, Table, insert
import json

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def handle_remove_error(func, path, exc_info):
        """
        Error handler for shutil.rmtree.
        Tries to change the file's mode to writable, then retries the removal.
        """
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")

class GitHubExtractor:
    def __init__(self, github_token: str):
        """
        Initialize the extractor with GitHub API token.

        Args:
            github_token (str): GitHub personal access token
        """
        self.g = Github(github_token)
        self.rate_limiter = self.g.get_rate_limit()
        self.github_token = github_token


    async def extract_repo_data(self, repo_url: str) -> Dict:
        """
        Asynchronously extract comprehensive data from a GitHub repository.

        Args:
            repo_url (str): URL to the GitHub repository

        Returns:
            Dict containing repository data including code, issues, PRs, and commits.
        """
        # Convert URL to owner/repo format
        repo_path = '/'.join(repo_url.rstrip('/').split('/')[-2:])
        basic_info = {
            'full_name': repo_path,
            'description': '',
            'created_at': datetime.now().isoformat(),
            'stargazers_count': 0,
            'forks_count': 0
        }


        # Create a temporary directory to clone the repository
        temp_dir = tempfile.mkdtemp(prefix="github_clone_")
        clone_cmd = ["git", "clone", "--depth=1", repo_url, temp_dir]
        # Run the clone command asynchronously
        proc = await asyncio.create_subprocess_exec(
            *clone_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            shutil.rmtree(temp_dir)
            raise Exception(f"Git clone failed: {stderr.decode().strip()}")
        
        code_files = []
        ignore_extensions = {'.git', '.png', '.jpg', '.ico', '.jpeg'}
        for root, dirs, files in os.walk(temp_dir):
            # Compute the relative directory path
            rel_dir = os.path.relpath(root, temp_dir)
            # If any part of the path is ".git", skip this directory completely.
            if ".git" in rel_dir.split(os.sep):
                continue
            # Optionally, you can filter out directories like 'node_modules' here
            dirs[:] = [d for d in dirs if d not in {"node_modules", "venv", "__pycache__", ".git"}]
            for file in files:
                if any(file.lower().endswith(ext) for ext in ignore_extensions):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    content = ""
                rel_path = os.path.relpath(file_path, temp_dir)
                code_files.append({"path": rel_path, "content": content})

        # Clean up the cloned repository directory
        shutil.rmtree(temp_dir, onerror=handle_remove_error)

        owner, repo_name = repo_path.split("/")

        issues = github_graphql.fetch_all_issues(owner, repo_name, self.github_token)
        prs = github_graphql.fetch_all_pull_requests(owner, repo_name, self.github_token)
        commits = github_graphql.fetch_all_commits(owner, repo_name, self.github_token)

        issues, prs, commits = await asyncio.gather(issues, prs, commits)

        # Use the event loop to run synchronous tasks in a ThreadPoolExecutor
        # loop = asyncio.get_event_loop()
        # repo = self.g.get_repo(repo_path)
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     basic_info = self._extract_basic_info(repo)
        #     future_issues = loop.run_in_executor(executor, self._extract_issues, repo)
        # #     future_prs = loop.run_in_executor(executor, self._extract_pull_requests, repo)
        # #     future_commits = loop.run_in_executor(executor, self._extract_commits, repo)

        #     # issues, prs, commits = await asyncio.gather(
        #     #     future_issues, future_prs, future_commits
        #     # )
        #     issues = await asyncio.gather(
        #         future_issues
        #     )

        return {
            'basic_info': basic_info,
            'issues': issues,
            'pull_requests': prs,
            'commits': commits,
            'code': code_files
        }


    def _extract_issue(self, issue) -> Dict:
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "created_at": issue.created_at.isoformat(),
            "comments": [{
                "body": comment.body,
                "created_at": comment.created_at.isoformat(),
                "user": comment.user.login,
            } for comment in issue.get_comments()]
        }

    def _extract_pull_request(self, pr) -> Dict:
        pr_data = {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "created_at": pr.created_at.isoformat(),
            "reviews": [{
                "body": review.body,
                "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
                "state": review.state,
                "user": review.user.login
            } for review in pr.get_reviews()],
            "comments": [{
                "body": comment.body,
                "created_at": comment.created_at.isoformat(),
                "user": comment.user.login
            } for comment in pr.get_comments()]
        }
        return pr_data

    def _extract_commit(self, commit) -> Dict:
        return {
            "sha": commit.sha,
            "author": commit.commit.author.name,
            "date": commit.commit.author.date.isoformat(),
            "message": commit.commit.message,
            "stats": commit.stats.raw_data,
        }
    





    def _extract_basic_info(self, repo) -> Dict:
        """Extract basic repository information."""
        return {
            'full_name': repo.full_name,
            'description': repo.description,
            'created_at': repo.created_at.isoformat(),
            'stargazers_count': repo.stargazers_count,
            'forks_count': repo.forks_count
        }


    def _extract_issues(self, repo) -> List[Dict]:
        """Extract all issues including comments."""
        issues = []
        for issue in repo.get_issues(state='all'):
            issue_data = {
                'number': issue.number,
                'title': issue.title,
                'body': issue.body,
                'state': issue.state,
                'created_at': issue.created_at.isoformat(),
                'comments': []
            }
            for comment in issue.get_comments():
                issue_data['comments'].append({
                    'body': comment.body,
                    'created_at': comment.created_at.isoformat(),
                    'user': comment.user.login
                })
            issues.append(issue_data)
        return issues

    def _extract_pull_requests(self, repo) -> List[Dict]:
        """Extract all pull requests including reviews and comments."""
        prs = []
        for pr in repo.get_pulls(state='all'):
            pr_data = {
                'number': pr.number,
                'title': pr.title,
                'body': pr.body,
                'state': pr.state,
                'created_at': pr.created_at.isoformat(),
                'reviews': [],
                'comments': []
            }
            for review in pr.get_reviews():
                pr_data['reviews'].append({
                    'body': review.body,
                    'submitted_at': review.submitted_at.isoformat() if review.submitted_at else None,
                    'state': review.state,
                    'user': review.user.login
                })
            for comment in pr.get_comments():
                pr_data['comments'].append({
                    'body': comment.body,
                    'created_at': comment.created_at.isoformat(),
                    'user': comment.user.login
                })
            prs.append(pr_data)
        return prs

    def _extract_commits(self, repo) -> List[Dict]:
        """Extract all commits with their details."""
        commits = []
        for commit in repo.get_commits():
            commits.append({
                'sha': commit.sha,
                'author': commit.commit.author.name,
                'date': commit.commit.author.date.isoformat(),
                'message': commit.commit.message,
                'stats': commit.stats.raw_data
            })
        return commits

    def _extract_discussions(self, repo) -> List[Dict]:
        """Extract repository discussions if available."""
        discussions = []
        try:
            for discussion in repo.get_discussions():
                discussion_data = {
                    'title': discussion.title,
                    'body': discussion.body,
                    'created_at': discussion.created_at.isoformat(),
                    'comments': []
                }
                for comment in discussion.get_comments():
                    discussion_data['comments'].append({
                        'body': comment.body,
                        'created_at': comment.created_at.isoformat(),
                        'user': comment.user.login
                    })
                discussions.append(discussion_data)
        except Exception as e:
            print(f"Unable to fetch discussions: {str(e)}")
        return discussions

# %% [markdown]
# # AI Agent

# %%
# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM
#from langchain.llms import HuggingFacePipeline
import torch

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import concurrent.futures
import threading
import queue

import iris
from langchain_iris import IRISVector

# from langchain.vectorstores import Chroma

import wget

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

#directory = r"repo_data/"
COLLECTION_NAME = "Commits history"
username = 'demo'
password = 'demo'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972'
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

def build_directory_tree(file_list):
    """
    file_list: a list of dicts like [{'path': 'src/main.py', 'content': '...'}, ...]

    Returns a nested dictionary representing the directory structure.
    Example:
    {
      'src': {
          'main.py': {},
          'routers': {
              'index.py': {}
          }
      },
      'docs': {
          'overview.md': {}
      }
    }
    """
    tree = {}
    for file_info in file_list:
        path = file_info["path"]
        parts = path.split("/")
        current_level = tree
        for idx, part in enumerate(parts):
            # If it's the last component of the path, treat it as a file
            if idx == len(parts) - 1:
                # Use an empty dict to represent a file
                current_level.setdefault(part, {})
            else:
                # This is a directory
                current_level = current_level.setdefault(part, {})
    return tree


def create_tree_string(tree, prefix="", is_last=True):
    """
    Recursively build a string that visually represents the directory tree.
    """
    # This will accumulate lines of text
    lines = []
    entries = list(tree.items())
    for i, (name, subtree) in enumerate(entries):
        # Check if this item is the last in its current directory
        current_is_last = (i == len(entries) - 1)

        # The branch prefix (├── or └──), with indentation
        branch = "└── " if current_is_last else "├── "

        # Build the line for this entry (file or dir)
        lines.append(prefix + branch + name)

        if subtree:  # means it’s a directory or sub-entries
            # Deeper prefix depends on whether this is the last entry
            extension = "    " if current_is_last else "│   "
            lines.extend(
                create_tree_string(subtree, prefix + extension, current_is_last)
            )
    return lines

def build_directory_structure_text(file_list):
    """
    Given a list of files, build a nested directory tree and produce the ASCII text.
    """
    tree = build_directory_tree(file_list)
    lines = create_tree_string(tree)
    lines.insert(0, "Directory Structure: ")
    return "\n".join(lines)


def chunk_all_repo_data(
    repo_data: Dict,
    output_dir: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    1) Optionally save the repository data to local text files
       using save_to_text.
    2) Build strings in memory for each section (basic info, code,
       issues, pull requests, commits).
    3) Chunk those strings using a RecursiveCharacterTextSplitter.
    4) Return the resulting list of Documents.
    """

    # -- 1) (Optional) Save data to text files locally --
    #    If you don't need on-disk files, you can comment this out.
    #save_to_text(repo_data, output_dir)

    # Prepare a text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Customize how you want to break up text
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []  # We'll collect chunked Documents here

    # -----------------------------------------------------------------
    #    2) Build strings in memory for each part of the repo data
    # -----------------------------------------------------------------

    # (A) Basic Info
    # -----------------------------------------------------------------
    basic_info = repo_data.get("basic_info", {})
    basic_info_text = (
        f"Repository: {basic_info.get('full_name', '')}\n"
        f"Description: {basic_info.get('description', '')}\n"
        f"Created at: {basic_info.get('created_at', '')}\n"
        f"Stars: {basic_info.get('stargazers_count', 0)}\n"
        f"Forks: {basic_info.get('forks_count', 0)}\n"
    )
    if basic_info_text.strip():
        doc = Document(page_content=basic_info_text, metadata={"type": "basic_info", "path": "N/A"})
        chunked_docs = splitter.split_documents([doc])
        all_docs.extend(chunked_docs)


    # (B) Code files
    # -----------------------------------------------------------------
    code_files = repo_data.get("code", [])
    dir_structure_text = build_directory_structure_text(code_files)
    print(dir_structure_text)
    dir_doc = Document(page_content=dir_structure_text, metadata={"type": "directory_tree", "path": "N/A"})
    all_docs.append(dir_doc)

    for file_data in code_files:
        file_path = file_data.get("path", "")
        file_content = file_data.get("content", "")
        if file_content.strip():
            source_bytes = file_content.encode("utf-8")
            tree = codeChunking.parser.parse(source_bytes)
            chunks = codeChunking.chunker(tree, file_content)

            for chunk in chunks:
                chunk_text = chunk.extract_lines(file_content)
                # Create a LangChain Document or store it however you like
                doc = Document(
                    page_content=chunk_text,
                    metadata={"type": "code", "path": file_path}
                )
                all_docs.append(doc)

    # (C) Issues
    # -----------------------------------------------------------------
    issues = repo_data.get("issues", [])
    issues_texts = []
    for issue in issues:
        i_num = issue.get("number", "")
        i_title = issue.get("title", "")
        i_state = issue.get("state", "")
        i_created = issue.get("created_at", "")
        i_body = issue.get("body", "")
        # Build a single string for each issue plus its comments
        issue_text = (
            f"ISSUE #{i_num}: {i_title}\n"
            f"State: {i_state}\n"
            f"Created: {i_created}\n"
            f"Description:\n{i_body}\n"
            "Comments:\n"
        )
        # for comment in issue.get("comments", []):
        #     comment_body = comment.get("body", "")
        #     comment_user = comment.get("user", "")
        #     comment_created = comment.get("created_at", "")
        #     issue_text += f"- {comment_created} by {comment_user}: {comment_body}\n"
        for comment in issue.get("comments", {}).get("nodes", []):
            comment_body = comment.get("body", "")
            comment_user = comment.get("author", {}).get("login", "")
            comment_created = comment.get("createdAt", "")
            issue_text += f"- {comment_created} by {comment_user}: {comment_body}\n"

        issues_texts.append(issue_text)

    # Chunk each issue text
    for text in issues_texts:
        if text.strip():
            doc = Document(page_content=text, metadata={"type": "issue", "path": "N/A"})
            chunked_docs = splitter.split_documents([doc])
            all_docs.extend(chunked_docs)

    # (D) Pull Requests
    # -----------------------------------------------------------------
    pull_requests = repo_data.get("pull_requests", [])
    pr_texts = []
    for pr in pull_requests:
        pr_num = pr.get("number", "")
        pr_title = pr.get("title", "")
        pr_body = pr.get("body", "")
        pr_state = pr.get("state", "")
        pr_created = pr.get("created_at", "")

        pr_text = (
            f"PR #{pr_num}: {pr_title}\n"
            f"State: {pr_state}\n"
            f"Created: {pr_created}\n"
            f"Description:\n{pr_body}\n"
            "Reviews:\n"
        )
        for review in pr.get("reviews", {}).get("nodes", []):
            r_body = review.get("body", "")
            r_state = review.get("state", "")
            r_submitted = review.get("submittedAt", "")  # note case might need to match your query
            r_user = review.get("author", {}).get("login", "")
            pr_text += f"- {r_submitted} by {r_user} ({r_state}): {r_body}\n"

        pr_text += "\nComments:\n"
        for comment in pr.get("comments", {}).get("nodes", []):
            c_body = comment.get("body", "")
            c_user = comment.get("author", {}).get("login", "")
            c_created = comment.get("createdAt", "")
            pr_text += f"- {c_created} by {c_user}: {c_body}\n"

        pr_texts.append(pr_text)

    for text in pr_texts:
        if text.strip():
            doc = Document(page_content=text, metadata={"type": "pull_request", "path": "N/A"})
            chunked_docs = splitter.split_documents([doc])
            all_docs.extend(chunked_docs)

    # (E) Commits
    # -----------------------------------------------------------------
    commits = repo_data.get("commits", [])
    commit_texts = []
    for commit in commits:
        c_sha = commit.get("sha", "")
        c_author = commit.get("author", "")
        c_date = commit.get("date", "")
        c_message = commit.get("message", "")
        c_stats = commit.get("stats", "")

        # Build a single string for each commit
        commit_text = (
            f"Commit: {c_sha}\n"
            f"Author: {c_author}\n"
            f"Date: {c_date}\n"
            f"Message:\n{c_message}\n"
            f"Stats: {c_stats}\n"
        )
        commit_texts.append(commit_text)

    for text in commit_texts:
        if text.strip():
            doc = Document(page_content=text, metadata={"type": "commit", "path": "N/A"})
            chunked_docs = splitter.split_documents([doc])
            all_docs.extend(chunked_docs)

    # (F) Discussions (uncomment if you have them)
    # -----------------------------------------------------------------
    # If discussions are present in `repo_data`, you can chunk them similarly:
    # discussions = repo_data.get("discussions", [])
    # discussion_texts = []
    # for discussion in discussions:
    #     ...
    #     discussion_texts.append(discussion_text)
    #
    # for text in discussion_texts:
    #     if text.strip():
    #         doc = Document(page_content=text, metadata={"type": "discussion", "path": "N/A"})
    #         chunked_docs = splitter.split_documents([doc])
    #         all_docs.extend(chunked_docs)

    # 3) Return the final list of chunked Documents
    return all_docs



def db_insertion_worker(batch_queue, embeddings, vector_store_container):
    try:
        iris_vector = IRISVector(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING
        )
    except Exception as e:
        print("Failed to open persistent connection:", e)
        return

    vector_store_container['store'] = iris_vector

    while True:
        batch = batch_queue.get()
        if batch is None:
            batch_queue.task_done()
            break
        try:
            iris_vector.add_documents(batch)
        except Exception as e:
            print("Failed to add documents to vector store")
        batch_queue.task_done()


def injest_to_db(docs, batch_size=50, num_workers=4):
    embeddings = HuggingFaceEmbeddings()
    doc_batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    batch_queue = queue.Queue()

    # A container to store the persistent vector store instance created by the worker
    vector_store_container = {}

    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target = db_insertion_worker, args=(batch_queue, embeddings, vector_store_container))
        t.start()
        threads.append(t)

    for batch in doc_batches:
        batch_queue.put(batch)

    for _ in range(num_workers):
        batch_queue.put(None)

    batch_queue.join()

    for t in threads:
        t.join()

    return vector_store_container.get('store')


# def get_all_file_paths(directory):
#     """Recursively collect all file paths in a directory and subdirectories."""
#     file_paths = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             # Join the root and file name, then replace backslashes with forward slashes
#             full_path = os.path.join(root, file).replace("\\", "/")
#             file_paths.append(full_path)
#     return file_paths


# def go_through_folder(files):
#     """
#     Takes all file paths, loads them, chunks them, and returns
#     a vector store reference (IRISVector).
#     """
#     # global docsearch, testvar  # Declare global variables at the start
#     # testvar += 1
#     MAX_CHUNK_SIZE = 50000 
#     embeddings = HuggingFaceEmbeddings()
#     all_valid_texts = []
#     for filename in files:
#         print(f"Processing file: {filename}")
#         loader = TextLoader(filename, encoding="utf-8")
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000, chunk_overlap=0, separators=["\n", " ", ""]
#         )
#         texts = text_splitter.split_documents(documents)

#         # Filter out oversized chunks
#         valid_texts = [text for text in texts if len(text.page_content) <= MAX_CHUNK_SIZE]
#         all_valid_texts.extend(valid_texts)


#     # Create a new IRIS vector store from all valid texts
#     docsearch = IRISVector.from_documents(
#         embedding=embeddings,
#         documents=all_valid_texts,
#         collection_name=COLLECTION_NAME,
#         connection_string=CONNECTION_STRING,
#     )
#     return docsearch

def clear_table(table_name):
    """
    Deletes all rows from the given IRIS table, but leaves the table structure intact.
    """
    with iris.connect(connectionstr = f"{hostname}:{port}/{namespace}", username="demo", password="demo") as conn:
        cur = conn.cursor()
        # If the table name includes spaces or special characters, keep it in double quotes.
        sql = f'DELETE FROM "{table_name}"'
        try:
            cur.execute(sql)
            conn.commit()
            print(f"All rows deleted from table: {table_name}")
        except Exception as e:
            print(f"Could not clear table {table_name}: {e}")


# %%
import accelerate

print(accelerate.__version__)

# %%
# !pip install --upgrade langchain langchain-core langchain-groq

# %%
from langchain.llms.base import LLM
from langchain_groq import ChatGroq
import requests
from typing import Optional, List
from langchain_core.retrievers import BaseRetriever
from langchain.memory import ConversationBufferMemory

# Define Groq API key and base URL
os.getenv("GROQ_API_KEY") # Replace with your actual Groq API key
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.7,
    max_tokens = None,
    max_retries = 2
)

# %%
# Prompt template
# prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, try to make your best possible guess based on the context.

# {context}

# Question: {question}
# """
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# %%
# interactive_qa()

# %%
import streamlit as st
import random
import time
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

DOC_TEMPLATE = """[FILE PATH: {path} | TYPE: {type} | ]
{page_content}
"""

STUFF_TEMPLATE = """You are an assistant. Use the following code or text chunks
and their metadata (like file paths) to answer the user's question.

{context}

Question: {question}
Helpful Answer:
"""

# Build the doc-level prompt (injects each chunk's content + metadata)
doc_prompt = PromptTemplate(
    template=DOC_TEMPLATE,
    input_variables=["page_content", "path", "type"],  # or any fields you have in metadata
)

question_generator_chain = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT  # or your own prompt
)

# Build the higher-level "stuff" prompt (merges the chunk texts for the LLM)
stuff_prompt = PromptTemplate(
    template=STUFF_TEMPLATE,
    input_variables=["context", "question"],
)

def build_qa_chain(docsearch):
    
    # Only create the memory once
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    # Only create the chain once
    if "qa_chain" not in st.session_state:
        # st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     memory=st.session_state.memory,
        #     get_chat_history=lambda h: h,
        #     return_source_documents=False
        # )
        stuff_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=stuff_prompt),
            document_prompt=doc_prompt,
            document_variable_name="context",  # This is how doc chunks get inserted
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 10})
        # 2) Build a ConversationalRetrievalChain that uses this custom chain
        st.session_state.qa_chain = ConversationalRetrievalChain(
            question_generator=question_generator_chain,
            combine_docs_chain=stuff_chain,
            retriever=retriever,
            memory=st.session_state.memory,
            get_chat_history=lambda h: h,
            # Optionally set this to True if you want to see the
            # source_documents returned for your own debugging/UI
            return_source_documents=False
        )
    # Return the chain and memory from session state
    return st.session_state.qa_chain, st.session_state.memory

def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)\
        

def display_output(link):
    st.text("The link you entered is: " + link)


    if(st.session_state.processed_link != link) or (st.session_state.docsearch is None):
        st.write("Processing repository for the first time. Please wait...")
        try:
            if os.name == "nt":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            profiler = cProfile.Profile()
            profiler.enable()
            clear_table(COLLECTION_NAME)
            github_token = os.getenv("GITHUB_TOKEN")
            extractor = GitHubExtractor(github_token)
            repo_data = asyncio.run(extractor.extract_repo_data(link))
            profiler.disable()

            # Print the 20 slowest functions sorted by cumulative time.
            stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
            stats.print_stats(20)

            with open("profile_stats.txt", "w") as f:
                stats.stream = f
                stats.print_stats(20)

            output_dir = "repo_data"
            st.write("Saving data to text files...")


            #iris_store.clear_collection("Commits_history")


            #save_to_text(repo_data, output_dir)
            all_docs = chunk_all_repo_data(repo_data, output_dir)
            st.write("Data saved. Now building the vector store. Please wait...")
            


            # --- 2) Build vector store from local files
            #all_file_paths = get_all_file_paths(output_dir)
            docsearch = injest_to_db(all_docs)

            
            st.write("Vector store ready!")


            # Step 4: Update session_state
            st.session_state.docsearch = docsearch
            st.session_state.processed_link = link
            st.write("Repository processed successfully!")
        except Exception as e:
            traceback.print_exc()
            st.error(f"Error extracting repository data: {e}")
            
            return
    else:
        docsearch = st.session_state.docsearch

    qa_chain, memory = build_qa_chain(docsearch)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = qa_chain({"question": prompt}, {"chat_history": st.session_state.history})
            answer = stream["answer"]
            st.markdown(answer) 
        st.session_state.history.append((prompt, answer))
        st.session_state.messages.append({"role": "assistant", "content": answer})

def reset_chat():
    st.session_state.messages = []
    st.session_state.history = []

def main():
    st.set_page_config(layout="wide")
    st.title("ChitChatCode")
    st.header("Analyze Github Repositories Easily")
    image = None
    if "docsearch" not in st.session_state:
        st.session_state.docsearch = None
    if "link" not in st.session_state:
        st.session_state.processed_link = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    link = st.text_input("Enter the link to the repository you want to understand", placeholder= "https://github.com/", value = None, on_change=reset_chat, key = "link")
    #filename = st.text_input("Enter the file name you want to understand", value = None, on_change=reset_chat, key = "filename")
    if link is not None:
        display_output(link)

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
    main()


# %%

# %%

# %%



