import os
import requests
import aiohttp
import asyncio

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

async def run_query(query: str, variables: dict, token: str) -> dict:
    """
    Helper function to execute a GraphQL query against GitHub's API.
    """
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(GITHUB_GRAPHQL_URL, json={"query": query, "variables": variables}, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                raise Exception(f"Query failed with status {response.status}: {text}")
    

async def fetch_all_issues(owner: str, repo_name: str, token: str) -> list:
    """
    Fetch all issues (open and closed) for the repository.
    """
    issues = []
    query = """
    query ($owner: String!, $name: String!, $after: String) {
      repository(owner: $owner, name: $name) {
        issues(first: 100, after: $after, states: [OPEN, CLOSED]) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            number
            title
            body
            state
            createdAt
            comments(first: 100) {
              nodes {
                body
                createdAt
                author {
                  login
                }
              }
            }
          }
        }
      }
    }
    """
    variables = {"owner": owner, "name": repo_name, "after": None}
    while True:
        result = await run_query(query, variables, token)
        issues_page = result["data"]["repository"]["issues"]
        issues.extend(issues_page["nodes"])
        if issues_page["pageInfo"]["hasNextPage"]:
            variables["after"] = issues_page["pageInfo"]["endCursor"]
        else:
            break
    return issues

async def fetch_all_pull_requests(owner: str, repo_name: str, token: str) -> list:
    """
    Fetch all pull requests (open, closed, merged) for the repository.
    """
    prs = []
    query = """
    query ($owner: String!, $name: String!, $after: String) {
      repository(owner: $owner, name: $name) {
        pullRequests(first: 100, after: $after, states: [OPEN, CLOSED, MERGED]) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            number
            title
            body
            state
            createdAt
            reviews(first: 100) {
              nodes {
                body
                submittedAt
                state
                author {
                  login
                }
              }
            }
            comments(first: 100) {
              nodes {
                body
                createdAt
                author {
                  login
                }
              }
            }
          }
        }
      }
    }
    """
    variables = {"owner": owner, "name": repo_name, "after": None}
    while True:
        result = await run_query(query, variables, token)
        pr_page = result["data"]["repository"]["pullRequests"]
        prs.extend(pr_page["nodes"])
        if pr_page["pageInfo"]["hasNextPage"]:
            variables["after"] = pr_page["pageInfo"]["endCursor"]
        else:
            break
    return prs

async def fetch_all_commits(owner: str, repo_name: str, token: str) -> list:
    """
    Fetch commits from the default branch of the repository.
    Note: This queries the commit history from the default branch reference.
    """
    commits = []
    query = """
    query ($owner: String!, $name: String!, $after: String) {
      repository(owner: $owner, name: $name) {
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 100, after: $after) {
                pageInfo {
                  hasNextPage
                  endCursor
                }
                nodes {
                  oid
                  message
                  committedDate
                  author {
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    variables = {"owner": owner, "name": repo_name, "after": None}
    while True:
        result = await run_query(query, variables, token)
        history = result["data"]["repository"]["defaultBranchRef"]["target"]["history"]
        commits.extend(history["nodes"])
        if history["pageInfo"]["hasNextPage"]:
            variables["after"] = history["pageInfo"]["endCursor"]
        else:
            break
    return commits

# Example usage:
if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise Exception("Please set the GITHUB_TOKEN environment variable.")
    owner = "octocat"       # Replace with the repository owner's username
    repo_name = "Hello-World"  # Replace with the repository name
    
    print("Fetching issues...")
    issues = fetch_all_issues(owner, repo_name, token)
    print(f"Total issues: {len(issues)}")
    
    print("Fetching pull requests...")
    prs = fetch_all_pull_requests(owner, repo_name, token)
    print(f"Total pull requests: {len(prs)}")
    
    print("Fetching commits...")
    commits = fetch_all_commits(owner, repo_name, token)
    print(f"Total commits: {len(commits)}")
