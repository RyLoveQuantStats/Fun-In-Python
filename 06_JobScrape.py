#!/usr/bin/env python3
import requests
import smtplib
import time
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from constants import JSEARCH_API, EMAIL, EMAIL_PASSWORD

# -------------------
# Configuration
# -------------------

# API configuration (update with your own API key)
API_KEY = JSEARCH_API  # Replace with your own API key
API_HOST = "jsearch.p.rapidapi.com"
BASE_URL = "https://jsearch.p.rapidapi.com"
HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-randomapi-host": API_HOST,  # not used but provided by documentation; keep for consistency
    "x-rapidapi-host": API_HOST
}

# Job search parameters
# List of search queries to cover finance, trading, portfolio management, investment, etc.
SEARCH_QUERIES = [
    "finance",
    "trading",
    "portfolio management",
    "investment",
    "financial_data_analyst",
    "asset_management",
    "hedge fund",
    "equity research",
    "investment banking",
    "quantitative analyst",
    "financial advisor",
    "risk management",
]
SEARCH_LOCATION = "Denver, Colorado", "San Diego, California", "Chicago, Illinois", "Missouri"  # Adjust location if needed (e.g., city, state)
NUM_PAGES = 1          # Number of pages to retrieve per query

# Email settings (update with your own email configuration)
SENDER_EMAIL = EMAIL  # Your email address
SENDER_PASSWORD = EMAIL_PASSWORD  # Your email password
SMTP_SERVER = "smtp.gmail.com"             # e.g., Gmail SMTP
SMTP_PORT = 587
RECIPIENT_EMAIL = EMAIL  # Where job alerts will be sent

# Retry settings for API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# -------------------
# Helper: API call with Retry
# -------------------
def call_api(endpoint, params):
    """Make an API call with retry logic on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL + endpoint, headers=HEADERS, params=params)
            # Check for HTTP-level errors (like 429, 500, etc.)
            if response.status_code == 429:
                print("Rate limit hit (429). Waiting for", RETRY_DELAY, "seconds...")
                time.sleep(RETRY_DELAY)
                continue
            elif response.status_code != 200:
                print(f"Error: Received status code {response.status_code} for params: {params}")
                return None
            return response
        except Exception as e:
            print("Exception during API call:", e)
            time.sleep(RETRY_DELAY)
    return None

# -------------------
# Job Search Functions
# -------------------

def search_jobs(query, location=SEARCH_LOCATION, num_pages=NUM_PAGES):
    """
    Search for jobs using the JSearch API.
    
    The API endpoint is GET /search. It returns a JSON with a 'status' field.
    In case of an error, the API returns a JSON with "status": "ERROR" and an "error" object.
    """
    endpoint = "/search"
    all_jobs = []
    
    for page in range(1, num_pages + 1):
        params = {
            "query": query,
            "page": page,
            "num_pages": num_pages,
            "location": location
        }
        response = call_api(endpoint, params)
        if response is None:
            print(f"Failed to get results for query '{query}', page {page}.")
            continue
        
        try:
            json_response = response.json()
        except Exception as e:
            print("Error decoding JSON:", e)
            continue

        # Check API response structure
        if json_response.get("status") != "OK":
            error_message = json_response.get("error", {}).get("message", "Unknown error")
            print(f"API error for query '{query}', page {page}: {error_message}")
            continue

        jobs = json_response.get("data", [])
        all_jobs.extend(jobs)
    return all_jobs

def get_job_details(job_id, country="us"):
    """
    Retrieve detailed info for a specific job using GET /job-details.
    
    This endpoint returns additional information including employer reviews and salary estimates.
    """
    endpoint = "/job-details"
    params = {
        "job_id": job_id,
        "country": country
    }
    response = call_api(endpoint, params)
    if response is None:
        print(f"Failed to get details for job_id {job_id}")
        return None
    
    try:
        json_response = response.json()
    except Exception as e:
        print("Error decoding JSON for job details:", e)
        return None

    if json_response.get("status") != "OK":
        error_message = json_response.get("error", {}).get("message", "Unknown error")
        print(f"API error for job details (job_id: {job_id}): {error_message}")
        return None

    return json_response.get("data", {})

# -------------------
# Email Function
# -------------------

def send_email(subject, body, recipient_email=RECIPIENT_EMAIL):
    """
    Send an HTML email with the given subject and body.
    """
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(message)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print("Error sending email:", e)

# -------------------
# Main Script Logic
# -------------------

def main():
    print("Starting job search for finance/trading/portfolio management/investment positions...")
    overall_jobs = {}  # Dictionary to hold jobs per query
    
    for query in SEARCH_QUERIES:
        print(f"Searching for jobs with query: '{query}'")
        jobs = search_jobs(query)
        overall_jobs[query] = jobs
        print(f"Found {len(jobs)} jobs for query '{query}'")
    
    # Build an HTML email body with job listings for each query
    email_body = "<h1>Weekly Job Listings Alert</h1>"
    for query, jobs in overall_jobs.items():
        email_body += f"<h2>Results for '{query}'</h2>"
        if not jobs:
            email_body += "<p>No jobs found.</p>"
            continue
        
        email_body += "<ul>"
        for job in jobs:
            title = job.get("job_title", "N/A")
            company = job.get("employer_name", "N/A")
            # Optionally include location details (the field names might vary)
            city = job.get("job_city", "")
            country = job.get("job_country", "")
            location = f"{city}, {country}" if city or country else "Location not provided"
            job_id = job.get("job_id", "")
            
            # Construct a URL to view job details via the API (or direct users to your custom page)
            details_url = f"{BASE_URL}/job-details?job_id={job_id}&country={SEARCH_LOCATION}"
            email_body += (
                f"<li><strong>{title}</strong> at {company}<br>"
                f"{location}<br>"
                f"<a href='{details_url}'>View Job Details</a></li><br>"
            )
        email_body += "</ul>"
    
    subject = "Weekly Job Alert: Finance, Trading, Portfolio Management & Investment"
    send_email(subject, email_body)

if __name__ == "__main__":
    main()
