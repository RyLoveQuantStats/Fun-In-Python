#!/usr/bin/env python3
import os
import time
import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from constants import JSEARCH_API, EMAIL, EMAIL_PASSWORD

# -------------------
# JSearch API Configuration
# -------------------
API_HOST = "jsearch.p.rapidapi.com"
BASE_URL = "https://jsearch.p.rapidapi.com"
HEADERS = {
    "x-rapidapi-key": JSEARCH_API,
    "x-rapidapi-host": API_HOST
}

# -------------------
# Job Search Parameters
# -------------------
SEARCH_QUERIES = [
    "quantitative trading"
]
SEARCH_LOCATION = "us"
NUM_PAGES = 1

# -------------------
# SendGrid Email Configuration
# -------------------
SENDGRID_SENDER_EMAIL = EMAIL  # Must be verified in SendGrid
RECIPIENT_EMAIL = EMAIL  # Update with your desired recipient email

# -------------------
# Retry Configuration
# -------------------
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# -------------------
# Helper Functions
# -------------------
def call_api(endpoint, params):
    """
    Makes an API call with retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL + endpoint, headers=HEADERS, params=params)
            if response.status_code == 429:
                print("Rate limit hit (429). Waiting for {} seconds...".format(RETRY_DELAY))
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

def search_jobs(query, location=SEARCH_LOCATION, num_pages=NUM_PAGES):
    """
    Searches for jobs using the JSearch API's /search endpoint.
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

        if json_response.get("status") != "OK":
            error_message = json_response.get("error", {}).get("message", "Unknown error")
            print(f"API error for query '{query}', page {page}: {error_message}")
            continue

        jobs = json_response.get("data", [])
        all_jobs.extend(jobs)
    return all_jobs

def send_email(subject, html_content, recipient_email=RECIPIENT_EMAIL):
    """
    Sends an email using SendGrid.
    """
    message = Mail(
        from_email=SENDGRID_SENDER_EMAIL,
        to_emails=recipient_email,
        subject=subject,
        html_content=html_content
    )
    try:
        sg = SendGridAPIClient(os.environ.get("SENDGRID_API_KEY"))
        response = sg.send(message)
        print("Email sent! Status code:", response.status_code)
    except Exception as e:
        print("Error sending email via SendGrid:", e)

# -------------------
# Main Function
# -------------------
def main():
    print("Starting job search for finance/trading/portfolio management/investment positions...")
    overall_jobs = {}

    # Search for each query and store the results
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
            city = job.get("job_city", "")
            country = job.get("job_country", "")
            location = f"{city}, {country}" if city or country else "Location not provided"
            job_id = job.get("job_id", "")
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
