import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

parameters = {
    "api_key": os.getenv('API_KEY'),
    'geoid': os.getenv('GEO_ID'),
    'field': os.getenv('FIELD').replace(" ", "%20"),
    'page': 1,
    'start': 2
}
jobs = []
base_url = "https://api.scrapingdog.com/linkedinjobs/"

# Create an empty list to store all jobs
all_jobs = []

while True:
    # Send a GET request with the parameters
    response = requests.get(base_url, params=parameters)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        print(f"Got data from the page {parameters['page']}")

        # Append the retrieved jobs to the list
        all_jobs.extend(data)

        # Check if the returned data contains the maximum number of items per page
        if len(data) == 25:  # Assuming 25 items per page
            # Increment the page number to retrieve the next page of results
            parameters['page'] += 1
        else:
            # No more pages, break out of the loop
            break
    else:
        print("Request failed with status code:", response.status_code)
        break

for job in all_jobs:
    job_clean = {"job_id": job["job_id"], "job_position": job["job_position"], "company_name": job["company_name"],
                 "job_location": job["job_location"], "job_posting_date": job["job_posting_date"]}
    job_link = job['job_link']
    job_response = requests.get(job_link)

    # Check if the request was successful
    if job_response.status_code == 200:
        # Parse the HTML content of the job description page
        soup = BeautifulSoup(job_response.text, 'html.parser')

        # Find the script tag containing the JSON-LD data
        script_tag = soup.find('script', type='application/ld+json')

        if script_tag:
            # Extract the JSON-LD data from the script tag
            json_data = json.loads(script_tag.string)

            # Extract the job description and company description from the JSON data
            job_description = json_data.get('description', 'Description not found')

            # Update the job dictionary with the descriptions
            job_clean['job_description'] = job_description

        else:
            print(f"Script tag not found for {job['job_position']}")
        jobs.append(job_clean)
    else:
        print(f"Failed to fetch job description for {job['job_position']}")

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(jobs)
# Save the DataFrame to a CSV file
df.to_csv('../data/jobs.csv', index=False)
print("DataFrame saved to jobs.csv")
