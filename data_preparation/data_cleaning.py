import re
import pandas as pd
from bs4 import BeautifulSoup

job_posts = pd.read_csv("../data/jobs.csv")
job_posts.dropna(subset=['job_description'], inplace=True)
job_posts.reset_index(drop=True, inplace=True)


def clean_html_text(html_text):
    # Parse the HTML-encoded text
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text(separator=' ')
    clean = re.compile('<.*?>')
    remove_html_tags = re.sub(clean, ' ', text)
    remove_extra_spaces = re.sub(r'\s+', ' ', remove_html_tags).strip()
    return remove_extra_spaces


def lowercase_remove_special(column):
    return column.str.lower().apply(lambda x: re.sub(r'[$&\-.]', '', x))


def split_location(location):
    city, *rest = location.split(', ')
    region, country = rest if len(rest) == 2 else (city, city)
    return pd.Series([city, region, country])


# Clean job descriptions
clean_job_descr = []
for i in range(len(job_posts["job_description"])):
    clean_job_descr.append(clean_html_text(job_posts["job_description"][i]))
job_posts["clean_job_description"] = clean_job_descr

# Apply lowercase function to selected columns
columns_to_lowercase = ['job_position', 'company_name', 'job_location']
job_posts[columns_to_lowercase] = job_posts[columns_to_lowercase].apply(lowercase_remove_special)

# Split location column and create new columns
job_posts[['city', 'region', 'country']] = job_posts['job_location'].apply(split_location)

# Split the job post date into year, month and day.
job_posts[['post_year', 'post_month', 'post_day']] = job_posts['job_posting_date'].str.split('-', expand=True).astype(int)

# Drop the unnecessary columns
job_posts.drop(['job_location', 'job_posting_date', 'job_description'], axis=1, inplace=True)

# Save the DataFrame to a CSV file
job_posts.to_csv('../data/jobs_cleaned.csv', index=False)
print("DataFrame saved to jobs_cleaned.csv")
