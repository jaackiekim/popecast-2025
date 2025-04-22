import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

# Load the Wikipedia page
URL = "https://en.wikipedia.org/wiki/List_of_living_cardinals"
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

# Grab the wikitable
table = soup.find("table", {"class": "wikitable"})
if not table:
    raise Exception("❌ Could not find the cardinal table.")

cardinals = []

# Loop through each row (skip header)
for row in table.find_all("tr")[1:]:
    cells = row.find_all("td")
    if len(cells) < 5:
        continue

    try:
        name_cell = cells[1]
        name = name_cell.get_text(strip=True).replace("*", "")
        link = name_cell.find("a")
        bio_url = "https://en.wikipedia.org" + link["href"] if link else ""

        country = cells[2].get_text(strip=True)

        birth_cell = cells[3].get_text(strip=True)
        match = re.search(r"(\d{1,2} \w+ \d{4})", birth_cell)
        birthdate_raw = match.group() if match else birth_cell

        try:
            birthdate = datetime.strptime(birthdate_raw, "%d %B %Y")
            age = (datetime.now() - birthdate).days // 365
        except:
            age = None

        current_role = cells[4].get_text(strip=True)

        cardinals.append({
            "name": name,
            "bio_url": bio_url,
            "birthdate_raw": birthdate_raw,
            "age": age,
            "country": country,
            "current_role": current_role
        })

    except Exception as e:
        print(f"❌ Error parsing row: {e}")
        continue

# Save to CSV
df = pd.DataFrame(cardinals)
df.to_csv("data/cardinals_raw.csv", index=False)
print("✅ Done! Cardinal data saved to data/cardinals_raw.csv")
