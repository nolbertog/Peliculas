import requests
import pandas as pd
import time

# TMDb API Key (replace with your actual key)
api_key = '2154f70f4fd6898749d2d6c4503dd224'

# Base URL for TMDb API
base_url = 'https://api.themoviedb.org/3'

# Function to get movies from TMDb API
def get_movies_from_tmdb(start_year, end_year, api_key):
    all_movies = []

    for year in range(start_year, end_year + 1):
        for page in range(1, 6):  # Limiting to 5 pages per year for this example
            url = f"{base_url}/discover/movie?api_key={api_key}&language=en-US&sort_by=popularity.desc&primary_release_year={year}&page={page}"
            
            for attempt in range(5):  # Retry logic
                try:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = response.json()

                    if 'results' in data:
                        all_movies.extend(data['results'])
                    else:
                        break  # Break if there are no more results

                    time.sleep(0.2)  # To avoid hitting the rate limit
                    break  # Exit retry loop if successful
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2)  # Wait before retrying
                    if attempt == 4:
                        print(f"Failed to retrieve data for year {year}, page {page}")
                        return all_movies

    return all_movies

# Function to extract relevant movie information
def extract_movie_info(movies):
    movie_data = []

    for movie in movies:
        movie_id = movie['id']
        title = movie['title']
        release_date = movie.get('release_date', '')
        genre_ids = movie.get('genre_ids', [])
        
        movie_data.append({
            'movie_id': movie_id,
            'title': title,
            'release_date': release_date,
            'genre_ids': genre_ids
        })
    
    return movie_data

# Function to get genre names from TMDb API
def get_genre_names(api_key):
    url = f"{base_url}/genre/movie/list?api_key={api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()
    
    if 'genres' in data:
        return {genre['id']: genre['name'] for genre in data['genres']}
    else:
        return {}

# Get movies from 1990 to 2022
movies = get_movies_from_tmdb(1990, 2022, api_key)
movie_data = extract_movie_info(movies)

# Get genre names
genre_names = get_genre_names(api_key)

# Convert genre_ids to genre names
for movie in movie_data:
    movie['genres'] = [genre_names.get(genre_id, '') for genre_id in movie['genre_ids']]
    del movie['genre_ids']

# Convert to DataFrame
df = pd.DataFrame(movie_data)

# Save to CSV
df.to_csv('movies.csv', index=False)

print("CSV file 'movies_genres_1990_2022.csv' created successfully.")