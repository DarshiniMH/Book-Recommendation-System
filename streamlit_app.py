import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
from rapidfuzz import process # You need to import this for fuzzy matching

# --- Set Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Book Recommendation System", layout="wide")

# Path to the NEW database with pre-computed neighbors
BOOKS_DB_PATH_WITH_RECS = "data/hybrid_recommendations.db"

@st.cache_data
def load_app_data():
    conn = None
    try:
        conn = sqlite3.connect(BOOKS_DB_PATH_WITH_RECS)
        books_df = pd.read_sql("""
            SELECT
                book_id,
                title,              -- Use 'title' as it's the renamed/processed column
                average_rating,
                ratings_count,
                similar_books_json,
                top_desc_neighbors_ids_json,
                top_shelf_neighbors_ids_json
            FROM books
        """, conn)

        # Post-process JSON columns back into Python lists
        books_df['similar_books_filtered'] = books_df['similar_books_json'].apply(json.loads)
        books_df['top_desc_neighbors_ids'] = books_df['top_desc_neighbors_ids_json'].apply(json.loads)
        books_df['top_shelf_neighbors_ids'] = books_df['top_shelf_neighbors_ids_json'].apply(json.loads)

        books_df = books_df.drop(columns=[
            'similar_books_json', 'top_desc_neighbors_ids_json', 'top_shelf_neighbors_ids_json'
        ])
        st.success("Main books data with precomputed neighbors loaded.")
        return books_df # Ensure you return the DataFrame

    except FileNotFoundError as e:
        st.error(f"Error loading main database: {e}. Ensure '{BOOKS_DB_PATH_WITH_RECS}' is correct.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading data from DB: {e}")
        st.stop()
    finally:
        if conn: conn.close()


books_df = load_app_data()



@st.cache_data # Cache search results
def search_books_streamlit(title_query_str: str, conn_path: str, books_df_full: pd.DataFrame, top_n: int = 5):
    conn = None
    conn = sqlite3.connect(conn_path)
    cursor = conn.cursor()

    fts5_query_str = f'"{title_query_str}"' 

        # 1. Direct FTS5 search (using title as indexed)
    sql_fts = """
        SELECT b.book_id, b.title, b.ratings_count, b.average_rating -- Removed b.authors
        FROM books_fts fts
        JOIN books b ON b.rowid = fts.rowid
        WHERE books_fts MATCH ?
        ORDER BY b.ratings_count DESC
        LIMIT ?;
    """
    cursor.execute(sql_fts, (fts5_query_str, top_n)) # Pass top_n to LIMIT
    fts_results = cursor.fetchall()

    formatted_results = []
    # Process FTS5 results - now (book_id, title, ratings_count, average_rating)
    for book_id, title, ratings_count, average_rating in fts_results:
        formatted_results.append((book_id, title, ratings_count, average_rating))
        
    if formatted_results:
        return formatted_results
    else:
        cursor.execute("SELECT title FROM books;")
        all_titles = [row[0] for row in cursor.fetchall()]
        fuzzy_matches = process.extract(title_query_str, all_titles, limit=top_n)
        matched_titles = [match[0] for match in fuzzy_matches]

        query = f"""
            SELECT book_id, title, ratings_count, average_rating
            FROM books
            WHERE title IN ({','.join(['?']*len(matched_titles))})
            ORDER BY ratings_count DESC
            LIMIT ?;
        """
        cursor.execute(query, (*matched_titles, top_n))
        results = cursor.fetchall()
    conn.close()

    return results





# --- Recommendation Fusion Function (SIMPLIFIED & OPTIMIZED) ---
def fuse_neighbors_streamlit(query_book_id: int, k: int = 10):
    query_book_row = books_df[books_df['book_id'] == query_book_id]
    if query_book_row.empty:
        st.warning(f"Query book ID {query_book_id} not found in main DataFrame for recommendation fusion.")
        return pd.DataFrame()
    query_book_row = query_book_row.iloc[0] # Get the single row (as a Series)

    desc_list_ordered = query_book_row['top_desc_neighbors_ids']
    shelf_list_ordered = query_book_row['top_shelf_neighbors_ids']
    
    fused_recommendations_ids = []
    fused_recommendations_sources = []
    added_ids_set = set() # To ensure uniqueness

    # Tier 1: Fill from Description
    for bid in desc_list_ordered:
        if bid not in added_ids_set:
            fused_recommendations_ids.append(bid)
            fused_recommendations_sources.append("Source: Description")
            added_ids_set.add(bid)
            if len(fused_recommendations_ids) >= k:
                break
    
    # Tier 2: Fill remaining spots from Popular Shelves
    if len(fused_recommendations_ids) < k:
        for bid in shelf_list_ordered:
            if bid not in added_ids_set:
                fused_recommendations_ids.append(bid)
                fused_recommendations_sources.append("Source: Popular Shelves")
                added_ids_set.add(bid)
                if len(fused_recommendations_ids) >= k:
                    break

    if not fused_recommendations_ids:
        return pd.DataFrame()

    matched_df = books_df[books_df["book_id"].isin(fused_recommendations_ids)].copy()

    rank_map = {bid: i for i, bid in enumerate(fused_recommendations_ids)}
    matched_df["_rank"] = matched_df["book_id"].map(rank_map)
    matched_df = matched_df.sort_values("_rank").reset_index(drop=True)

    source_map = {fused_recommendations_ids[i]: fused_recommendations_sources[i]
                  for i in range(len(fused_recommendations_ids))}
    matched_df["source"] = matched_df["book_id"].map(source_map)

    # Return final desired columns, now without "authors"
    return matched_df[[
        "title", "average_rating", "ratings_count", "book_id", "source"
    ]]


def main():
    st.title("📚 Book Recommendation System")

    st.markdown(
        """
        1. Type (or paste) a partial/complete title in the search box below.
        2. Select the correct match from the dropdown.
        3. View fused recommendations based on Description and Popular Shelves.
        """
    )

    # --- Search Section ---
    search_title = st.text_input("Search for a Book Title", key="title_query")

    # Initialize session state for search results and selection
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_book_id' not in st.session_state:
        st.session_state.selected_book_id = None
    if 'selected_book_title' not in st.session_state:
        st.session_state.selected_book_title = None
    
    # --- Button Logic (Triggers Search) ---
    if st.button("Get Recommendations"): # Button to trigger the search
        if search_title:
            with st.spinner(f"Searching for '{search_title}'..."):
                matched_books_results = search_books_streamlit(search_title, BOOKS_DB_PATH_WITH_RECS, books_df, top_n=5)
                st.session_state.search_results = matched_books_results # Store results in session state
                st.session_state.selected_book_id = None # Reset selected book on new search
                st.session_state.selected_book_title = None

            if not matched_books_results:
                st.warning(f" No books found matching '{search_title}'. Try a different title or spelling.")
        else:
            st.info("Please enter a book title to start.")

    # --- Display Search Results and Selection (Conditional on search results being present) ---
    if st.session_state.search_results:
        st.subheader("Top Matches for Your Search:")

        # Prepare options for st.selectbox: List of (book_id, title, ratings_count, average_rating) tuples
        selectbox_value_options = [tup[0] for tup in st.session_state.search_results] # Book_ids

        # This line is now much simpler as no authors are involved
        selectbox_display_options = [tup[1] for tup in st.session_state.search_results] # Display just the title
        
        default_index = 0
        if st.session_state.selected_book_id in selectbox_value_options:
            default_index = selectbox_value_options.index(st.session_state.selected_book_id)
        
        selected_book_id_from_selectbox = st.selectbox(
            "Select a book:",
            options=selectbox_value_options,
            format_func=lambda x: selectbox_display_options[selectbox_value_options.index(x)],
            index=default_index,
            key='book_selection_selectbox'
        )
        
        # --- Recommendation Generation and Display (Conditional on selection) ---
        if selected_book_id_from_selectbox is not None:
            # Update session state with the current selected book (from selectbox)
            current_selected_book_row = books_df[books_df['book_id'] == selected_book_id_from_selectbox].iloc[0]
            st.session_state.selected_book_id = current_selected_book_row['book_id']
            st.session_state.selected_book_title = current_selected_book_row['title']
            
            if st.session_state.selected_book_id is not None:
                st.write(f"**Selected Book:** {st.session_state.selected_book_title} (ID: {st.session_state.selected_book_id})")

                with st.spinner(f"Generating recommendations for '{st.session_state.selected_book_title}'..."):
                    recommendations_df = fuse_neighbors_streamlit(st.session_state.selected_book_id, k=10) # Get top 10

                if not recommendations_df.empty:
                    st.subheader("Recommended Books:")
                    st.dataframe(recommendations_df)
                else:
                    st.warning(f"No recommendations found for '{st.session_state.selected_book_title}'.")
        else:
            st.info("Please select a book from the matches to get recommendations.")

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Pandas, NumPy, SQLite, RapidFuzz, Scikit-learn, and FAISS.")

if __name__ == "__main__":
    main() # Call the main function when script is executed
