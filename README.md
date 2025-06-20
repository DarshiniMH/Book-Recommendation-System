# Hybrid Book Recommendation System

---

## Introduction 

This project presents an **advanced** book recommendation system. While simple approaches might seem appealing, real-world book data presents unique challenges: many books lack direct "similar" links, and traditional keyword searches can miss the nuanced meaning of a story.

Critically, while readily available "similar books" data (often based on human curation or simple heuristics) offers valuable suggestions, it's inherently limited—it can't provide recommendations for every book, especially brand new additions or those lacking pre-existing explicit connections. My system tackles these fundamental challenges head-on, delivering comprehensive, semantically rich, and genre-aligned recommendations for every single book in my collection, seamlessly handling novel items and ensuring the system's future adaptability.

---

## Key Features

My Recommendation System offers a seamless and insightful experience:

* **Interactive Book Search:** Quickly find books by title using a highly optimized, fuzzy-matching search engine.
* **Hybrid Recommendation Engine:** Fuses similarity signals from multiple advanced sources for robust suggestions.
* **Tiered Recommendation Prioritization:** Organizes recommendations based on the reliability and type of similarity (e.g., explicit links, semantic understanding, genre alignment).
* **Comprehensive Coverage:** Provides recommendations for all 149,000 books in the dataset, ensuring no reader is left without a suggestion, even if explicit "similar books" data is missing.
* **Transparent Source Attribution:** Each recommendation clearly indicates which model(s) contributed to its selection, demystifying the "why."
* **User-Friendly Web Application:** Showcased through an interactive Streamlit application.

---

## Application Demonstration

Here's a quick look at the interactive Streamlit application:

![Book Recommendation App Demo](images/book_recommendation_demo.gif)

---

## Getting Started

Follow these instructions to set up the project environment, prepare the data, and run the application.

### Prerequisites

* **Python 3.8+** (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DarshiniMH/Book-Recommendation-System.git]
    cd Book-Recommendation-System
    ```
2.  **Create a virtual environment** (recommended for dependency management):
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(**Note:** `faiss-gpu` installation can be complex. Refer to its official documentation for specific CUDA compatibility and troubleshooting. If issues persist, consider `faiss-cpu` for CPU-only operation.)*

### Data Setup & Preprocessing

The raw dataset (approx. 8GB JSON) and the subsequent processed/indexed files (`.pkl`, `.parquet`, SQLite `.db`, Whoosh index folders) are too large to be stored directly on GitHub. You will need to either download the final preprocessed assets or run the preprocessing script yourself.

#### Option 1: Download Preprocessed Data (Recommended for quick setup)

This is the fastest way to get the app running, as it bypasses the need to run the full preprocessing pipeline.

1.  **Download preprocessed database file** (`hybrid_recommendations.db`)from the following Google Drive link:
    [** Download Processed Project Data from Google Drive **](https://drive.google.com/drive/folders/1OMvP0IwxAF4cgVmrMD6irew1nFYirmWQ?usp=drive_link)


2.  **Create a `data` directory** in the root of your project folder if it doesn't already exist.
    ```bash
    mkdir data
    ```
3.  **Place the downloaded `hybrid_recommendations_db.db` file** inside the `data/` directory. Your project structure should look like this:
    ```
    your-repository-name/
    ├── data/
    │   └── hybrid_recommendations.db  # Your final SQLite DB
    ├── streamlit_app.py                # Your Streamlit app file
    ├── requirements.txt
    └── README.md
    ```

#### Option 2: Run Local Data Preprocessing (For full pipeline demonstration)

This option allows you to run the entire data pipeline from the raw JSON file to generate all necessary assets locally.

1.  **Obtain the raw dataset:**
    * Navigate to the [UCSD Goodreads Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html).
    * Look for the link named "**Detailed book graph (~2gb, about 2.3m books): `goodreads_books.json.gz`**" and download this file.
    * Create a `data/` directory in the root of your project folder if it doesn't already exist:
        ```bash
        mkdir -p data
        ```
    * Place the downloaded `goodreads_books.json.gz` file into the `data/` directory.

2.  **Execute the preprocessing script:**
    * Run dedicated data preprocessing script (e.g., 01-data-preprocessing-and-model-creation.ipynb) which orchestrates all steps from raw JSON to the final `.pkl`, `.npy`, `.db`, and Whoosh index files.
    * *(**Note:** This script will take significant time and computational resources, especially for embedding generation and index building, and requires the necessary Python packages and potentially GPU/TPU access.)*

3.  **Verify Data Structure:**
    After running the script, your project structure should match the one described in Option 1, containing all the generated files within the `data/` directory.

### Running the Application

Once the data setup (either Option 1 or Option 2) is complete:

1.  **Activate your virtual environment:**
    ```bash
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
2.  **Start the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```
    This command will open the application in your web browser.
---


## Technical Architecture & Implementation

This project was a journey through real-world data science problems, demanding robust engineering and clever algorithmic solutions.

### 1. Data Acquisition and Preprocessing

My project began with the ambitious task of harnessing a massive 8GB JSON dataset containing 2.3 million book entries, each packed with nested, inconsistently formatted data. The first challenge was streamlining this raw influx: I meticulously extracted only the necessary fields, transforming heterogeneous data types (e.g., nested dictionaries to lists, strings to integers) into a clean, usable format. Subsequently, I performed a critical pruning phase: only books with an average rating of at least 3.7 and over 100 ratings were retained, reducing the dataset from 2.3 million to a more manageable 270k entries. Further refinement involved a multi-stage deduplication and English language filtering process. I normalized titles, applied regular expressions to strip versioning and subtitles, and used `fastText` to identify and retain only genuinely English-language books. To handle duplicate book entries, I uniquely identified books by a normalized title-author ID combination, consolidating their 'similar_books' information and keeping the most popular version, ultimately trimming the dataset to a clean 149k books.

### 2. Genre Feature Engineering

A major hurdle was the absence of explicit genre information. I tackled this by extracting insights from the 'popular_shelves' field, a rich but incredibly noisy source. This column contained over 896,000 unique, user-generated tags (such as "to-read," "favorites," or "ebooks"), most of which were uninformative for content-based recommendations. To extract meaningful genre signals, I devised a multi-pronged filtering pipeline. First, a frequency cutoff (at 150 occurrences per tag, identified via histogram analysis) drastically reduced the pool to 8,105 unique words, focusing on tags more likely to reflect shared categorization. These terms underwent normalization, followed by matching against a curated genre seed list (exact, synonym, and fuzzy matching) to classify 2,407 highly confident genre terms. For the remaining 5,698 unclassified terms, I leveraged embedding-based clustering: I generated semantic embeddings using an all-MiniLM-L6-v2 Sentence Transformer, then applied HDBSCAN clustering to group semantically similar tags. This allowed me to automatically classify a final set of 5,689 highly informative genre terms, transforming raw user tags into a powerful recommendation feature.

### 3. Initial Approach: User-Item Collaborative Filtering (SVD & FAISS)

My first attempt focused on model-based collaborative filtering, aiming to leverage user-book interaction patterns. I constructed a large, sparse Book-User interaction matrix from rating data and applied Truncated Singular Value Decomposition (SVD) to reduce its dimensionality. SVD decomposed the matrix into lower-dimensional "latent factor" embeddings for each book, representing underlying behavioral patterns. These book embeddings were then indexed using FAISS to enable rapid similarity search. However, this approach faced a significant challenge: user behavior proved too diverse for meaningful similarity discovery in this context. A user who enjoys a philosophical novel might also read a crime thriller or a biography. This broad, overlapping interest meant that the latent factors struggled to delineate clear, semantically cohesive book groups, leading to recommendations that lacked thematic or genre coherence.


![image](https://github.com/user-attachments/assets/dbbb8f7d-68ca-4f94-a574-7e04d3cc716e)
*Figure: Recommendations generated by the SVD-based model for "The Alchemist." As observed, it recommended popular fantasy titles like "The Lord of the Rings" and "Harry Potter," which, despite their popularity, lack semantic or thematic alignment with "The Alchemist's" core philosophical narrative.*

### 4. Content-Based Modeling: Keyword-Centric Approaches

Initially, I explored keyword-based recommendation methods for book descriptions, starting with TF-IDF, then pivoting to BM25 (via `rank_bm25`) due to its more nuanced penalization of term frequency and document length. However, scaling `rank_bm25` for my 149k books proved computationally infeasible ($O(N^2)$ complexity led to over 12 hours of incomplete runtime). I then optimized by integrating Whoosh, which leveraged multi-threaded indexing and a sophisticated internal BM25 implementation, reducing the processing time to approximately 6 hours. Despite this speedup, the recommendation quality was poor: Whoosh's term-based matching struggled with the inherent semantic diversity of book descriptions, finding relevant matches for only ~10% (12,924) of books. It became clear that keyword-only approaches failed to capture the true essence or nuanced meaning of book descriptions, often leading to irrelevant recommendations due to a lack of literal word overlap.

### 5. Content-Based Modeling: Semantic Embeddings (Sentence Transformers & FAISS)

Recognizing the limitations of keyword-based matching, the next crucial step was to build intelligent content-based recommendations by capturing **semantic similarity**. I leveraged **state-of-the-art Sentence Transformers** (specifically the `intfloat/e5-large` model) to generate dense, 1024-dimensional semantic embeddings for each book description. This powerful model transforms text into numerical vectors where meaning, rather than just word overlap, determines proximity. For the 131,000 books with descriptions, these embeddings allowed me to construct highly relevant neighbor lists. The sheer scale of operations (149k books) for embedding generation and search was efficiently managed by utilizing **GPU/TPU batch processing** during encoding and building a **FAISS index** for lightning-fast Approximate Nearest Neighbor (ANN) search. This pivot proved highly successful: the recommendations generated are significantly better, correctly identifying semantically similar books and demonstrating a profound understanding of description content, as further illustrated in Section 7.

### 6. Robust Fallback: Genre-Based Recommendations from Popular Shelves

To ensure comprehensive recommendation coverage for all books, and to serve as a robust fallback for the small subset lacking detailed descriptions, I developed a parallel content-based model utilizing `popular_shelves` data. Almost all books in the collection had this information available. After a meticulous multi-stage process to extract "clean" genre tags from the raw, noisy shelf data, these processed genres were transformed into a binary feature matrix using `MultiLabelBinarizer`. Finally, a FAISS index was constructed on these genre vectors, enabling the efficient generation of neighbor lists. While the recommendations from this genre-based model are inherently more general compared to the nuanced semantic insights from descriptions, they consistently provide relatable and valuable suggestions, ensuring every book receives a recommendation, as demonstrated in Section 7.

### 7. Performance Evaluation & Hybrid System Demonstration

The true strength of this project lies in its **hybrid fusion strategy**, which intelligently combines these diverse recommendation signals. My system prioritizes explicit `similar_books` connections, gracefully falls back to nuanced semantic similarity from descriptions, and then leverages broader genre alignment from popular shelves. This multi-tiered approach ensures both precision and breadth.

To illustrate this, let's examine the recommendations for **"Born a Crime: Stories From a South African Childhood" (ID: 2978025)**, a poignant and humorous memoir.

#### 7.1. Curated Similar Books (Benchmark Data)

*Here, I attempted to retrieve recommendations from the pre-existing 'similar_books' data for "Born a Crime."*

![image](https://github.com/user-attachments/assets/5d809b71-813a-4a88-addf-9b9dee0ccf73)
*Figure: As seen, no 'similar_books' data was found for this particular title in the dataset. This highlights a critical limitation of relying solely on external curation and directly showcases the real-world problem my system is built to solve.*

#### 7.2. Semantic Description Recommendations

*Despite the absence of explicit 'similar_books' data, my model leverages state-of-the-art Sentence Transformers to understand the deep thematic and emotional content of book descriptions, providing relevant recommendations.*

![image](https://github.com/user-attachments/assets/853efb37-4183-4eb2-b118-3880887128f5)
*Figure: This model successfully identifies books like "Born a Crime and Other Stories" (a companion work), "Kaffir Boy" (a classic memoir on growing up Black in apartheid South Africa), "Nobody's Son," and "Believe Me." It demonstrates exceptional semantic understanding, capturing the blend of memoir, humor, and themes of race/identity present in "Born a Crime," proving its ability to provide valuable recommendations even for books lacking traditional links.*

#### 7.3. Genre-Based Recommendations

*This model utilizes a meticulously filtered set of genre tags to identify books sharing similar categorical profiles, serving as a powerful alternative when other data is scarce.*

![image](https://github.com/user-attachments/assets/0a974e06-4c34-4fe2-bca3-100c51fc98ce)
*Figure: This model also yields highly relevant results, hitting books by other prominent comedians addressing similar themes of race and identity with humor, such as "The Awkward Thoughts of W. Kamau Bell," "The Misadventures of Awkward Black Girl," and "You Can't Touch My Hair." The presence of "Dreams from My Father" further reinforces its strength in capturing deeply aligned memoirs. This demonstrates strong categorical and stylistic alignment, often feeling even more precise for specific stylistic elements like humor and social commentary among memoirs, complementing the semantic description model.*

---

## Conclusion & Future Directions

My hybrid recommendation system has demonstrated its power across various facets of book similarity. By combining validated semantic and genre-based content understanding with a robust fusion strategy, I can confidently provide highly relevant recommendations for every single book in the dataset, even those completely lacking pre-existing 'similar_books' data. This comprehensive capability negates the future dependency on manual curation for "similar books" links; my models can effectively learn, adapt, and provide ongoing recommendations for new and existing titles alike, requiring minimal human intervention in the ongoing recommendation process.
